# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import math
from typing import List, Optional, Dict, Tuple
from pathlib import Path

import torch
from torch import nn
from torch._C import has_cudnn
import fairseq
from fairseq.models import (FairseqEncoder, FairseqEncoderDecoderModel,
                            FairseqIncrementalDecoder, register_model,
                            register_model_architecture)
from fairseq import checkpoint_utils, utils
from fairseq.modules import (
    TransformerEncoderLayer, TransformerDecoderLayer
)
from fairseq.models.text_to_speech.tacotron2 import Prenet, Postnet
from fairseq.modules import LayerNorm, PositionalEmbedding, FairseqDropout
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq import utils
from fairseq.models.transformer import TransformerDecoder

from torch import Tensor


logging.basicConfig(
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def encoder_init(m):
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight, torch.nn.init.calculate_gain("relu"))


def Embedding(num_embeddings, embedding_dim):
    m = nn.Embedding(num_embeddings, embedding_dim)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    return m


class Conv1dSubsampler(nn.Module):
    """Convolutional subsampler: a stack of 1D convolution (along temporal
    dimension) followed by non-linear activation via gated linear units
    (https://arxiv.org/abs/1911.08460)

    Args:
        in_channels (int): the number of input channels
        mid_channels (int): the number of intermediate channels
        out_channels (int): the number of output channels
        kernel_sizes (List[int]): the kernel size for each convolutional layer
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = (3, 3),
    ):
        super(Conv1dSubsampler, self).__init__()
        self.n_layers = len(kernel_sizes)
        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                in_channels if i == 0 else mid_channels // 2,
                mid_channels if i < self.n_layers - 1 else out_channels * 2,
                k,
                stride=2,
                padding=k // 2,
            )
            for i, k in enumerate(kernel_sizes)
        )

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        for _ in range(self.n_layers):
            out = ((out.float() - 1) / 2 + 1).floor().long()
        return out

    def forward(self, src_tokens, src_lengths):
        bsz, in_seq_len, _ = src_tokens.size()  # B x T x (C x D)
        x = src_tokens.transpose(1, 2).contiguous()  # -> B x (C x D) x T
        for conv in self.conv_layers:
            x = conv(x)
            x = nn.functional.glu(x, dim=1)
        _, _, out_seq_len = x.size()
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # -> T x B x (C x D)
        return x, self.get_out_seq_lens_tensor(src_lengths)


class S2STTransformerEncoder(FairseqEncoder):
    """Speech-to-speech translation Transformer encoder that consists of input subsampler and
    Transformer encoder."""

    def __init__(self, args, embed_speaker):
        super().__init__(None)

        #self.encoder_freezing_updates = args.encoder_freezing_updates
        self.num_updates = 0
        self.middle_layers = list([int(k) for k in args.middle_layers.split(",")]) 
        self.dropout_module = FairseqDropout(
            p=args.dropout, module_name=self.__class__.__name__
        )
        self.embed_scale = math.sqrt(args.encoder_embed_dim)
        if args.no_scale_embedding:
            self.embed_scale = 1.0
        self.padding_idx = 1

        self.subsample = Conv1dSubsampler(
            args.input_feat_per_channel * args.input_channels,
            args.conv_channels,
            args.encoder_embed_dim,
            [int(k) for k in args.conv_kernel_sizes.split(",")],
        )

        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, args.encoder_embed_dim, self.padding_idx
        )

        self.transformer_layers = nn.ModuleList(
            [TransformerEncoderLayer(args) for _ in range(args.encoder_transformer_layers)]
        )
        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)
        else:
            self.layer_norm = None

    def _forward(self, src_tokens, src_lengths, return_all_hiddens=False):
        x, input_lengths = self.subsample(src_tokens, src_lengths)
        x = self.embed_scale * x

        encoder_padding_mask = lengths_to_padding_mask(input_lengths)
        positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
        x += positions
        x = self.dropout_module(x)

        encoder_states = []
        out_middle_layers = []

        for id,layer in enumerate(self.transformer_layers):
            x = layer(x, encoder_padding_mask)
            if id in self.middle_layers:
                out_middle_layers.append(x)
            if return_all_hiddens:
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)


        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask] if encoder_padding_mask.any() else [],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "out_middle_layers": out_middle_layers,
            "src_tokens": [],
            "src_lengths": input_lengths,
        }

    def forward(self, src_tokens, src_lengths, speaker,return_all_hiddens=False, **kwargs):
        #if self.num_updates < self.encoder_freezing_updates:
        #     with torch.no_grad():
        #         x = self._forward(src_tokens, src_lengths,
        #                           return_all_hiddens=return_all_hiddens)
        # else:
        x = self._forward(src_tokens, src_lengths,
                            return_all_hiddens=return_all_hiddens)
        return x

    def reorder_encoder_out(self, encoder_out, new_order):
        new_encoder_out = (
            [] if len(encoder_out["encoder_out"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["encoder_out"]]
        )

        new_encoder_padding_mask = (
            [] if len(encoder_out["encoder_padding_mask"]) == 0
            else [x.index_select(0, new_order) for x in encoder_out["encoder_padding_mask"]]
        )

        new_encoder_embedding = (
            [] if len(encoder_out["encoder_embedding"]) == 0
            else [x.index_select(0, new_order) for x in encoder_out["encoder_embedding"]]
        )

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],  # B x T
            "src_lengths": [],  # B x 1
        }

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.num_updates = num_updates



def decoder_init(m):
    if isinstance(m, torch.nn.Conv1d):
        nn.init.xavier_uniform_(m.weight, torch.nn.init.calculate_gain("tanh"))


class S2STTransformerDecoder(FairseqIncrementalDecoder):
    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(None)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self._future_mask = torch.empty(0)

        self.args = args
        self.middle_layers_decoder = list([int(k) for k in args.middle_layers_decoder.split(",")])
        self.padding_idx = src_dict.pad()
        self.n_frames_per_step = args.n_frames_per_step
        self.out_dim = args.output_frame_dim * args.n_frames_per_step

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.embed_positions = PositionalEmbedding(
            args.max_target_positions, args.decoder_embed_dim, self.padding_idx
        )
        self.pos_emb_alpha = nn.Parameter(torch.ones(1))
        self.prenet = nn.Sequential(
            Prenet(self.out_dim, args.prenet_layers, args.prenet_dim,
                   args.prenet_dropout),
            nn.Linear(args.prenet_dim, args.decoder_embed_dim),
        )

        self.n_transformer_layers = args.decoder_transformer_layers
        self.transformer_layers = nn.ModuleList(
            TransformerDecoderLayer(args)
            for _ in range(self.n_transformer_layers)
        )
        if args.decoder_normalize_before:
            self.layer_norm = LayerNorm(args.decoder_embed_dim)
        else:
            self.layer_norm = None

        self.feat_proj = nn.Linear(args.decoder_embed_dim, self.out_dim)
        self.eos_proj = nn.Linear(args.decoder_embed_dim, 1)

        self.postnet = Postnet(self.out_dim, args.postnet_conv_dim,
                               args.postnet_conv_kernel_size,
                               args.postnet_layers, args.postnet_dropout)

        self.ctc_proj = None
        self.ctc_proj_tgt = None
        if getattr(args, "ctc_weight", 0.) > 0.:
            self.ctc_proj = nn.Linear(args.encoder_embed_dim, len(src_dict))
            logging.info(self.ctc_proj)
        if getattr(args, "ctc_weight_tgt", 0.) > 0.:
            self.ctc_proj_tgt = nn.Linear(args.decoder_embed_dim, len(tgt_dict))
            logging.info(self.ctc_proj_tgt)

        self.apply(decoder_init)

    def extract_features(
            self, prev_outputs, encoder_out=None, incremental_state=None,
            target_lengths=None, speaker=None, **kwargs
    ):
        alignment_layer = self.n_transformer_layers - 1
        self_attn_padding_mask = lengths_to_padding_mask(target_lengths)
        positions = self.embed_positions(
            self_attn_padding_mask, incremental_state=incremental_state
        )

        if incremental_state is not None:
            prev_outputs = prev_outputs[:, -1:, :]
            self_attn_padding_mask = self_attn_padding_mask[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        x = self.prenet(prev_outputs)
        x += self.pos_emb_alpha * positions
        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        if not self_attn_padding_mask.any():
            self_attn_padding_mask = None

        attn: Optional[torch.Tensor] = None
        inner_states: List[Optional[torch.Tensor]] = [x]
        out_middle_layers_decoder = []
        for idx, transformer_layer in enumerate(self.transformer_layers):
            if incremental_state is None:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = transformer_layer(
                x,
                encoder_out["encoder_out"][0]
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                else None,
                encoder_out["encoder_padding_mask"][0]
                if (
                        encoder_out is not None
                        and len(encoder_out["encoder_padding_mask"]) > 0
                )
                else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if idx in self.middle_layers_decoder:
                out_middle_layers_decoder.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            # average probabilities over heads, transpose to
            # (B, src_len, tgt_len)
            attn = attn.mean(dim=0).transpose(2, 1)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, {"attn": attn, "inner_states": inner_states, "out_middle_layers_decoder": out_middle_layers_decoder}

    def forward(self, prev_output_tokens, encoder_out=None,
                incremental_state=None, target_lengths=None, speaker=None,
                **kwargs):
        x, extra = self.extract_features(
            prev_output_tokens, encoder_out=encoder_out,
            incremental_state=incremental_state, target_lengths=target_lengths,
            speaker=speaker, **kwargs
        )
        attn = extra["attn"]
        feat_out = self.feat_proj(x)
        bsz, seq_len, _ = x.size()
        eos_out = self.eos_proj(x)
        post_feat_out = feat_out + self.postnet(feat_out)
        return post_feat_out, eos_out, {"attn": attn, "feature_out": feat_out, "out_middle_layers": encoder_out["out_middle_layers"], "out_middle_layers_decoder": extra["out_middle_layers_decoder"]}

    def get_normalized_probs(self, net_output, log_probs, sample, tag="ctc"):
        if tag == "ctc_tgt":
            logits = self.ctc_proj_tgt(net_output[2][0].transpose(0,1))
        else:
            logits = self.ctc_proj(net_output[2][0].transpose(0,1))
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]


class TransformerDecoderScriptable(TransformerDecoder):
    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        # call scriptable method from parent class
        x, _ = self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )
        return x, None


@register_model("s2st_transformer_mtl")
class S2STTransformerModel(FairseqEncoderDecoderModel):
    """
    Implementation for https://arxiv.org/pdf/1809.08895.pdf
    """

    @staticmethod
    def add_args(parser):
        parser.add_argument("--dropout", type=float)
        parser.add_argument("--output-frame-dim", type=int)
        parser.add_argument("--speaker-embed-dim", type=int)
        # output middle hidden state for ctc loss
        parser.add_argument("--middle-layers", default="6", type=str)
        # output middle hidden state for ctc loss for target language
        parser.add_argument("--middle-layers-decoder", default="6", type=str)
        # encoder prenet
        parser.add_argument("--encoder-dropout", type=float)
        parser.add_argument("--encoder-conv-layers", type=int)
        parser.add_argument("--encoder-conv-kernel-size", type=int)
        # Convolutional subsampler
        parser.add_argument("--conv_kernel-sizes", default="5,5", type=str)
        parser.add_argument("--conv-channels", default=1024, type=int)
        parser.add_argument("--input-feat-per-channel", default=80, type=int)
        parser.add_argument("--input_channels", default=1, type=int)
        # encoder transformer layers
        parser.add_argument("--encoder-transformer-layers", type=int)
        parser.add_argument("--encoder-embed-dim", type=int)
        parser.add_argument("--encoder-ffn-embed-dim", type=int)
        parser.add_argument("--encoder-normalize-before", action="store_true")
        parser.add_argument("--encoder-attention-heads", type=int)
        parser.add_argument("--attention-dropout", type=float)
        parser.add_argument("--activation-dropout", "--relu-dropout", type=float)
        parser.add_argument("--activation-fn", type=str, default="relu")
        parser.add_argument("--no_scale_embedding", default=False)
        # decoder prenet
        parser.add_argument("--prenet-dropout", type=float)
        parser.add_argument("--prenet-layers", type=int)
        parser.add_argument("--prenet-dim", type=int)
        # decoder postnet
        parser.add_argument("--postnet-dropout", type=float)
        parser.add_argument("--postnet-layers", type=int)
        parser.add_argument("--postnet-conv-dim", type=int)
        parser.add_argument("--postnet-conv-kernel-size", type=int)
        # decoder transformer layers
        parser.add_argument("--decoder-transformer-layers", type=int)
        parser.add_argument("--decoder-embed-dim", type=int)
        parser.add_argument("--decoder-ffn-embed-dim", type=int)
        parser.add_argument("--decoder-normalize-before", action="store_true")
        parser.add_argument("--decoder-attention-heads", type=int)

        # loading pretraining model
        parser.add_argument(
            "--load-pretrained-encoder-from",
            type=str,
            metavar="STR",
            help="model to take encoder weights from (for initialization)",
        )
        parser.add_argument(
            "--load-pretrained-hubert-from",
            type=str,
            metavar="STR",
            help="model to take hubert weights from (for initialization)",
        )
        
        parser.add_argument(
            "--load-pretrained-decoder-from",
            type=str,
            metavar="STR",
            help="model to take decoder weights from (for initialization)",
        )

    def __init__(self, encoder, decoder, hubert):
        super().__init__(encoder, decoder)
        self.hubert = hubert
        self._num_updates = 0

    @classmethod
    def build_hubert(cls, args):
        hubert = None
        pretraining_path = getattr(args, "load_pretrained_encoder_from", None)
        if pretraining_path is not None:
            if not Path(pretraining_path).exists():
                logger.warning(
                    f"skipped pretraining because {pretraining_path} does not exist"
                )
            else:
                hubert, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([pretraining_path])
                logger.info(f"loaded pretrained encoder from: {pretraining_path}")
        return hubert

    @classmethod
    def build_encoder(cls, args,task):
        embed_speaker = task.get_speaker_embeddings(args)
        encoder = S2STTransformerEncoder(args,embed_speaker)
        pretraining_path = getattr(args, "load_pretrained_encoder_from", None)
        if pretraining_path is not None:
            if not Path(pretraining_path).exists():
                logger.warning(
                    f"skipped pretraining because {pretraining_path} does not exist"
                )
            else:
                encoder = checkpoint_utils.load_pretrained_component_from_model(
                    component=encoder, checkpoint=pretraining_path
                )
                print(f"loaded pretrained encoder from: {pretraining_path}")
                logger.info(f"loaded pretrained encoder from: {pretraining_path}")
        return encoder
        
    @classmethod
    def build_decoder(cls, args,task):
        decoder = S2STTransformerDecoder(args, task.src_dict, task.tgt_dict)
        pretraining_path = getattr(args, "load_pretrained_decoder_from", None)
        if pretraining_path is not None:
            if not Path(pretraining_path).exists():
                logger.warning(
                    f"skipped pretraining because {pretraining_path} does not exist"
                )
            else:
                decoder = checkpoint_utils.load_pretrained_component_from_model(
                    component=decoder, checkpoint=pretraining_path
                )
                logger.info(f"loaded pretrained decoder from: {pretraining_path}")
        return decoder
    
    @classmethod
    def build_text_decoder(cls, args, task, embed_tokens):
        return TransformerDecoderScriptable(args, task.tgt_dict, embed_tokens)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            # return Embedding(num_embeddings, embed_dim, padding_idx)
            return Embedding(num_embeddings, embed_dim)
        
        decoder_embed_tokens = build_embedding(
            task.target_dictionary, args.decoder_embed_dim
        )

        hubert = cls.build_hubert(args)
        encoder = cls.build_encoder(args, task)
        decoder = cls.build_decoder(args, task)
        # text_decoder = cls.build_text_decoder(args, task, decoder_embed_tokens)
        return cls(encoder, decoder,hubert)
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
        tag = "ctc"
    ):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = self.get_normalized_probs_scriptable(net_output, log_probs, sample, tag=tag)
        lprobs.batch_first = True
        return lprobs

    def get_normalized_probs_scriptable(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
        tag = "ctc",
    ):
        """Scriptable helper function for get_normalized_probs in ~BaseFairseqModel"""
        if hasattr(self, "decoder"):
            return self.decoder.get_normalized_probs(net_output, log_probs, sample, tag=tag)
        elif torch.is_tensor(net_output):
            # syntactic sugar for simple models which don't have a decoder
            # (e.g., the classification tutorial)
            logits = net_output.float()
            if log_probs:
                return F.log_softmax(logits, dim=-1)
            else:
                return F.softmax(logits, dim=-1)
        raise NotImplementedError

    def get_targets(self, sample, test_type, net_output):
        """Get targets from either the sample or the net's output."""
        if test_type == "asr":
            return sample["src_text"]
        else:
            return sample["tgt_text"]

    def forward_encoder(self, src_tokens, src_lengths, speaker=None, **kwargs):
        return self.encoder(src_tokens, src_lengths=src_lengths,
                            speaker=speaker, **kwargs)

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self._num_updates = num_updates


@register_model_architecture("s2st_transformer_mtl", "s2st_transformer_mtl")
def base_architecture(args):
    args.dropout = getattr(args, "dropout", 0.1)
    args.output_frame_dim = getattr(args, "output_frame_dim", 80)
    args.speaker_embed_dim = getattr(args, "speaker_embed_dim", 64)
    args.middle_layers = getattr(args, "middle_layers", "6")
    args.middle_layers_decoder = getattr(args, "middle_layers_decoder", "6")
    #args.load_pretrained_encoder_from = getattr(args, "load_pretrained_encoder_from", None)
    # Convolutional subsampler
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_chaFnnels", 1024)
    # encoder transformer layers
    args.encoder_transformer_layers = getattr(args, "encoder_transformer_layers", 12)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 8 * args.encoder_embed_dim)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    # decoder prenet
    args.prenet_dropout = getattr(args, "prenet_dropout", 0.5)
    args.prenet_layers = getattr(args, "prenet_layers", 2)
    args.prenet_dim = getattr(args, "prenet_dim", 256)
    # decoder postnet
    args.postnet_dropout = getattr(args, "postnet_dropout", 0.5)
    args.postnet_layers = getattr(args, "postnet_layers", 5)
    args.postnet_conv_dim = getattr(args, "postnet_conv_dim", 512)
    args.postnet_conv_kernel_size = getattr(args, "postnet_conv_kernel_size", 5)
    # decoder transformer layers
    args.decoder_transformer_layers = getattr(args, "decoder_transformer_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4 * args.decoder_embed_dim)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
