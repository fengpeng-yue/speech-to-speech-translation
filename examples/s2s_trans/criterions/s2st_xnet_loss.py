# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import os
import sys
import logging
from typing import Any, Dict, List
from functools import lru_cache
from dataclasses import dataclass, field

import torch
from omegaconf import II

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.data.data_utils import lengths_to_mask
import torch.nn.functional as F

logging.basicConfig(
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

@dataclass
class S2STCriterionConfig(FairseqDataclass):
    bce_pos_weight: float = field(
        default=1.0,
        metadata={"help": "weight of positive examples for BCE loss"},
    )
    n_frames_per_step: int = field(
        default=0,
        metadata={"help": "Number of frames per decoding step"},
    )
    use_guided_attention_loss: bool = field(
        default=False,
        metadata={"help": "use guided attention loss"},
    )
    guided_attention_loss_sigma: float = field(
        default=0.4,
        metadata={"help": "weight of positive examples for BCE loss"},
    )
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    ctc_weight: float = field(
        default=0.0, metadata={"help": "weight for CTC loss"}
    )
    st_ce_weight: float = field(
        default=0.0, metadata={"help": "weight for sr CE loss"}
    )
    sentence_avg: bool = II("optimization.sentence_avg")


class GuidedAttentionLoss(torch.nn.Module):
    """
    Efficiently Trainable Text-to-Speech System Based on Deep Convolutional
    Networks with Guided Attention (https://arxiv.org/abs/1710.08969)
    """

    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    @staticmethod
    @lru_cache(maxsize=8)
    def _get_weight(s_len, t_len, sigma):
        grid_x, grid_y = torch.meshgrid(torch.arange(t_len), torch.arange(s_len))
        grid_x = grid_x.to(s_len.device)
        grid_y = grid_y.to(s_len.device)
        w = (grid_y.float() / s_len - grid_x.float() / t_len) ** 2
        return 1.0 - torch.exp(-w / (2 * (sigma ** 2)))

    def _get_weights(self, src_lens, tgt_lens):
        bsz, max_s_len, max_t_len = len(src_lens), max(src_lens), max(tgt_lens)
        weights = torch.zeros((bsz, max_t_len, max_s_len))
        for i, (s_len, t_len) in enumerate(zip(src_lens, tgt_lens)):
            weights[i, :t_len, :s_len] = self._get_weight(s_len, t_len,
                                                          self.sigma)
        return weights

    @staticmethod
    def _get_masks(src_lens, tgt_lens):
        in_masks = lengths_to_mask(src_lens)
        out_masks = lengths_to_mask(tgt_lens)
        return out_masks.unsqueeze(2) & in_masks.unsqueeze(1)

    def forward(self, attn, src_lens, tgt_lens, reduction="mean"):
        weights = self._get_weights(src_lens, tgt_lens).to(attn.device)
        masks = self._get_masks(src_lens, tgt_lens).to(attn.device)
        loss = (weights * attn.transpose(1, 2)).masked_select(masks)
        loss = torch.sum(loss) if reduction == "sum" else torch.mean(loss)
        return loss

@register_criterion("s2st_xnet_loss", dataclass=S2STCriterionConfig)
class S2STCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg, n_frames_per_step,
                 use_guided_attention_loss, guided_attention_loss_sigma,
                 bce_pos_weight, ctc_weight, st_ce_weight, label_smoothing,
                 ignore_prefix_size=0,report_accuracy=False,):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.n_frames_per_step = n_frames_per_step
        self.bce_pos_weight = bce_pos_weight
        self.args = task.args
        self.guided_attn = None
        if use_guided_attention_loss:
            self.guided_attn = GuidedAttentionLoss(guided_attention_loss_sigma)
        self.ctc_weight = ctc_weight
        self.st_ce_weight = st_ce_weight
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

        if self.ctc_weight > 0:
             self.ctc_loss = torch.nn.CTCLoss(
                reduction="mean", zero_infinity=True
            )
        

    def forward(self, model, sample, reduction="mean"):
        bsz, max_len, _ = sample["tgt_speech"].size()
        idx = sample["id"]
        feat_tgt = sample["tgt_speech"]

        feat_len = sample["target_lengths"].view(bsz, 1).expand(-1, max_len)
        eos_tgt = torch.arange(max_len).to(sample["tgt_speech"].device)
        eos_tgt = eos_tgt.view(1, max_len).expand(bsz, -1)
        eos_tgt = (eos_tgt == (feat_len - 1)).float()
        src_tokens = sample["net_input"]["src_speech"]
        src_text = sample["src_text"]
        src_text_len = sample["src_text_len"]
        src_txt_ntokens = sample["src_txt_ntokens"]
        tgt_txt_ntokens = sample["tgt_txt_ntokens"]

        src_lens = sample["net_input"]["src_speech_lens"]
        tgt_speech_lens = sample["target_lengths"]
        tgt_text_lens = sample["tgt_text_len"]
        collated_audios = sample["net_input"]["collated_audios_orig"]
        padding_mask = sample["net_input"]["padding_mask"]


        prev_src_text_tokens = None
        if self.st_ce_weight > 0:
            prev_tgt_text_tokens = sample["net_input"]["prev_tgt_text_tokens"]
        else:
            prev_tgt_text_tokens = None
        main_task_output, joint_st_output = model(
            src_tokens=src_tokens,
            src_lengths=src_lens,
            src_txt = src_text,
            src_txt_lens = src_text_len,
            collated_audios=collated_audios,
            padding_mask = padding_mask,
            prev_output_tokens=sample["net_input"]["prev_output_tokens"],
            prev_src_text_tokens = prev_src_text_tokens,
            prev_tgt_text_tokens = prev_tgt_text_tokens,
            incremental_state=None,
            tgt_speech_lens=tgt_speech_lens,
            tgt_text_lens = tgt_text_lens,
            speaker=sample["speaker"]
        )
        feat_out, eos_out, extra = main_task_output
        #logging.info(feat_out.size())
        l1_loss, mse_loss, eos_loss = self.compute_loss(
            extra["feature_out"], feat_out, eos_out, feat_tgt, eos_tgt,
            tgt_speech_lens, reduction,
        )
        attn_loss = torch.tensor(0.).type_as(l1_loss)
        if self.guided_attn is not None:
            attn_loss = self.guided_attn(extra['attn'], src_lens, tgt_speech_lens, reduction)
        ctc_loss = torch.tensor(0.).type_as(l1_loss)
        if self.ctc_weight > 0.:
            net_output = (feat_out, eos_out, extra) 
            for k in list([int(k) for k in self.args.conv_kernel_sizes.split(",")]):
                    src_lens = (src_lens - k + 2*(k//2)) // 2 + 1 
            lprobs = model.get_normalized_probs(net_output, log_probs=True)
            lprobs = lprobs.transpose(0, 1)  # T x B x C
            
            src_mask = lengths_to_mask(src_text_len)
            src_tokens_flat = src_text.masked_select(src_mask)
            # ctc_loss = F.ctc_loss(
            #     lprobs, src_tokens_flat, src_lens, src_text_len,
            #     reduction=reduction, zero_infinity=True
            # ) * self.ctc_weight
            with torch.backends.cudnn.flags(deterministic=True):
                ctc_loss = self.ctc_loss(lprobs, src_tokens_flat, src_lens, src_text_len) * self.ctc_weight
        #aux_asr_loss = torch.tensor(0.).type_as(l1_loss)
        #if self.asr_ce_weight > 0:
            #print("src_text_token % d" % src_txt_ntokens)
            #aux_asr_loss, asr_nll_loss = self.compute_aux_loss(model.aux_asr_decoder.get_normalized_probs, model.get_targets , aux_asr_output, sample,"asr")
            #aux_asr_loss = aux_asr_loss / src_txt_ntokens  * self.asr_ce_weight
            #print(aux_asr_loss)
        
        joint_st_loss = torch.tensor(0.).type_as(l1_loss)
        if self.st_ce_weight > 0:
            #print("src_text_token % d" % src_txt_ntokens)
            # print(sample["tgt_text"].size())
            # print(joint_st_output[0].size())
            joint_st_loss, st_nll_loss = self.compute_aux_loss(model.get_normalized_probs, model.get_targets , joint_st_output, sample,"st")
            joint_st_loss = joint_st_loss / tgt_txt_ntokens  * self.st_ce_weight
        

        loss = l1_loss + mse_loss + eos_loss + attn_loss + ctc_loss + joint_st_loss

        if loss > 10:
            print(idx)
            print(l1_loss)
            print(mse_loss)
            print(eos_loss)
            print(joint_st_loss)
            print(tgt_txt_ntokens)


        sample_size = sample["nsentences"] if self.sentence_avg \
            else sample["ntokens"]
        logging_output = {
            "loss": utils.item(loss.data),
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
            "l1_loss": utils.item(l1_loss.data),
            "mse_loss": utils.item(mse_loss.data),
            "eos_loss": utils.item(eos_loss.data),
            "attn_loss": utils.item(attn_loss.data),
            "ctc_loss": utils.item(ctc_loss.data),
            "joint_st_loss": utils.item(joint_st_loss.data),
        }
        # if self.report_accuracy and self.asr_ce_weight > 0:
        #     n_correct, total = self.compute_accuracy(model.aux_asr_decoder.get_normalized_probs, model.get_targets ,aux_asr_output, "asr",sample) 
        #     logging_output["asr_n_correct"] = utils.item(n_correct.data)
        #     logging_output["asr_total"] = utils.item(total.data)
        if self.report_accuracy and self.st_ce_weight > 0:
            n_correct, total = self.compute_accuracy(model.get_normalized_probs, model.get_targets ,joint_st_output, "st",sample) 
            logging_output["st_n_correct"] = utils.item(n_correct.data)
            logging_output["st_total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def compute_loss(self, feat_out, feat_out_post, eos_out, feat_tgt,
                     eos_tgt, tgt_lens, reduction="mean"):
        mask = lengths_to_mask(tgt_lens)
        _eos_out = eos_out[mask].squeeze()
        _eos_tgt = eos_tgt[mask]
        _feat_tgt = feat_tgt[mask]
        _feat_out = feat_out[mask]
        _feat_out_post = feat_out_post[mask]

        l1_loss = (
            F.l1_loss(_feat_out, _feat_tgt, reduction=reduction) +
            F.l1_loss(_feat_out_post, _feat_tgt, reduction=reduction)
        )
        mse_loss = (
            F.mse_loss(_feat_out, _feat_tgt, reduction=reduction) +
            F.mse_loss(_feat_out_post, _feat_tgt, reduction=reduction)
        )
        eos_loss = F.binary_cross_entropy_with_logits(
            _eos_out, _eos_tgt, pos_weight=torch.tensor(self.bce_pos_weight),
            reduction=reduction
        )
        return l1_loss, mse_loss, eos_loss

    def get_lprobs_and_target(self, get_normalized_probs, get_targets, net_output, test_type, sample):
        lprobs = get_normalized_probs(net_output, log_probs=True)
        # print(lprobs.size())
        target = get_targets(sample,test_type, net_output)
        # print(target.size())
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_aux_loss(self, get_normalized_probs, get_targets, net_output, sample, test_type,  reduce=True):
        lprobs, target = self.get_lprobs_and_target(get_normalized_probs, get_targets,net_output,test_type,  sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_accuracy(self, get_normalized_probs, get_targets, net_output,test_type,  sample):
        lprobs, target = self.get_lprobs_and_target(get_normalized_probs, get_targets, net_output, test_type, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs: List[Dict[str, Any]]) -> None:
        ns = [log.get("sample_size", 0) for log in logging_outputs]
        ntot = sum(ns)
        ws = [n / (ntot + 1e-8) for n in ns]
        for key in ["loss", "l1_loss", "mse_loss", "eos_loss", "attn_loss", "ctc_loss","joint_st_loss"]:
            vals = [log.get(key, 0) for log in logging_outputs]
            val = sum(val * w for val, w in zip(vals, ws))
            metrics.log_scalar(key, val, ntot, round=3)
        metrics.log_scalar("sample_size", ntot, len(logging_outputs))


        # asr_total = utils.item(sum(log.get("asr_total", 0) for log in logging_outputs))
        # if asr_total > 0:
        #     metrics.log_scalar("asr_total", asr_total)
        #     asr_n_correct = utils.item(
        #         sum(log.get("asr_n_correct", 0) for log in logging_outputs)
        #     )
        #     metrics.log_scalar("asr_n_correct", asr_n_correct)
        #     metrics.log_derived(
        #         "asr_accuracy",
        #         lambda meters: round(
        #             meters["asr_n_correct"].sum * 100.0 / meters["asr_total"].sum, 3
        #         )
        #         if meters["asr_total"].sum > 0
        #         else float("nan"),
        #     )

        st_total = utils.item(sum(log.get("st_total", 0) for log in logging_outputs))
        if st_total > 0:
            metrics.log_scalar("st_total", st_total)
            st_n_correct = utils.item(
                sum(log.get("st_n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("st_n_correct", st_n_correct)
            metrics.log_derived(
                "st_accuracy",
                lambda meters: round(
                    meters["st_n_correct"].sum * 100.0 / meters["st_total"].sum, 3
                )
                if meters["st_total"].sum > 0
                else float("nan"),
            )


        # inference metrics
        if "targ_frames" not in logging_outputs[0]:
            return
        n = sum(log.get("targ_frames", 0) for log in logging_outputs)
        for key, new_key in [
                ("mcd_loss", "mcd_loss"),
                ("pred_frames", "pred_ratio"),
                ("nins", "ins_rate"),
                ("ndel", "del_rate"),
        ]:
            val = sum(log.get(key, 0) for log in logging_outputs)
            metrics.log_scalar(new_key, val / n, n, round=3)




    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return False
