# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from itertools import groupby
from fairseq import scoring

from examples.s2s_trans.data.data_cfg import S2STDataConfig


class SpeechGenerator(object):
    def __init__(self, model, vocoder, data_cfg: S2STDataConfig):
        self.model = model
        self.vocoder = vocoder
        stats_npz_path = data_cfg.tgt_global_cmvn_stats_npz
        self.gcmvn_stats = None
        if stats_npz_path is not None:
            self.gcmvn_stats = np.load(stats_npz_path)

    def gcmvn_denormalize(self, x):
        # x: B x T x C
        if self.gcmvn_stats is None:
            return x
        mean = torch.from_numpy(self.gcmvn_stats["mean"]).to(x)
        std = torch.from_numpy(self.gcmvn_stats["std"]).to(x)
        assert len(x.shape) == 3 and mean.shape[0] == std.shape[0] == x.shape[2]
        x = x * std.view(1, 1, -1).expand_as(x)
        return x + mean.view(1, 1, -1).expand_as(x)

    def get_waveform(self, feat):
        # T x C -> T
        return None if self.vocoder is None else self.vocoder(feat).squeeze(0)


class AutoRegressiveSpeechGenerator(SpeechGenerator):
    def __init__(
            self, model, vocoder, data_cfg, max_iter: int = 6000,
            eos_prob_threshold: float = 0.5,
    ):
        super().__init__(model, vocoder, data_cfg)
        self.max_iter = max_iter
        self.eos_prob_threshold = eos_prob_threshold

    @torch.no_grad()
    def generate(self, model, sample, has_targ=False, **kwargs):
        model.eval()

        src_tokens = sample["net_input"]["src_speech"]
        src_lengths = sample["net_input"]["src_speech_lens"]
        src_texts = sample["source_texts"]
        bsz, src_len, _ = src_tokens.size()
        n_frames_per_step = model.decoder.n_frames_per_step
        out_dim = model.decoder.out_dim
        raw_dim = out_dim // n_frames_per_step

        # initialize
        encoder_out = model.forward_encoder(src_tokens, src_lengths,
                                            speaker=sample["speaker"])

        # scoring
        scorer = scoring.build_scorer('wer', model.decoder.src_dict)


        # init finalize
        finalized = [{}for _ in range(bsz)]

        # decode source text
        if "decode_source_text" in kwargs and kwargs["decode_source_text"]:

            encoded_ctc = encoder_out["out_middle_layers"][0]
            logits_ctc = model.decoder.ctc_proj(encoded_ctc).transpose(0,1)
            log_probs = torch.log_softmax(logits_ctc, dim=-1)
            best_paths = log_probs.argmax(-1)  # `[B, L]`

            src_hyps = []
            for b in range(logits_ctc.size(0)):
                indices = [best_paths[b, t].item() for t in range(encoder_out["src_lengths"][b])]

                # Step 1. Collapse repeated labels
                collapsed_indices = [x[0] for x in groupby(indices)]

                # Step 2. Remove all blank labels
                best_hyp = [x for x in filter(lambda x: x != 0, collapsed_indices)]
                src_hyps.append([best_hyp])
            hyps_src_texts = [model.decoder.src_dict.string(src_hyps[i][0]) for i in range(logits_ctc.size(0))]

            # compute wer
            for hyps_src_str, src_str in zip(hyps_src_texts, src_texts):
                assert hasattr(scorer, "add_string")
                scorer.add_string(src_str, hyps_src_str)

            for b in range(bsz):
                finalized[b]["src_texts"] = src_texts[b]
                finalized[b]["hyps_src_texts"] = hyps_src_texts[b]
            # print(scorer.score())

        if "decode_target_mel" in kwargs and kwargs["decode_target_mel"]:
            # decode target mel
            incremental_state = {}
            feat, attn, eos_prob = [], [], []
            finished = src_tokens.new_zeros((bsz,)).bool()
            out_lens = src_lengths.new_zeros((bsz,)).long().fill_(self.max_iter)

            prev_feat_out = encoder_out["encoder_out"][0].new_zeros(bsz, 1, out_dim)

            for step in range(self.max_iter):
                cur_out_lens = out_lens.clone()
                cur_out_lens.masked_fill_(cur_out_lens.eq(self.max_iter), step + 1)
                _, cur_eos_out, cur_extra = model.forward_decoder(
                    prev_feat_out, encoder_out=encoder_out,
                    incremental_state=incremental_state,
                    target_lengths=cur_out_lens, speaker=sample["speaker"], **kwargs
                )
                cur_eos_prob = torch.sigmoid(cur_eos_out).squeeze(2)
                feat.append(cur_extra['feature_out'])
                attn.append(cur_extra['attn'])
                eos_prob.append(cur_eos_prob)

                cur_finished = (cur_eos_prob.squeeze(1) > self.eos_prob_threshold)
                out_lens.masked_fill_((~finished) & cur_finished, step + 1)
                finished = finished | cur_finished
                if finished.sum().item() == bsz:
                    break
                prev_feat_out = cur_extra['feature_out']

            feat = torch.cat(feat, dim=1)
            feat = model.decoder.postnet(feat) + feat
            eos_prob = torch.cat(eos_prob, dim=1)
            attn = torch.cat(attn, dim=2)
            alignment = attn.max(dim=1)[1]

            feat = feat.reshape(bsz, -1, raw_dim)
            feat = self.gcmvn_denormalize(feat)

            eos_prob = eos_prob.repeat_interleave(n_frames_per_step, dim=1)
            attn = attn.repeat_interleave(n_frames_per_step, dim=2)
            alignment = alignment.repeat_interleave(n_frames_per_step, dim=1)
            out_lens = out_lens * n_frames_per_step

            for b, out_len in zip(range(bsz), out_lens):
                finalized[b]['feature'] = feat[b, :out_len]
                finalized[b]['eos_prob'] = eos_prob[b, :out_len]
                finalized[b]['attn'] =  attn[b, :, :out_len]
                finalized[b]['alignment'] = alignment[b, :out_len]
                finalized[b]['waveform'] = self.get_waveform(feat[b, :out_len])

        if has_targ:
            assert sample["tgt_speech"].size(-1) == out_dim
            tgt_feats = sample["tgt_speech"].view(bsz, -1, raw_dim)
            tgt_feats = self.gcmvn_denormalize(tgt_feats)
            tgt_lens = sample["target_lengths"] * n_frames_per_step
            for b, (f, l) in enumerate(zip(tgt_feats, tgt_lens)):
                finalized[b]["targ_feature"] = f[:l]
                finalized[b]["targ_waveform"] = self.get_waveform(f[:l])

        return finalized


class NonAutoregressiveSpeechGenerator(SpeechGenerator):
    @torch.no_grad()
    def generate(self, model, sample, has_targ=False, **kwargs):
        model.eval()

        bsz, max_src_len = sample["net_input"]["src_tokens"].size()
        n_frames_per_step = model.encoder.n_frames_per_step
        out_dim = model.encoder.out_dim
        raw_dim = out_dim // n_frames_per_step

        feat, feat_post, out_lens, log_dur_out, _, _ = model(
            src_tokens=sample["net_input"]["src_tokens"],
            src_lengths=sample["net_input"]["src_lengths"],
            prev_output_tokens=sample["net_input"]["prev_output_tokens"],
            incremental_state=None,
            target_lengths=sample["target_lengths"],
            speaker=sample["speaker"]
        )
        if feat_post is not None:
            feat = feat_post

        feat = feat.view(bsz, -1, raw_dim)
        feat = self.gcmvn_denormalize(feat)

        dur_out = torch.clamp(
            torch.round(torch.exp(log_dur_out) - 1).long(), min=0
        )

        def get_dur_plot_data(d):
            r = []
            for i, dd in enumerate(d):
                r += [i + 1] * dd.item()
            return r

        out_lens = out_lens * n_frames_per_step
        finalized = [
            {
                'feature': feat[b, :l] if l > 0 else feat.new_zeros([1, raw_dim]),
                'waveform': self.get_waveform(
                    feat[b, :l] if l > 0 else feat.new_zeros([1, raw_dim])
                ),
                'attn': feat.new_tensor(get_dur_plot_data(dur_out[b])),
            }
            for b, l in zip(range(bsz), out_lens)
        ]

        if has_targ:
            tgt_feats = sample["target"].view(bsz, -1, raw_dim)
            tgt_feats = self.gcmvn_denormalize(tgt_feats)
            tgt_lens = sample["target_lengths"] * n_frames_per_step
            for b, (f, l) in enumerate(zip(tgt_feats, tgt_lens)):
                finalized[b]["targ_feature"] = f[:l]
                finalized[b]["targ_waveform"] = self.get_waveform(f[:l])
        return finalized


class TeacherForcingAutoRegressiveSpeechGenerator(AutoRegressiveSpeechGenerator):
    @torch.no_grad()
    def generate(self, model, sample, has_targ=False, **kwargs):
        model.eval()

        src_tokens = sample["net_input"]["src_tokens"]
        src_lens = sample["net_input"]["src_lengths"]
        prev_out_tokens = sample["net_input"]["prev_output_tokens"]
        tgt_lens = sample["target_lengths"]
        n_frames_per_step = model.decoder.n_frames_per_step
        raw_dim = model.decoder.out_dim // n_frames_per_step
        bsz = src_tokens.shape[0]

        feat, eos_prob, extra = model(
            src_tokens, src_lens, prev_out_tokens, incremental_state=None,
            target_lengths=tgt_lens, speaker=sample["speaker"]
        )

        attn = extra["attn"]  # B x T_s x T_t
        alignment = attn.max(dim=1)[1]
        feat = feat.reshape(bsz, -1, raw_dim)
        feat = self.gcmvn_denormalize(feat)
        eos_prob = eos_prob.repeat_interleave(n_frames_per_step, dim=1)
        attn = attn.repeat_interleave(n_frames_per_step, dim=2)
        alignment = alignment.repeat_interleave(n_frames_per_step, dim=1)
        tgt_lens = sample["target_lengths"] * n_frames_per_step

        finalized = [
            {
                'feature': feat[b, :tgt_len],
                'eos_prob': eos_prob[b, :tgt_len],
                'attn': attn[b, :, :tgt_len],
                'alignment': alignment[b, :tgt_len],
                'waveform': self.get_waveform(feat[b, :tgt_len]),
            }
            for b, tgt_len in zip(range(bsz), tgt_lens)
        ]

        if has_targ:
            tgt_feats = sample["target"].view(bsz, -1, raw_dim)
            tgt_feats = self.gcmvn_denormalize(tgt_feats)
            for b, (f, l) in enumerate(zip(tgt_feats, tgt_lens)):
                finalized[b]["targ_feature"] = f[:l]
                finalized[b]["targ_waveform"] = self.get_waveform(f[:l])
        return finalized
