# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from itertools import groupby
from fairseq import scoring

from examples.s2s_trans.data.data_cfg import S2STDataConfig



class ScoreGenerator(object):
    def __init__(
        self, model, test_type, max_iter: int = 6000
    ):
        self.model = model
        self.test_type = test_type

    @torch.no_grad()
    def generate(self, model, sample):
        self.model.eval()

        src_tokens = sample["net_input"]["src_speech"]
        src_lengths = sample["net_input"]["src_speech_lens"]
        src_texts = sample["src_texts"]
        tgt_texts = sample["tgt_texts"]
        bsz, src_len, _ = src_tokens.size()
        
        # initialize
        gt_texts = None
        # scoring
        if self.test_type == "joint_asr" or self.test_type == "asr":
            scorer = scoring.build_scorer('wer', self.model.decoder.src_dict)
            gt_texts = src_texts
        else:
            scorer = scoring.build_scorer('bleu', self.model.decoder.tgt_dict)
            gt_texts = tgt_texts
        encoder_out = model.forward_encoder(src_tokens, src_lengths,
                                        speaker=sample["speaker"])

        # decode  text
        if self.test_type == "joint_asr":
            encoded_ctc = encoder_out["out_middle_layers"][0] 
            logits_ctc = model.decoder.ctc_proj(encoded_ctc).transpose(0,1)
            log_probs = torch.log_softmax(logits_ctc, dim=-1)
            best_paths = log_probs.argmax(-1)  # `[B, L]`

            hyps = []
            for b in range(logits_ctc.size(0)):
                indices = [best_paths[b, t].item() for t in range(encoder_out["src_lengths"][b])]

                # Step 1. Collapse repeated labels
                collapsed_indices = [x[0] for x in groupby(indices)]

                # Step 2. Remove all blank labels
                best_hyp = [x for x in filter(lambda x: x != 0, collapsed_indices)]
                hyps.append([best_hyp])
            hyps_res_texts = [model.decoder.src_dict.string(hyps[i][0]) for i in range(logits_ctc.size(0))]
        elif self.test_type == "asr":

            pass

        # compute wer
        for hyps_res_str, res_str in zip(hyps_res_texts, gt_texts):
            assert hasattr(scorer, "add_string")
            scorer.add_string(res_str, hyps_res_str)     
        for b in range(bsz):
            finalized = [
                    {
                    'res_texts': gt_texts[b],
                    'hyps_res_texts': hyps_res_texts[b],
                }
                    for b, out_len in zip(range(bsz), out_lens)
            ]
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
