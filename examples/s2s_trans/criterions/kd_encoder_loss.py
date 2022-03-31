# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

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
class KDCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    n_frames_per_step: int = field(
        default=0,
        metadata={"help": "Number of frames per decoding step"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ce_weight: float = field(
        default=0.0, metadata={"help": "weight for CE loss"}
    )
    ctc_weight: float = field(
        default=0.0, metadata={"help": "weight for CTC loss"}
    )
    sentence_avg: bool = II("optimization.sentence_avg")



@register_criterion("kd_encoder_loss", dataclass=KDCriterionConfig)
class KDCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg,label_smoothing, n_frames_per_step,
                 report_accuracy, ce_weight,
                 ctc_weight,ignore_prefix_size=0):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.n_frames_per_step = n_frames_per_step
        self.report_accuracy = report_accuracy
        self.ignore_prefix_size = ignore_prefix_size
        self.ce_weight = ce_weight
        self.ctc_weight = ctc_weight

    def forward(self, model, sample, reduction="mean"):
        #bsz, max_len, _ = sample["target"].size()
 
        src_speech = sample["net_input"]["src_speech"]
        src_speech_lens = sample["net_input"]["src_speech_lens"]
        tgt_text = sample["tgt_text_orig"]
        #print(tgt_text)

        predict_feature, teacher_feature, pre_softmax, tgt_tokens, tgt_token_lens = model(
            src_speech,
            src_speech_lens,
            tgt_text
        )
        # print(tgt_tokens)
        mse_loss = self.compute_loss(
            predict_feature,
            teacher_feature,
            tgt_token_lens

        )
        # print(tgt_tokens.size())
        # print(pre_softmax.size())
        ce_loss =  torch.tensor(0.).type_as(mse_loss)
        if self.ce_weight > 0:
            ce_loss, ce_nll_loss = self.compute_ce_loss(model.get_normalized_probs, tgt_tokens , pre_softmax, sample)
            ce_loss = ce_loss / sum(tgt_token_lens)  * self.ce_weight
            #pass
        ctc_loss = torch.tensor(0.).type_as(mse_loss)
        # if self.ctc_weight > 0.:
        #     net_output = (feat_out, eos_out, extra)
        #     lprobs = model.get_normalized_probs(net_output, log_probs=True)
        #     lprobs = lprobs.transpose(0, 1)  # T x B x C
        #     src_mask = lengths_to_mask(src_lens)
        #     src_tokens_flat = src_tokens.masked_select(src_mask)
        #     ctc_loss = F.ctc_loss(
        #         lprobs, src_tokens_flat, tgt_lens, src_lens,
        #         reduction=reduction, zero_infinity=True
        #     ) * self.ctc_weight
        loss =  mse_loss + ctc_loss + ce_loss

        sample_size = sample["nsentences"] if self.sentence_avg \
            else sample["ntokens"]
        logging_output = {
            "loss": utils.item(loss.data),
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
            "mse_loss": utils.item(mse_loss.data),
            "ctc_loss": utils.item(ctc_loss.data),
            "ce_loss": utils.item(ce_loss.data),
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model.get_normalized_probs, tgt_tokens, pre_softmax, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def compute_loss(self, predict_feature, teacher_feature, tgt_lens, reduction="mean"):
        mask = lengths_to_mask(tgt_lens)
        predict_feature = predict_feature[mask]
        teacher_feature = teacher_feature[mask]

        # l1_loss = (
        #     F.l1_loss(_feat_out, _feat_tgt, reduction=reduction) +
        #     F.l1_loss(_feat_out_post, _feat_tgt, reduction=reduction)
        # )
        mse_loss = F.mse_loss(predict_feature, teacher_feature, reduction=reduction)
           
        # )
        # eos_loss = F.binary_cross_entropy_with_logits(
        #     _eos_out, _eos_tgt, pos_weight=torch.tensor(self.bce_pos_weight),
        #     reduction=reduction
        # )
        return mse_loss
    def get_lprobs_and_target(self, get_normalized_probs, target, net_output, sample):
        lprobs = get_normalized_probs(net_output, log_probs=True).contiguous()
        target = target.contiguous()
        # print(lprobs.size())
        # print(target.size())
        # target = get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_ce_loss(self, get_normalized_probs, target, net_output, sample,  reduce=True):
        lprobs, target = self.get_lprobs_and_target(get_normalized_probs, target,net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_accuracy(self, get_normalized_probs, target, net_output,  sample):
        lprobs, target = self.get_lprobs_and_target(get_normalized_probs, target, net_output, sample)
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
        for key in ["loss", "mse_loss", "ctc_loss", "ce_loss"]:
            vals = [log.get(key, 0) for log in logging_outputs]
            val = sum(val * w for val, w in zip(vals, ws))
            metrics.log_scalar(key, val, ntot, round=3)
        metrics.log_scalar("sample_size", ntot, len(logging_outputs))

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            st_n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", st_n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )


    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return False
