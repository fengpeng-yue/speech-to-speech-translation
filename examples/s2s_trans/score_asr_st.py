# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import soundfile as sf
import sys
import torch
import torchaudio

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.logging import progress_bar
from fairseq.tasks.text_to_speech import plot_tts_output
from fairseq.data.audio.text_to_speech_dataset import TextToSpeechDataset
from fairseq import scoring


logging.basicConfig()
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_parser():
    parser = options.get_speech_generation_parser()
    parser.add_argument("--dump-features", action="store_true")
    parser.add_argument("--dump-waveforms", action="store_true")
    parser.add_argument("--dump-attentions", action="store_true")
    parser.add_argument("--dump-eos-probs", action="store_true")
    parser.add_argument("--dump-plots", action="store_true")
    parser.add_argument("--dump-target", action="store_true")
    parser.add_argument("--output-sample-rate", default=24000, type=int)
    parser.add_argument("--score-type", default="asr", type=str)
    parser.add_argument("--results-path", default=None, type=str)

    return parser


def main(args):

    if args.max_tokens is None and args.batch_size is None:
        args.max_tokens = 8000
    logger.info(args)

    use_cuda = torch.cuda.is_available() and not args.cpu
    task = tasks.setup_task(args)
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [args.path],
        task=task,
    )
    model = models[0].cuda() if use_cuda else models[0]
    # use the original n_frames_per_step
    task.load_dataset(args.gen_subset, task_cfg=saved_cfg.task)

    generator = task.build_generator_score([model], args)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.batch_size,
        max_positions=(sys.maxsize, sys.maxsize),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
        data_buffer_size=args.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    if args.score_type == "asr":
        score = scoring.build_scorer("wer", model.decoder.src_dict)
    else:
        score = scoring.build_scorer("bleu", model.decoder.src_dict)

    fsrc_texts = open(f"{args.results_path}/res_texts.txt", 'w')
    fhyps_src_texts = open(f"{args.results_path}/hyps_res_texts.txt", 'w')
    Path(args.results_path).mkdir(exist_ok=True, parents=True)


    with progress_bar.build_progress_bar(args, itr) as t:
        for sample in t:
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            hypos = generator.generate(model, sample, has_targ=args.dump_target)
            score.add_string(hypos["res_texts"], hypos["hyps_res_texts"])
            fsrc_texts.write(hypos["res_texts"]+'\n')
            fhyps_src_texts.write(hypos["hyps_res_texts"]+'\n')


def cli_main():
    parser = make_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
