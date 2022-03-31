# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from collections import Counter, defaultdict

import pandas as pd
import torchaudio
from tqdm import tqdm

from fairseq.data.audio.audio_utils import convert_waveform
from examples.speech_to_text.data_utils import (
    create_zip,
    extract_fbank_features,
    gen_vocab,
    get_zip_manifest,
    load_tsv_to_dicts,
    save_df_to_tsv
)
from examples.s2s_trans.preprocessing.data_utils import gen_config_yaml,ipa_phonemize
from examples.speech_synthesis.data_utils import (
    extract_logmel_spectrogram, extract_pitch, extract_energy, get_global_cmvn
)
from pypinyin import pinyin
from pypinyin import Style

    # phones = [phone[0] for phone in pinyin(text, style=Style.TONE3)]
    # return phones



log = logging.getLogger(__name__)


def process(args):
    assert "train" in args.splits
    out_root = Path(args.output_root).absolute()
    out_root.mkdir(exist_ok=True)

    print("Fetching data...")
    audio_manifest_root = Path(args.audio_manifest_root).absolute()
    samples = []
    for s in args.splits:
        for e in load_tsv_to_dicts(audio_manifest_root / f"{s}.audio.phone.tsv"):
            e["split"] = s
            samples.append(e)
    sample_ids = [s["id"] for s in samples]



    # Extract features and pack features into ZIP
    src_feature_name = "src_logmelspec80_8k"
    tgt_feature_name = "tgt_logmelspec80"
    src_zip_path = out_root / f"{src_feature_name}.zip"
    tgt_zip_path = out_root / f"{tgt_feature_name}.zip"
    src_gcmvn_npz_path = out_root / "src_gcmvn_stats_8k.npz"
    tgt_gcmvn_npz_path = out_root / "tgt_gcmvn_stats.npz"

    if src_zip_path.exists() and src_gcmvn_npz_path.exists() and \
        tgt_zip_path.exists() and tgt_gcmvn_npz_path.exists():
        print(f"{src_zip_path} and {src_gcmvn_npz_path} exist.")
        print(f"{tgt_zip_path} and {tgt_gcmvn_npz_path} exist.")
    else:
        src_feature_root = out_root / src_feature_name
        src_feature_root.mkdir(exist_ok=True)
        tgt_feature_root = out_root / tgt_feature_name
        tgt_feature_root.mkdir(exist_ok=True)

        print("Extracting Mel spectrogram features...")
        for sample in tqdm(samples):
            src_waveform, src_sample_rate = torchaudio.load(sample["src_audio"])
            tgt_waveform, tgt_sample_rate = torchaudio.load(sample["tgt_audio"])
           
            sample_id = sample["id"]

            src_waveform, src_sample_rate = convert_waveform(
                src_waveform, src_sample_rate, 
                to_sample_rate=8000
            )
            src_features = extract_fbank_features(
                        src_waveform, src_sample_rate, src_feature_root / f"{sample_id}.npy"
                    )
            
            tgt_waveform, tgt_sample_rate = convert_waveform(
                tgt_waveform, tgt_sample_rate, normalize_volume=args.normalize_volume,
                to_sample_rate=args.sample_rate
            )
            target_length = None
            extract_logmel_spectrogram(
                tgt_waveform, tgt_sample_rate, tgt_feature_root / f"{sample_id}.npy",
                win_length=args.win_length, hop_length=args.hop_length,
                n_fft=args.n_fft, n_mels=args.n_mels, f_min=args.f_min,
                f_max=args.f_max, target_length=target_length
            )
            
        print("ZIPing features...")
        create_zip(src_feature_root, src_zip_path)
        get_global_cmvn(src_feature_root, src_gcmvn_npz_path)
        shutil.rmtree(src_feature_root)

        create_zip(tgt_feature_root, tgt_zip_path)
        get_global_cmvn(tgt_feature_root, tgt_gcmvn_npz_path)
        shutil.rmtree(tgt_feature_root)



    print("Fetching ZIP manifest...")
    src_audio_paths, src_audio_lengths = get_zip_manifest(src_zip_path)
    tgt_audio_paths, tgt_audio_lengths = get_zip_manifest(tgt_zip_path)

    # Generate TSV manifest
    print("Generating manifest...")
    manifest_by_split = {split: defaultdict(list) for split in args.splits}
    for sample in tqdm(samples):
        sample_id, split = sample["id"], sample["split"]

        src_utt = sample["src_text"]
        tgt_utt = sample["tgt_text"]

        manifest_by_split[split]["id"].append(sample_id)
        manifest_by_split[split]["src_audio"].append(src_audio_paths[sample_id])
        manifest_by_split[split]["tgt_audio"].append(tgt_audio_paths[sample_id])
        manifest_by_split[split]["src_n_frames"].append(src_audio_lengths[sample_id])
        manifest_by_split[split]["tgt_n_frames"].append(tgt_audio_lengths[sample_id])
        manifest_by_split[split]["src_text"].append(src_utt)
        manifest_by_split[split]["speaker"].append(sample["speaker"])
        manifest_by_split[split]["tgt_text"].append(tgt_utt)


    for split in args.splits:
        save_df_to_tsv(
            pd.DataFrame.from_dict(manifest_by_split[split]),
            out_root / f"{split}.tsv"
        )
    # Generate vocab
    src_vocab_name, tgt_vocab_name, spm_filename = None, None, None
    if  args.ipa_vocab:
        src_vocab = Counter()
        for t in manifest_by_split["train"]["src_text"]:
            src_vocab.update(t.split(" "))
        src_vocab_name = "src_vocab.txt"
        with open(out_root / src_vocab_name, "w") as f:
            for s, c in src_vocab.most_common():
                f.write(f"{s} {c}\n")

        tgt_vocab = Counter()
        for t in manifest_by_split["train"]["tgt_text"]:
            tgt_vocab.update(t.split(" "))
        tgt_vocab_name = "tgt_vocab.txt"
        with open(out_root / tgt_vocab_name, "w") as f:
            for s, c in tgt_vocab.most_common():
                f.write(f"{s} {c}\n")
    else:
        spm_filename_prefix = "spm_char"
        spm_filename = f"{spm_filename_prefix}.model"
        with NamedTemporaryFile(mode="w") as f:
            for t in manifest_by_split["train"]["tgt_text"]:
                f.write(t + "\n")
            f.flush()  # needed to ensure gen_vocab sees dumped text
            gen_vocab(Path(f.name), out_root / spm_filename_prefix, "char")
    # Generate speaker list
    speakers = sorted({sample["speaker"] for sample in samples})
    speakers_path = out_root / "speakers.txt"
    with open(speakers_path, "w") as f:
        for speaker in speakers:
            f.write(f"{speaker}\n")
    # Generate config YAML
    win_len_t = args.win_length / args.sample_rate
    hop_len_t = args.hop_length / args.sample_rate
    extra = {
        "sample_rate": args.sample_rate,
        "features": {
            "type": "spectrogram+melscale+log",
            "eps": 1e-5, "n_mels": args.n_mels, "n_fft": args.n_fft,
            "window_fn": "hann", "win_length": args.win_length,
            "hop_length": args.hop_length, "sample_rate": args.sample_rate,
            "win_len_t": win_len_t, "hop_len_t": hop_len_t,
            "f_min": args.f_min, "f_max": args.f_max,
            "n_stft": args.n_fft // 2 + 1
        }
    }
    if len(speakers) > 1:
        extra["speaker_set_filename"] = "speakers.txt"


    gen_config_yaml(
        out_root, spm_filename=spm_filename, 
        src_vocab_name=src_vocab_name,
        tgt_vocab_name=tgt_vocab_name,
        audio_root=out_root.as_posix(), 
        input_channels=None,
        input_feat_per_channel=None, 
        specaugment_policy='ld',
        cmvn_type="global", 
        src_gcmvn_path=src_gcmvn_npz_path,
        tgt_gcmvn_path=tgt_gcmvn_npz_path,
        extra=extra
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-manifest-root", "-m", required=True, type=str)
    parser.add_argument("--output-root", "-o", required=True, type=str)
    parser.add_argument("--splits", "-s", type=str, nargs="+",
                        default=["train","dev","tst"])
    parser.add_argument("--ipa-vocab", action="store_true")
    parser.add_argument("--lang", type=str, default="en-us")
    parser.add_argument("--use-g2p", action="store_true")
    parser.add_argument("--win-length", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=300)
    parser.add_argument("--n-fft", type=int, default=1200)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--f-min", type=int, default=20)
    parser.add_argument("--f-max", type=int, default=8000)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--normalize-volume", "-n", action="store_true")
    parser.add_argument("--textgrid-zip", type=str, default=None)
    parser.add_argument("--id-to-units-tsv", type=str, default=None)
    parser.add_argument("--add-fastspeech-targets", action="store_true")
    parser.add_argument("--snr-threshold", type=float, default=None)
    parser.add_argument("--cer-threshold", type=float, default=None)
    parser.add_argument("--cer-tsv-path", type=str, default="")
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
