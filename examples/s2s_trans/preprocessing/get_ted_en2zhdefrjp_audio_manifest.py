# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path
from collections import defaultdict
import os
import csv
from typing import Tuple, Union
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import torchaudio
from torchaudio.datasets.utils import download_url, extract_archive
from torch import Tensor
from torch.utils.data import Dataset

from examples.speech_to_text.data_utils import save_df_to_tsv

from examples.s2s_trans.preprocessing.cn_tn import run_cn_tn

import tensorflow as tf

import nltk
for dependency in ("brown", "names", "wordnet", "averaged_perceptron_tagger", "universal_tagset"):
    nltk.download(dependency)

from normalise import normalise, tokenize_basic


log = logging.getLogger(__name__)

SPLITS = ["train", "dev2010", "tst2015"]
LANG = ["en2zh"]

_RELEASE_CONFIGS = {
    "ted_enzh": {
        "src_folder_in_archive": {"train": "train-segment", "dev2010": "dev2010", "tst2015": "tst2015"},
        "tgt_folder_in_archive": "ted_enzh",
        "src_url": "data/speech_translation/slt_4_ted_en2zhdefrjp/raw/",
        "tgt_url": "data/speech_translation/slt_4_ted_en2zhdefrjp/tts_by_lab/ted_enzh.tar.gz",
        "metadata": {"train": "data/speech_translation/slt_4_ted_en2zhdefrjp/raw/train.en2zh.yml",
                     "dev2010": "data/speech_translation/slt_4_ted_en2zhdefrjp/raw/dev2010.en2zh.yml",
                     "tst2015": "data/speech_translation/slt_4_ted_en2zhdefrjp/raw/tst2015.en2zh.yml"},
        "src_align": "data/speech_translation/slt_4_ted_en2zhdefrjp/raw/fix_align_src.txt",
        "tgt_align": "data/speech_translation/slt_4_ted_en2zhdefrjp/raw/fix_align_tgt.txt",
    }
}


class Ted_En2ZhDeFrJp(Dataset):
    """Create a Dataset for Ted_En2ZhDeFrJp.
    Args:
        tgt_root (str or Path): Path to the directory where the dataset is found or downloaded.
        url (str, optional): The URL to download the dataset from.
            (default: ``"data/speech_translation/slt_4_ted_en2zhdefrjp/tts_by_lab/ted_enzh.tar.gz"``)
        tgt_folder_in_archive (str, optional):
            The top-level directory of the dataset. (default: ``"ted_enzh"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
    """

    def __init__(self,
                 src_root: Union[str, Path],
                 tgt_root: Union[str, Path],
                 src_url: str = _RELEASE_CONFIGS["ted_enzh"]["src_url"],
                 tgt_url: str = _RELEASE_CONFIGS["ted_enzh"]["tgt_url"],
                 src_folder_in_archive: dict = _RELEASE_CONFIGS["ted_enzh"]["src_folder_in_archive"],
                 tgt_folder_in_archive: str = _RELEASE_CONFIGS["ted_enzh"]["tgt_folder_in_archive"],
                 download: bool = False,
                 split: str = "train") -> None:
        self._split = split
        self._parse_src_filesystem(src_root, src_url, src_folder_in_archive, download)
        self._parse_tgt_filesystem(tgt_root, tgt_url, tgt_folder_in_archive, download)
        with tf.io.gfile.GFile(_RELEASE_CONFIGS["ted_enzh"]["metadata"][self._split], "r") as metadata:
            flist = csv.reader(metadata, delimiter="\t", quoting=csv.QUOTE_NONE)
            self._flist = list(flist)

        self.dict_yaml_audio_src = {}
        self.dict_yaml_audio_tgt = {}

        with tf.io.gfile.GFile(_RELEASE_CONFIGS["ted_enzh"]["src_align"], "r") as falign_src:
            for l in falign_src:
                yamlid, audioid_src = l.strip().split('\t')
                self.dict_yaml_audio_src[yamlid] = audioid_src

        with tf.io.gfile.GFile(_RELEASE_CONFIGS["ted_enzh"]["tgt_align"], "r") as falign_tgt:
            for l in falign_tgt:
                yamlid, audioid_tgt = l.strip().split('\t')
                self.dict_yaml_audio_tgt[yamlid] = audioid_tgt

    def _parse_src_filesystem(self, root: str, url: str, folder_in_archive: str, download: bool) -> None:
        url = f"{url}/{self._split}-segment.tar.gz"
        basename = os.path.basename(url)
        archive = os.path.join(root, basename)
        basename = basename.split(".tar.gz")[0]

        self._src_path = os.path.join(root, f"{folder_in_archive[self._split]}")
        if download:
            if not os.path.isdir(self._src_path):
                if not os.path.isfile(archive):
                    os.system(f"cp {url} {root}")
                extract_archive(archive)

    def _parse_tgt_filesystem(self, root: str, url: str, folder_in_archive: str, download: bool) -> None:
        basename = os.path.basename(url)
        archive = os.path.join(root, basename)
        basename = basename.split(".tar.gz")[0]

        self._tgt_path = os.path.join(root, folder_in_archive)
        if download:
            if not os.path.isdir(self._tgt_path):
                if not os.path.isfile(archive):
                    os.system(f"cp {url} {root}")
                extract_archive(archive)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str]:
        """Load the n-th sample from the dataset.
        Args:
            n (int): The index of the sample to be loaded
        Returns:
            (Tensor, int, str, str):
            ``(waveform, sample_rate, transcript, normalized_transcript)``
        """
        line = self._flist[n]
        fileid, transcript, translation = line
        normalized_transcript = None
        normalized_translation = None
        src_fileid, tgt_fileid = fileid, fileid
        try:
            src_fileid = self.dict_yaml_audio_src[fileid]
            tgt_fileid = self.dict_yaml_audio_tgt[fileid]
        except:
            pass
        src_fileid_audio = os.path.join(self._src_path, src_fileid)
        tgt_fileid_audio = os.path.join(self._tgt_path, self._split, tgt_fileid)

        # Load audio
        try:
            src_waveform, src_sample_rate = torchaudio.load(src_fileid_audio)
            tgt_waveform, tgt_sample_rate = torchaudio.load(tgt_fileid_audio)
            return (
                src_waveform,
                src_sample_rate,
                tgt_waveform,
                tgt_sample_rate,
                transcript,
                normalized_transcript,
                translation,
                normalized_translation,
                self._split,
                src_fileid_audio,
                tgt_fileid_audio,
            )
        except:
            # damaged audio
            return (
                None,
                None,
                None,
                None,
                transcript,
                normalized_transcript,
                translation,
                normalized_translation,
                self._split,
                src_fileid_audio,
                tgt_fileid_audio,
            )

    def __len__(self) -> int:
        return len(self._flist)

def process(args):
    src_out_root = Path(args.output_data_root_src).absolute()
    src_out_root.mkdir(parents=True, exist_ok=True)

    tgt_out_root = Path(args.output_data_root_tgt).absolute()
    tgt_out_root.mkdir(parents=True, exist_ok=True)

    # Generate TSV manifest
    print("Generating manifest...")
    manifest_by_split = {split: defaultdict(list) for split in SPLITS}

    for split in SPLITS:
        dataset = Ted_En2ZhDeFrJp(src_out_root.as_posix(), tgt_out_root.as_posix(), download=False, split=split)

        progress = tqdm(enumerate(dataset), total=len(dataset))
        for i, (src_waveform, src_sr, tgt_waveform, tgt_sr, src_utt, src_normalized_utt, tgt_utt, tgt_normalized_utt, split, src_fileid_audio, tgt_fileid_audio) in progress:
            if not src_sr:
                continue
            sample_id = dataset._flist[i][0]
            manifest_by_split[split]["id"].append(sample_id)
            src_audio_path, tgt_audio_path = src_fileid_audio, tgt_fileid_audio
            manifest_by_split[split]["src_audio"].append(src_audio_path)
            manifest_by_split[split]["src_n_frames"].append(len(src_waveform[0]))
            manifest_by_split[split]["src_text"].append(src_utt)
            manifest_by_split[split]["tgt_audio"].append(tgt_audio_path)
            manifest_by_split[split]["tgt_n_frames"].append(len(tgt_waveform[0]))
            manifest_by_split[split]["tgt_text"].append(tgt_utt)
            manifest_by_split[split]["speaker"].append("ted_en2zhdefrjp")

    manifest_root = Path(args.output_manifest_root_src_tgt).absolute()
    manifest_root.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        save_df_to_tsv(
            pd.DataFrame.from_dict(manifest_by_split[split]),
            manifest_root / f"{split}.audio.tsv"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-data-root-src", "-dsrc", required=True, type=str)
    parser.add_argument("--output-data-root-tgt", "-dtgt", required=True, type=str)
    parser.add_argument("--output-manifest-root-src-tgt", "-msrc_tgt", required=True, type=str)
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()