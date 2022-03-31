# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.abs

from fairseq import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

import numpy as np
import random
import re
import logging
import torch

from examples.s2s_trans.data.data_cfg import S2STDataConfig
from fairseq.data.audio.speech_to_text_dataset import (
    SpeechToTextDataset, SpeechToTextDatasetCreator,
    _collate_frames, get_features_or_waveform
)
from fairseq.data import Dictionary, data_utils as fairseq_data_utils
from fairseq.data.audio.feature_transforms import CompositeAudioFeatureTransform

logger = logging.getLogger(__name__)



@dataclass
class TextToSpeechDatasetItem(object):
    index: int
    src_speech: torch.Tensor
    src_text: torch.Tensor
    tgt_speech: torch.Tensor
    tgt_text: torch.Tensor
    src_orig: Optional[torch.Tensor] = None  # for hubert
    tgt_text_orig: Optional[str] = None  # for kd encoder
    speaker_id: Optional[int] = None
    duration: Optional[torch.Tensor] = None
    pitch: Optional[torch.Tensor] = None
    energy: Optional[torch.Tensor] = None


class S2STDataset(SpeechToTextDataset):
    def __init__(
            self,
            split: str,
            is_train_split: bool,
            cfg: S2STDataConfig,
            src_audio_paths: List[str],
            src_orig_paths: List[str],
            tgt_audio_paths: List[str],
            src_n_frames: List[int],
            tgt_n_frames: List[int],
            src_texts: Optional[List[str]] = None,
            tgt_texts: Optional[List[str]] = None,
            tgt_text_orig: Optional[List[str]] = None,
            speakers: Optional[List[str]] = None,
            src_langs: Optional[List[str]] = None,
            tgt_langs: Optional[List[str]] = None,
            ids: Optional[List[str]] = None,
            src_dict: Optional[Dictionary] = None,
            tgt_dict: Optional[Dictionary] = None,
            pre_tokenizer=None,
            bpe_tokenizer=None,
            n_frames_per_step=1,
            speaker_to_id=None,
            max_sample_size = 9600000,
            durations: Optional[List[List[int]]] = None,
            pitches: Optional[List[str]] = None,
            energies: Optional[List[str]] = None,
            random_crop: Optional[bool] = False,
            pad_audio: Optional[bool] = True,
            normalize: Optional[bool] = False,
    ):

        super(S2STDataset, self).__init__(
            split, is_train_split, cfg, src_audio_paths, src_n_frames,
            src_texts=src_texts, tgt_texts=tgt_texts, speakers=speakers,
            src_langs=src_langs, tgt_langs=tgt_langs, ids=ids,
            tgt_dict=tgt_dict, pre_tokenizer=pre_tokenizer,
            bpe_tokenizer=bpe_tokenizer, n_frames_per_step=n_frames_per_step,
            speaker_to_id=speaker_to_id
        )
        self.check_tgt_lang_tag()
        self.cfg = cfg
        self.shuffle = cfg.shuffle if is_train_split else False

        self.feature_transforms_src = CompositeAudioFeatureTransform.from_config_dict_for_src(
            self.cfg.get_feature_transforms_for_src(split, is_train_split)
        )
        self.feature_transforms_tgt = CompositeAudioFeatureTransform.from_config_dict_for_tgt(
            self.cfg.get_feature_transforms_for_tgt(split, is_train_split)
        )
        self.src_audio_paths = src_audio_paths
        self.src_orig_paths = src_orig_paths
        self.tgt_audio_paths = tgt_audio_paths
        self.tgt_text_orig = tgt_text_orig
        self.src_n_frames = src_n_frames
        self.tgt_n_frames = tgt_n_frames
        self.src_dict = src_dict
        #print(self.src_dict.indices)
        self.tgt_dict = tgt_dict
        self.durations = durations
        self.pitches = pitches
        self.energies = energies

        #for hubert
        self.pad_audio = pad_audio
        self.normalize = normalize
        self.random_crop = random_crop
        self.max_sample_size = max_sample_size
        logger.info(
            f"pad_audio={pad_audio}, random_crop={random_crop}, "
            f"normalize={normalize}, max_sample_size={self.max_sample_size}"
        )

        self.tgt_lens = self.get_tgt_lens_and_check_oov()
        self.src_lens = self.get_src_lens_and_check_oov()
    def __repr__(self):
        return (
            self.__class__.__name__
            + f'(split="{self.split}", n_samples={self.n_samples:_}, '
            f"prepend_tgt_lang_tag={self.cfg.prepend_tgt_lang_tag}, "
            f"shuffle={self.shuffle}, transforms={self.feature_transforms}, "
            f"n_frames_per_step={self.n_frames_per_step}"
        )
    def get_tgt_lens_and_check_oov(self):
        if self.tgt_texts is None:
            return [0 for _ in range(self.n_samples)]
        tgt_lens = []
        n_tokens, n_oov_tokens = 0, 0
        for i in range(self.n_samples):
            tokenized = self.get_tokenized_tgt_text(i).split(" ")
            oov_tokens = [
                t
                for t in tokenized
                if self.tgt_dict.index(t) == self.tgt_dict.unk_index
            ]
            n_tokens += len(tokenized)
            n_oov_tokens += len(oov_tokens)
            tgt_lens.append(len(tokenized))
        logger.info(f"'{self.split}' has {n_oov_tokens / n_tokens * 100:.2f}% OOV in target texts")
        return tgt_lens
    def get_src_lens_and_check_oov(self):
        if self.src_texts is None:
            return [0 for _ in range(self.n_samples)]
        src_lens = []
        n_tokens, n_oov_tokens = 0, 0
        for i in range(self.n_samples):
            tokenized = self.get_tokenized_src_text(i).split(" ")
            oov_tokens = [
                t
                for t in tokenized
                if self.src_dict.index(t) == self.src_dict.unk_index
            ]
            n_tokens += len(tokenized)
            n_oov_tokens += len(oov_tokens)
            src_lens.append(len(tokenized))
        logger.info(f"'{self.split}' has {n_oov_tokens / n_tokens * 100:.2f}% OOV in source texts")
        return src_lens


    def __getitem__(self, index: int) -> TextToSpeechDatasetItem:

        if self.cfg.use_hubert:
            src_orig = self.get_audio(self.src_orig_paths[index])
        else:
            src_orig = None
        #print(self.src_audio_paths[index])
        src_speech = get_features_or_waveform(
            self.src_audio_paths[index],
            need_waveform=self.cfg.use_audio_input,
            use_sample_rate=self.cfg.use_sample_rate,
        )
        # this feature transform may be used to specaugment
        if self.feature_transforms_src is not None:
            assert not self.cfg.use_audio_input
            src_speech = self.feature_transforms_src(src_speech)
        src_speech = torch.from_numpy(src_speech).float()

        #print(self.tgt_audio_paths[index])
        tgt_speech = get_features_or_waveform(
            self.tgt_audio_paths[index],
            need_waveform=self.cfg.use_audio_input,
            use_sample_rate=self.cfg.use_sample_rate,
        )
        if self.feature_transforms_tgt is not None:
            assert not self.cfg.use_audio_input
            tgt_speech = self.feature_transforms_tgt(tgt_speech)

        tgt_speech = torch.from_numpy(tgt_speech).float()
        # stack fbank for target speech of s2st task
        tgt_speech = self.pack_frames(tgt_speech)

        tgt_text = None
        if self.tgt_texts is not None:
            tokenized = self.get_tokenized_tgt_text(index)
            tgt_text = self.tgt_dict.encode_line(
                tokenized, add_if_not_exist=False, append_eos=True
            ).long()
            if self.cfg.prepend_tgt_lang_tag:
                lang_tag_idx = self.get_lang_tag_idx(
                    self.tgt_langs[index], self.tgt_dict
                )
                tgt_text = torch.cat((torch.LongTensor([lang_tag_idx]), tgt_text), 0)


        src_text = None
        if self.src_texts is not None:
            tokenized = self.get_tokenized_src_text(index)
            src_text = self.src_dict.encode_line(
                tokenized, add_if_not_exist=False, append_eos=True
            ).long()
            # remove the eos token
            #src_text = src_text[:len(src_text)-1]
            if self.cfg.prepend_tgt_lang_tag:
                lang_tag_idx = self.get_lang_tag_idx(
                    self.src_langs[index], self.src_dict
                )
                src_text = torch.cat((torch.LongTensor([lang_tag_idx]), src_text), 0)

        speaker_id = None
        if self.speaker_to_id is not None:
            speaker_id = self.speaker_to_id[self.speakers[index]]

        duration, pitch, energy = None, None, None
        if self.durations is not None:
            duration = torch.tensor(
                self.durations[index] + [0], dtype=torch.long  # pad 0 for EOS
            )
        if self.pitches is not None:
            pitch = get_features_or_waveform(self.pitches[index])
            pitch = torch.from_numpy(
                np.concatenate((pitch, [0]))  # pad 0 for EOS
            ).float()
        if self.energies is not None:
            energy = get_features_or_waveform(self.energies[index])
            energy = torch.from_numpy(
                np.concatenate((energy, [0]))  # pad 0 for EOS
            ).float()

        tgt_text_orig = None
        if self.tgt_text_orig is not None:
            tgt_text_orig = self.tgt_text_orig[index]

        return TextToSpeechDatasetItem(
            index=index, 
            src_speech=src_speech,
            src_orig= src_orig,
            src_text=src_text,
            tgt_text_orig = tgt_text_orig,
            tgt_speech=tgt_speech,
            tgt_text = tgt_text,
            speaker_id=speaker_id, duration=duration, pitch=pitch,
            energy=energy
        )
    def check_tgt_lang_tag(self):
        if self.cfg.prepend_tgt_lang_tag:
            assert self.tgt_langs is not None and self.tgt_dict is not None
            tgt_lang_tags = [
                self.LANG_TAG_TEMPLATE.format(t) for t in set(self.tgt_langs)
            ]
            assert all(t in self.tgt_dict for t in tgt_lang_tags)
    def get_tokenized_src_text(self, index: int):
        text = self.tokenize(self.pre_tokenizer,self.src_texts[index])
        text = self.tokenize(self.bpe_tokenizer, text)

        return text

    #starting point of fuction for hubert audio input
    def get_audio(self, wav_path):
        import soundfile as sf

        #wav_path = os.path.join(self.audio_root, self.audio_names[index])
        wav, cur_sample_rate = sf.read(wav_path)
        wav = torch.from_numpy(wav).float()
        wav = self.postprocess(wav, cur_sample_rate)
        return wav
    def crop_to_max_size(self, wav, target_size):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav, 0

        start, end = 0, target_size
        if self.random_crop:
            start = np.random.randint(0, diff + 1)
            end = size - diff + start
        return wav[start:end], start
    def collater_audio(self, audios, audio_size):
        collated_audios = audios[0].new_zeros(len(audios), audio_size)
        padding_mask = (
            torch.BoolTensor(collated_audios.shape).fill_(False)
            # if self.pad_audio else None
        )
        audio_starts = [0 for _ in audios]
        for i, audio in enumerate(audios):
            diff = len(audio) - audio_size
            if diff == 0:
                collated_audios[i] = audio
            elif diff < 0:
                assert self.pad_audio
                collated_audios[i] = torch.cat(
                    [audio, audio.new_full((-diff,), 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_audios[i], audio_starts[i] = self.crop_to_max_size(
                    audio, audio_size
                )
        return collated_audios, padding_mask, audio_starts
    def postprocess(self, wav, cur_sample_rate):
        if wav.dim() == 2:
            wav = wav.mean(-1)
        assert wav.dim() == 1, wav.dim()

        # if cur_sample_rate != self.sample_rate:
        #     raise Exception(f"sr {cur_sample_rate} != {self.sample_rate}")

        if self.normalize:
            with torch.no_grad():
                wav = F.layer_norm(wav, wav.shape)
        return wav
        
    # stopping point
    def collater(self, samples: List[TextToSpeechDatasetItem]) -> Dict[str, Any]:
        if len(samples) == 0:
            return {}

        src_feat_lengths, order = torch.tensor(
            [s.src_speech.shape[0] for s in samples], dtype=torch.long
        ).sort(descending=True)
        id_ = torch.tensor([s.index for s in samples],
                           dtype=torch.long).index_select(0, order)

        src_feat = None
        if not self.cfg.use_hubert:
            src_feat = _collate_frames(
                [s.src_speech for s in samples], self.cfg.use_audio_input
            ).index_select(0, order)

        #for hubert
        if self.cfg.use_hubert:
            audios = [s.src_orig for s in samples]
            audio_sizes = [ s.size(0) for s in audios]
            if self.pad_audio:
                audio_size = min(max(audio_sizes), self.max_sample_size)
            else:
                audio_size = min(min(audio_sizes), self.max_sample_size)
            collated_audios, padding_mask, audio_starts = self.collater_audio(
                audios, audio_size
            )
            collated_audios = collated_audios.index_select(0, order)
            padding_mask = padding_mask.index_select(0, order)
        else:
            collated_audios = None
            padding_mask = None

        src_text = fairseq_data_utils.collate_tokens(
            [s.src_text for s in samples],
            self.src_dict.pad(),
            self.src_dict.eos(),
            left_pad=False,
            move_eos_to_beginning=False,
        ).index_select(0, order)
        src_text_len = torch.tensor([s.src_text.size(0) for s in samples], dtype=torch.long).index_select(0, order)

        tgt_feat_lengths = torch.tensor(
            [s.tgt_speech.shape[0] for s in samples], dtype=torch.long
        ).index_select(0, order)
        tgt_feat = _collate_frames(
            [s.tgt_speech for s in samples], self.cfg.use_audio_input
        ).index_select(0, order)
        tgt_text = fairseq_data_utils.collate_tokens(
            [s.tgt_text for s in samples],
            self.tgt_dict.pad(),
            self.tgt_dict.eos(),
            left_pad=False,
            move_eos_to_beginning=False,
        ).index_select(0, order)
        tgt_text_len = torch.tensor([s.tgt_text.size(0) for s in samples], dtype=torch.long).index_select(0, order)

        speaker = None
        if self.speaker_to_id is not None:
            speaker = torch.tensor(
                [s.speaker_id for s in samples], dtype=torch.long
            ).index_select(0, order).view(-1, 1)

        bsz, _, d = tgt_feat.size()
        prev_s2s_output_tokens = torch.cat(
            (tgt_feat.new_zeros((bsz, 1, d)), tgt_feat[:, :-1, :]), dim=1
        )
        prev_src_text_tokens = fairseq_data_utils.collate_tokens(
                [x.src_text for x in samples],
                self.src_dict.pad(),
                self.src_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=True,
            ).index_select(0, order)
        prev_tgt_text_tokens = fairseq_data_utils.collate_tokens(
                [x.tgt_text for x in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=True,
            ).index_select(0, order)

        durations, pitches, energies = None, None, None
        if self.durations is not None:
            durations = fairseq_data_utils.collate_tokens(
                [s.duration for s in samples], 0
            ).index_select(0, order)
            assert src_text.shape[1] == durations.shape[1]
        if self.pitches is not None:
            pitches = _collate_frames([s.pitch for s in samples], True)
            pitches = pitches.index_select(0, order)
            assert src_text.shape[1] == pitches.shape[1]
        if self.energies is not None:
            energies = _collate_frames([s.energy for s in samples], True)
            energies = energies.index_select(0, order)
            assert src_text.shape[1] == energies.shape[1]
        target_texts = [self.tgt_dict.string(samples[i].tgt_text) for i in order]
        tgt_text_orig = [ samples[i].tgt_text_orig for i in order ]
        return {
            "id": id_,
            "net_input": {
                "src_speech": src_feat,
                "src_speech_lens": src_feat_lengths,
                "prev_output_tokens": prev_s2s_output_tokens,
                "prev_src_text_tokens": prev_src_text_tokens,
                "prev_tgt_text_tokens": prev_tgt_text_tokens,
                "collated_audios_orig": collated_audios,
                "padding_mask": padding_mask,
                "speaker": speaker,
            },
            "speaker": speaker,
            "src_text": src_text,
            "src_text_len": src_text_len,
            "tgt_text": tgt_text,
            "tgt_text_len": tgt_text_len,
            "tgt_speech": tgt_feat,
            "target_lengths": tgt_feat_lengths,
            "durations": durations,
            "pitches": pitches,
            "energies": energies,
            "ntokens": sum(tgt_feat_lengths).item(),
            "src_txt_ntokens": sum(src_text_len).item(),
            "tgt_txt_ntokens": sum(tgt_text_len).item(),
            "nsentences": len(samples),
            "target_texts": target_texts,
            "tgt_text_orig": tgt_text_orig,
        }


class S2STDatasetCreator(SpeechToTextDatasetCreator):
    KEY_DURATION = "duration"
    KEY_PITCH = "pitch"
    KEY_ENERGY = "energy"
    # mandatory columns 
    KEY_ID, KEY_SRC_AUDIO, KEY_SRC_ORIG, KEY_TGT_AUDIO = "id", "src_audio", "src_orig", "tgt_audio"
    KEY_SRC_N_FRAMES,KEY_TGT_N_FRAMES = "src_n_frames", "tgt_n_frames"
    TGT_ORIG_TXT = "tgt_text_orig" 
    KEY_SRC_TEXT, KEY_TGT_TEXT = "src_text", "tgt_text"
    # optional columns
    KEY_SPEAKER = "speaker"
    KEY_SRC_LANG, KEY_TGT_LANG = "src_lang", "tgt_lang"
    # default values
    DEFAULT_SPEAKER = DEFAULT_SRC_TEXT = DEFAULT_LANG = ""
    @classmethod
    def _from_list(
        cls,
        split_name: str,
        is_train_split,
        samples: List[Dict],
        cfg: S2STDataConfig,
        src_dict,
        tgt_dict,
        pre_tokenizer,
        bpe_tokenizer,
        n_frames_per_step,
        speaker_to_id
    ) -> S2STDataset:
        audio_root = Path(cfg.audio_root)
        ids = [s[cls.KEY_ID] for s in samples]
        src_audio_paths = [(audio_root / s[cls.KEY_SRC_AUDIO]).as_posix() for s in samples]
        src_orig_paths = None
        tgt_orig_text = None
        if cfg.use_hubert:
            src_orig_paths = [(audio_root / s[cls.KEY_SRC_ORIG]).as_posix() for s in samples]
        if cfg.kd_encoder:
            tgt_orig_text = [s[cls.TGT_ORIG_TXT] for s in samples]
        tgt_audio_paths = [(audio_root / s[cls.KEY_TGT_AUDIO]).as_posix() for s in samples]
        src_n_frames = [int(s[cls.KEY_SRC_N_FRAMES]) for s in samples]
        tgt_n_frames = [int(s[cls.KEY_TGT_N_FRAMES]) for s in samples]
        tgt_texts = [s[cls.KEY_TGT_TEXT] for s in samples]
        src_texts = [s.get(cls.KEY_SRC_TEXT, cls.DEFAULT_SRC_TEXT) for s in samples]
        speakers = [s.get(cls.KEY_SPEAKER, cls.DEFAULT_SPEAKER) for s in samples]
        src_langs = [s.get(cls.KEY_SRC_LANG, cls.DEFAULT_LANG) for s in samples]
        tgt_langs = [s.get(cls.KEY_TGT_LANG, cls.DEFAULT_LANG) for s in samples]

        durations = [s.get(cls.KEY_DURATION, None) for s in samples]
        durations = [
            None if dd is None else [int(d) for d in dd.split(" ")]
            for dd in durations
        ]
        durations = None if any(dd is None for dd in durations) else durations

        pitches = [s.get(cls.KEY_PITCH, None) for s in samples]
        pitches = [
            None if pp is None else (audio_root / pp).as_posix()
            for pp in pitches
        ]
        pitches = None if any(pp is None for pp in pitches) else pitches

        energies = [s.get(cls.KEY_ENERGY, None) for s in samples]
        energies = [
            None if ee is None else (audio_root / ee).as_posix()
            for ee in energies]
        energies = None if any(ee is None for ee in energies) else energies

        return S2STDataset(
            split_name, is_train_split, cfg, src_audio_paths, src_orig_paths, tgt_audio_paths,
            src_n_frames, tgt_n_frames,
            src_texts, tgt_texts, tgt_orig_text,
            speakers, src_langs, tgt_langs, ids, 
            src_dict, tgt_dict,
            pre_tokenizer, bpe_tokenizer, n_frames_per_step, 
            speaker_to_id = speaker_to_id, durations=durations, pitches=pitches, energies=energies

        )
    @classmethod
    def _from_tsv(
        cls,
        root: str,
        cfg: S2STDataConfig,
        split: str,
        src_dict,
        tgt_dict,
        is_train_split: bool,
        pre_tokenizer,
        bpe_tokenizer,
        n_frames_per_step,
        speaker_to_id
    ) -> SpeechToTextDataset:
        samples = cls._load_samples_from_tsv(root, split)
        return cls._from_list(
            split, is_train_split, samples, cfg, src_dict,tgt_dict, pre_tokenizer,
            bpe_tokenizer, n_frames_per_step, speaker_to_id
        )
    @classmethod
    def from_tsv(
        cls,
        root: str,
        cfg: S2STDataConfig,
        splits: str,
        src_dict,
        tgt_dict,
        pre_tokenizer,
        bpe_tokenizer,
        is_train_split: bool,
        epoch: int,
        seed: int,
        n_frames_per_step: int = 1,
        speaker_to_id=None
    ) -> SpeechToTextDataset:
        datasets = [
            cls._from_tsv(
                root, cfg, split, src_dict, tgt_dict, is_train_split, pre_tokenizer,
                bpe_tokenizer, n_frames_per_step, speaker_to_id
            )
            for split in splits.split(",")
        ]

        if is_train_split and len(datasets) > 1 and cfg.sampling_alpha != 1.0:
            # temperature-based sampling
            size_ratios = cls.get_size_ratios(datasets, alpha=cfg.sampling_alpha)
            datasets = [
                ResamplingDataset(
                    d, size_ratio=r, seed=seed, epoch=epoch, replace=(r >= 1.0)
                )
                for r, d in zip(size_ratios, datasets)
            ]

        return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]