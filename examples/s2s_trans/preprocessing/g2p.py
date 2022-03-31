#!/usr/bin/env python3
import os
import csv
import argparse
from tqdm import tqdm
import pandas as pd
from collections import Counter, defaultdict
from examples.speech_to_text.data_utils import(
    load_tsv_to_dicts,
    save_df_to_tsv
)
from examples.s2s_trans.preprocessing.data_utils import ipa_phonemize
from pypinyin import pinyin
from pypinyin import Style
from examples.s2s_trans.preprocessing.cn_tn import run_cn_tn
import tacotron_cleaner
import re
def pypinyin_g2p_phone(text):
    from pypinyin import pinyin
    from pypinyin import Style
    from pypinyin.style._utils import get_finals
    from pypinyin.style._utils import get_initials

    phones = [
        p
        for phone in pinyin(text, style=Style.TONE3)
        for p in [
            get_initials(phone[0], strict=True),
            get_finals(phone[0], strict=True),
        ]
        if len(p) != 0
    ]
    return phones
# check english character
en_pattern = re.compile(r'[A-Za-z]',re.S)
# 去掉标点
reg = "[^0-9A-Za-z\u4e00-\u9fa5]"
def process(args):
    samples = []
    for e in load_tsv_to_dicts(args.input_file):
        samples.append(e)
    total_nums = len(samples)
        
    manifest_by_split=defaultdict(list)
    for j in tqdm(range(total_nums)):
        res = re.findall(en_pattern,samples[j]["tgt_text"])
        if len(res) > 0:
            print("find english character at id %s, context is %s" %(samples[j]["id"], samples[j]["tgt_text"]))
            continue
        #qprint("doing")
        normalized_src_utt = tacotron_cleaner.cleaners.custom_english_cleaners(samples[j]["src_text"])
        #print(normalized_src_utt)
        normalized_src_utt = ipa_phonemize(
                normalized_src_utt , lang="en-us", use_g2p=True
            )
        #print(normalized_src_utt)
        #print(samples[j]["tgt_text"])
        normalized_tgt_utt = run_cn_tn(samples[j]["tgt_text"].replace(" ",""))
        normalized_tgt_utt = re.sub(reg, '', normalized_tgt_utt)
        #print(normalized_tgt_utt)
        space_normalized_tgt_utt = ""
        for i in range(len(normalized_tgt_utt)):
            if i == len(normalized_tgt_utt) -1:
                space_normalized_tgt_utt = space_normalized_tgt_utt + normalized_tgt_utt[i]
            else:
                space_normalized_tgt_utt = space_normalized_tgt_utt + normalized_tgt_utt[i] + "|"
        #print(space_normalized_tgt_utt)
        space_normalized_tgt_utt = (" ").join(pypinyin_g2p_phone(space_normalized_tgt_utt))
        #print(space_normalized_tgt_utt)

        manifest_by_split["id"].append(samples[j]["id"])
        manifest_by_split["src_audio"].append(samples[j]["src_audio"])
        manifest_by_split["src_n_frames"].append(samples[j]["src_n_frames"])
        manifest_by_split["src_text"].append(normalized_src_utt)
        manifest_by_split["tgt_audio"].append(samples[j]["tgt_audio"])
        manifest_by_split["tgt_n_frames"].append(samples[j]["tgt_n_frames"])
        manifest_by_split["tgt_text"].append(space_normalized_tgt_utt)
        manifest_by_split["speaker"].append(samples[j]["speaker"])

    save_df_to_tsv(
        pd.DataFrame.from_dict(manifest_by_split),
        args.output_file
        )




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True, type=str)
    parser.add_argument("--output-file",required=True, type=str)


    args = parser.parse_args()

    process(args)

if __name__ == "__main__":
    main()
