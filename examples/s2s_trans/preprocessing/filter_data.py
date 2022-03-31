import os
import csv
import argparse
import pandas as pd
from collections import Counter, defaultdict
from tqdm import tqdm
from examples.speech_to_text.data_utils import(
    load_tsv_to_dicts,
    save_df_to_tsv
)
source_file = "/opt/tiger/feature_manifest_src_tgt/dev.tsv"
filter_file="/opt/tiger/data_feature/ted-enzh-badcase"
reference_file = "/opt/tiger/audio_manifest_src_tgt/fisher_dev2.audio.tsv"
#reference_file = 
file_name ="/opt/tiger/feature_manifest_src_tgt/dev_kd.tsv"

# filter_list = []
# with open(filter_file,"r") as f:
#     lines = f.readlines()
#     for line in lines:
#         filter_list.append(line.strip('\n').split(",")[0].replace(" ","") + ".wav")
samples = []
manifest_by_split=defaultdict(list)
referece_dict = {}
for e in load_tsv_to_dicts(source_file):
    samples.append(e)
for e in load_tsv_to_dicts(reference_file):
    referece_dict[e["id"]] = [e["tgt_text"]]
    #referece_dict[e["id"]] = e["src_audio"]
#print(referece_dict)
filter_num = 0
for sample in tqdm(samples):
    src_n_frames = sample["src_n_frames"]
    tgt_n_frames = sample["tgt_n_frames"]
    idx = sample["id"]
    if not idx in referece_dict:
        continue
    # print(idx)
    # print(filter_list[:10])
    # if idx in filter_list:    
    #     filter_num += 1
    #     continue
    # if int(tgt_n_frames) > 800:
    #     continue
    # if idx in referece_dict.keys():
    #     src_text = referece_dict[idx][0]
    #     tgt_text = referece_dict[idx][1].replace("| | |", "|")
    # else:
    #     filter_num += 1
    #     continue
    # src_text = sample["src_text"]
    # tgt_text = sample["tgt_text"]
    #src_text = referece_dict[idx][0]
    tgt_text = referece_dict[idx][0]
    manifest_by_split["id"].append(idx)
    #manifest_by_split["src_orig"].append(sample["src_orig"])
    manifest_by_split["src_audio"].append(sample["src_audio"])
    manifest_by_split["src_n_frames"].append(sample["src_n_frames"])
    manifest_by_split["src_text"].append(sample["src_text"])
    manifest_by_split["tgt_audio"].append(sample["tgt_audio"])
    manifest_by_split["tgt_n_frames"].append(sample["tgt_n_frames"])
    manifest_by_split["tgt_text"].append(sample["tgt_text"])
    manifest_by_split["tgt_text_orig"].append(tgt_text)
    manifest_by_split["speaker"].append(sample["speaker"])
#tgt_vocab = Counter()
# for t in manifest_by_split["tgt_text"]:
#     tgt_vocab.update(t.split(" "))
# tgt_vocab_name = "tgt_vocab.txt"
# with open("/opt/tiger/" + tgt_vocab_name, "w") as f:
#     for s, c in tgt_vocab.most_common():
#         f.write(f"{s} {c}\n")
# print(filter_num)
save_df_to_tsv(
    pd.DataFrame.from_dict(manifest_by_split),
    file_name
    )
