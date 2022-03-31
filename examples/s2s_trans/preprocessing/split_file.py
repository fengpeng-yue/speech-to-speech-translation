import os
import csv
import argparse
import pandas as pd
from collections import Counter, defaultdict
from examples.speech_to_text.data_utils import(
    load_tsv_to_dicts,
    save_df_to_tsv
)
def process(args):
    samples = []
    manifest_by_split=defaultdict(list)
    if args.type == "split":
        for e in load_tsv_to_dicts(args.input_file):
            samples.append(e)
        total_nums = len(samples)
        avg_nums = total_nums // args.n
        files_n = [ avg_nums ] * args.n
        files_n[args.n - 1]  += total_nums % args.n
        for i in range(1,args.n):
            files_n[i] += files_n[i-1]
        files_n.insert(0,0)
        for i in range(args.n):
            file_name = args.output_path + "/" + os.path.basename(args.input_file).split(".")[0] + \
                        ".audio.%s"%(i+1) + ".tsv"
            #print(file_name)      
            manifest_by_split=defaultdict(list)
            for j in range(files_n[i],files_n[i+1]):
                src_n_frames = samples[j]["src_n_frames"]
                tgt_n_frames = samples[j]["tgt_n_frames"]
                manifest_by_split["id"].append(samples[j]["id"])
                manifest_by_split["src_audio"].append(samples[j]["src_audio"])
                manifest_by_split["src_n_frames"].append(samples[j]["src_n_frames"])
                manifest_by_split["src_text"].append(samples[j]["src_text"])
                manifest_by_split["tgt_audio"].append(samples[j]["tgt_audio"])
                manifest_by_split["tgt_n_frames"].append(samples[j]["tgt_n_frames"])
                manifest_by_split["tgt_text"].append(samples[j]["tgt_text"])
                manifest_by_split["speaker"].append(samples[j]["speaker"])

            save_df_to_tsv(
                pd.DataFrame.from_dict(manifest_by_split),
                file_name
                )
    else:
        for file in args.file_part:
            for sample in load_tsv_to_dicts(file):
                src_n_frames = sample["src_n_frames"]
                tgt_n_frames = sample["tgt_n_frames"]
                # if int(src_n_frames) < 16000 or int(tgt_n_frames)<16000:
                #     print(sample["id"])
                #     continue
                manifest_by_split["id"].append(sample["id"])
                manifest_by_split["src_audio"].append(sample["src_audio"])
                manifest_by_split["src_n_frames"].append(sample["src_n_frames"])
                manifest_by_split["src_text"].append(sample["src_text"])
                manifest_by_split["tgt_audio"].append(sample["tgt_audio"])
                manifest_by_split["tgt_n_frames"].append(sample["tgt_n_frames"])
                manifest_by_split["tgt_text"].append(sample["tgt_text"])
                manifest_by_split["speaker"].append(sample["speaker"])
        file_name = args.output_path + "/" + os.path.basename(args.file_part[0]).split(".")[0] + \
                        ".audio_phone.tsv"
        save_df_to_tsv(
            pd.DataFrame.from_dict(manifest_by_split),
            file_name
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--output-path", required=True, type=str)
    parser.add_argument("--file-part", "-s", type=str, nargs="+")
    parser.add_argument("--type", required=True, type=str,choices=["split","contat"])
    parser.add_argument("--n", type=int)


    args = parser.parse_args()

    process(args)

if __name__ == "__main__":
    main()
