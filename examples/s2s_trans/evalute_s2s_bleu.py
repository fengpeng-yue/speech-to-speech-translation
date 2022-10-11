# !pip install transformers
# !pip install datasets
import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
import csv
import argparse
import os
from fairseq import scoring
from fairseq.scoring.tokenizer import EvaluationTokenizer

# os.environ['CUDA_VISIBLE_DEVICES']= '7'

def recognize(model, processor, batch, sr):

    # pad input values and return pt tensor
    inputs = processor(batch, sampling_rate=sr, return_tensors="pt", padding="longest")
    input_values = inputs.input_values.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

    # INFERENCE

    # retrieve logits & take argmax
    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits
    predicted_ids = torch.argmax(logits, dim=-1)

    # transcribe
    transcription = processor.batch_decode(predicted_ids)

    # print(transcription)
    return transcription

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_manifest_file', type=str)
    parser.add_argument('--decode_save_path', type=str)
    parser.add_argument('--decode_save_path_subdir', type=str, default="wav_24000hz_griffin_lim")
    parser.add_argument('--out_result_file', type=str)
    parser.add_argument('--scoring', type=str)
    parser.add_argument('--batch_size', type=int, default=160000)
    parser.add_argument('--punctuation_removal', type=bool, default=True)

    args = parser.parse_args()
    # load pretrained model
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self").to("cuda")
    scorer = scoring.build_scorer(args.scoring, None)
    with open(args.audio_manifest_file, 'r') as fin:
        flist = csv.reader(fin, delimiter="\t", quoting=csv.QUOTE_NONE)
        _flist = list(flist)

    def gen_batch(_flist):
        batch = []
        lens = []
        gt_transcriptions = []
        for i in range(1, len(_flist)):
            audio_path = os.path.join(args.decode_save_path, args.decode_save_path_subdir, _flist[i][0]+".wav")
            signals, sample_rate = librosa.load(audio_path, sr=16000)
            lens.append(len(signals))
            batch.append(signals)
            gt_transcriptions.append(_flist[i][6])
            if len(batch) * max(lens) <= args.batch_size and i < len(_flist) -1:
                continue
            yield batch, gt_transcriptions
            print(f"Processed: {i}th example.")
            batch = []
            lens = []
            gt_transcriptions = []
    fout = open(args.out_result_file, 'w')
    for batch, gt_transcriptions in gen_batch(_flist):
        transcriptions = recognize(model, processor, batch, 16000)
        for transcription, gt_transcription in zip(transcriptions, gt_transcriptions):
            if args.punctuation_removal:
                transcription = EvaluationTokenizer.remove_punctuation(transcription)
                gt_transcription = EvaluationTokenizer.remove_punctuation(gt_transcription)
            print('\t'.join([transcription.lower(), gt_transcription.lower()]), file=fout)
            scorer.add_string(gt_transcription.lower(), transcription.lower())
    fout.close()
    sacrebleu = scorer.result_string(4)
    print(f"Total Sentences: {len(_flist)-1}, Sacrebleu: {sacrebleu}")
