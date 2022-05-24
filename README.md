This is an implementation of the [paper](https://arxiv.org/abs/2205.08993), based on the [Fairseq](https://github.com/pytorch/fairseq). 
If you have any questions, please email to us (11930381@mail.sustech.edu.cn, dongqianqian@bytedance.com).
# Requirements
Follow the [installation](https://github.com/pytorch/fairseq) method of Fairseq.  
# Data Preparation: 
stage1~stage3 in the ./examples/s2s_trans/run_baseline.sh is used to preprate dataset format.   
train_fisher.tsv (format):  
```
id	src_audio	tgt_audio	src_n_frames	tgt_n_frames	src_text	speaker		tgt_text  
0050908_182943_22_fsp-A-000055-000156   src_fbank80.zip:15791794419:31808  tgt_logmelspec80.zip:10869679859:18048   99  56  a l o | fisher_spanish  HH AH0 L OW1 | ?    
20050908_182943_22_fsp-B-000141-000252  src_fbank80.zip:10950129224:35008  tgt_logmelspec80.zip:7525531784:18048    109 56  a l o | fisher_spanish  HH AH0 L OW1 | ?
```
The preprocessed directory mainfest should look like as follows:
```
├── mainfest
│   ├── config.yaml
│   ├── dev_fisher.tsv
│   ├── dev2_fisher.tsv
│   ├── src_fank80.zip
│   ├── src_gcmvn_stats.npz
│   ├── src_vocab.txt
│   ├── test_fisher.tsv
│   ├── tgt_gcmvn_stats.npz
|   ├── tgt_logmelspec80.zip
│   ├── tgt_vocav.txt
│   ├── train_fisher.tsv

```
The configure file config.yaml should look like as follows:
```
audio_root: to/path/manifest
features:
  eps: 1.0e-05
  f_max: 8000
  f_min: 20
  hop_len_t: 0.0125
  hop_length: 300
  n_fft: 2048
  n_mels: 80
  n_stft: 1025
  sample_rate: 24000
  type: spectrogram+melscale+log
  win_len_t: 0.05
  win_length: 1200
  window_fn: hann
sample_rate: 24000
specaugment:
  freq_mask_F: 27
  freq_mask_N: 2
  time_mask_N: 2
  time_mask_T: 100
  time_mask_p: 1.0
  time_wrap_W: 0
src_global_cmvn:
  stats_npz_path: /to/path/manifest/src_gcmvn_stats.npz
src_vocab_filename: src_vocab.txt
tgt_global_cmvn:
  stats_npz_path: /to/path/manifest/tgt_gcmvn_stats.npz
tgt_vocab_filename: tgt_vocab.txt
src_transforms:
  '*':
  - src_global_cmvn
  _train:
  - src_global_cmvn
  - specaugment
tgt_transforms:
  '*':
  - tgt_global_cmvn
  _train:
  - tgt_global_cmvn
~
```
# Training: 
We give an example based on the fisher dataset to implement our method.  
The ./examples/s2s_trans/run_baseline.sh is a script for training baseline model.    
stage5 is the traininig stage:  
```
    export CUDA_VISIBLE_DEVICES=${train_gpus}
    SAVE_DIR=${s2st_save_dir}
    python3 -m fairseq_cli.train ${FEATURE_MANIFEST_ROOT} --save-dir ${SAVE_DIR} \
    --user-dir /opt/tiger/s2st_fairseq/examples/s2s_trans \
    --config-yaml config.yaml --train-subset train_fisher --valid-subset dev_fisher \
    --num-workers 4 --max-tokens ${s2st_max_tokens} --max-update ${s2st_max_update} \
    --task s2s_translation --criterion s2st_loss --arch s2st_transformer \
    --clip-norm ${s2st_clip_norm} --n-frames-per-step 4 --bce-pos-weight 5.0 \
    --dropout ${dropout} --attention-dropout 0.1 --activation-dropout 0.01 \
    --encoder-normalize-before --decoder-normalize-before \
    --optimizer adam --lr ${s2st_lr} --lr-scheduler inverse_sqrt --warmup-updates ${s2st_warmup_updates} \
    --seed 1 --update-freq ${update_freq} --eval-inference --best-checkpoint-metric mcd_loss \
    --load-pretrained-hubert-from ${hubert_path}/hubert_base_ls960.pt --use-hubert ${use_hubert} \
    --label-smoothing 0.1 --asr-ce-weight ${asr_ce_weight} --st-ce-weight ${st_ce_weight} --report-accuracy \
    --skip-invalid-size-inputs-valid-test --ctc-weight ${ctc_weight}  --middle-layers ${middle_layers} \
    --log-file ${log_path} --log-format json --tensorboard-logdir ${tensorboard_path} \
    --asr-decoder-layers ${aux_asr_decoder} --st-decoder-layers ${aux_st_decoder} \
    --asr-decoder-embed-dim ${asr_decoder_embed_dim} --st-decoder-embed-dim ${st_decoder_embed_dim} \
    --prenet-dim ${prenet_dim} --max-source-positions ${max_source_positions} \
    --fp16 --find-unused-parameters \
    --validate-after-updates 300000 \
    --disable-validation \
    --keep-best-checkpoints 50 \
    --keep-last-epochs 50 \
    --encoder-attention-heads ${encoder_attention_heads} \
    --decoder-attention-heads ${decoder_attention_heads} \
    --decoder-ffn-embed-dim ${decoder_ffn_embed_dim} \
    --asr-decoder-attention-heads ${asr_decoder_attention_heads} \
    --asr-decoder-ffn-embed-dim ${asr_decoder_ffn_embed_dim} \
    --st-decoder-attention-heads ${st_decoder_attention_heads} \
    --st-decoder-ffn-embed-dim ${st_decoder_ffn_embed_dim}
```
The ./examples/s2s_trans/run_pretrainig.sh is a script for pretraining method described in our paper.   
The ./examples/s2s_trans/run_mix_tuning.sh is a script for mix tuning method.   
The ./examples/s2s_trans/run_prompt_tuning.sh is a script for prompg method.   
# Evaluation
generate target speech:
```
   export CUDA_VISIBLE_DEVICES=${infer_gpus}
    SAVE_DIR=${s2st_save_dir}
    SPLIT=test_fisher
    CHECKPOINT_FILENAME=checkpoint_last_avg15.pt
    CHECKPOINT_PATH=${SAVE_DIR}/${CHECKPOINT_FILENAME}
    python3 -m examples.s2s_trans.generate_waveform ${FEATURE_MANIFEST_ROOT} \
    --user-dir /opt/tiger/s2st_fairseq/examples/s2s_trans \
    --config-yaml config.yaml --gen-subset ${SPLIT} --task s2s_translation \
    --use-hubert False \
    --path ${CHECKPOINT_PATH} --max-tokens 100000 --spec-bwd-max-iter 64 \
    --n-frames-per-step 4  \
    --dump-waveforms --dump-attentions --dump-features --dump-plots --dump-target \
    --results-path ${SAVE_DIR}/dump_wav_${SPLIT}_${CHECKPOINT_FILENAME}
```
decode target text:
```
    export CUDA_VISIBLE_DEVICES=${infer_gpus}
    SAVE_DIR=${s2st_save_dir}
    SPLIT=test_fisher
    CHECKPOINT_FILENAME=checkpoint_last_avg15.pt
    python3 ${fairseq_dir}/examples/speech_to_speech_translation/evalute_s2s_bleu.py --audio_manifest_file   ${AUDIO_MANIFEST_ROOT}/fisher_test.audio.tsv \
                                    --decode_save_path ${SAVE_DIR}/dump_wav_${SPLIT}_${CHECKPOINT_FILENAME} \
                                    --out_result_file ${SAVE_DIR}/dump_wav_${SPLIT}_${CHECKPOINT_FILENAME}/decode.txt \
                                    --scoring sacrebleu
```
calculate BLEU score:
```
    echo "Evaluate multi-references BLEU"
    SAVE_DIR=${s2st_save_dir}
    SPLIT=test_fisher
    CHECKPOINT_FILENAME=checkpoint_last_avg15.pt
    cut -f1 ${SAVE_DIR}/dump_wav_${SPLIT}_${CHECKPOINT_FILENAME}/decode.txt > ${SAVE_DIR}/dump_wav_${SPLIT}_${CHECKPOINT_FILENAME}/decode.txt.prediction
    # put  reference text (en.0.norm.lc.rm en.1.norm.lc.rm en.2.norm.lc.rm en.3.norm.lc.rm) under current dir.
    sacrebleu en.0.norm.lc.rm en.1.norm.lc.rm en.2.norm.lc.rm en.3.norm.lc.rm -i ${SAVE_DIR}/dump_wav_${SPLIT}_${CHECKPOINT_FILENAME}/decode.txt.prediction -lc
```
test the WER of auxiliary ASR task:
```
    export CUDA_VISIBLE_DEVICES=${infer_gpus}
    SAVE_DIR=${s2st_save_dir}
    SPLIT=test_fisher
    CHECKPOINT_FILENAME=checkpoint_last_avg15.pt
    CHECKPOINT_PATH=${SAVE_DIR}/${CHECKPOINT_FILENAME}
    python3 /opt/tiger/s2st_fairseq/fairseq_cli/generate_for_s2st.py ${FEATURE_MANIFEST_ROOT} \
    --user-dir /opt/tiger/s2st_fairseq/examples/s2s_trans \
    --config-yaml config.yaml --gen-subset ${SPLIT} --task s2s_translation \
    --use-hubert False \
    --path ${CHECKPOINT_PATH} \
    --max-tokens 50000 --beam 5  \
    --scoring wer --wer-tokenizer 13a --wer-lowercase --wer-remove-punct
```
test the BLEU score of auxiliary MT task:
```
    export CUDA_VISIBLE_DEVICES=${infer_gpus}
    SAVE_DIR=${s2st_save_dir}
    SPLIT=test_fisher
    CHECKPOINT_FILENAME=checkpoint_last_avg15.pt
    CHECKPOINT_PATH=${SAVE_DIR}/${CHECKPOINT_FILENAME}
    python3 /opt/tiger/s2st_fairseq/fairseq_cli/generate_for_s2st.py ${FEATURE_MANIFEST_ROOT} \
    --user-dir /opt/tiger/s2st_fairseq/examples/s2s_trans \
    --config-yaml config.yaml --gen-subset ${SPLIT} --task s2s_translation \
    --use-hubert False \
    --path ${CHECKPOINT_PATH} \
    --max-tokens 50000 --beam 5  \
    --scoring sacrebleu
```
# Citation
```
@article{dong2022leveraging,
  title={Leveraging Pseudo-labeled Data to Improve Direct Speech-to-Speech Translation},
  author={Dong, Qianqian and Yue, Fengpeng and Ko, Tom and Wang, Mingxuan and Bai, Qibing and Zhang, Yu},
  journal={arXiv preprint arXiv:2205.08993},
  year={2022}
}
```
