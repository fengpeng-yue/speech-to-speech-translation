#! /usr/bin/env bash

set -x

stage=4
stop_stage=4


AUDIO_MANIFEST_ROOT=/path/fisher_mls_mix_audio
FEATURE_MANIFEST_ROOT=/path/fisher_mls_mix_feature
mode=train  # train or debug
fairseq_dir=/opt/tiger/s2st_fairseq




# setting for s2st experiment
s2st_exp_name=fisher_mls_mix_translatotron_baseline
arch=s2st_transformer
s2st_save_dir=/opt/tiger/$s2st_exp_name
s2st_max_tokens=60000
s2st_max_update=100000
s2st_warmup_updates=4000
ctc_weight=0.0
asr_ce_weight=0.3
st_ce_weight=0.3
middle_layers=4,9
s2st_lr=1.5e-3
s2st_clip_norm=1.0
encoder_embed_dim=512
decoder_embed_dim=512
encoder_ffn_embed_dim=2048
decoder_ffn_embed_dim=2048
update_freq=1 
prenet_dim=32 # follow the google's paper setting
max_source_positions=3000
dropout=0.1
log_path=${s2st_save_dir}/log.txt
tensorboard_path=${s2st_save_dir}/tensorboard
aux_asr_decoder=6
aux_st_decoder=6
asr_decoder_embed_dim=256
st_decoder_embed_dim=256
pretrainmodel=
s2tmodel=
decoder_attention_heads=4
encoder_attention_heads=4
asr_decoder_attention_heads=4
st_decoder_attention_heads=4
train_gpus=1,2,3,4
infer_gpus=0


. ${fairseq_dir}/examples/s2s_trans/parse_options.sh
echo $stage

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    python3 -m examples.s2s_trans.preprocessing.get_ted_en2zhdefrjp_audio_manifest \
   --output-data-root-src ${SRC_AUDIO_DATA_ROOT} \
   --output-data-root-tgt ${TGT_AUDIO_DATA_ROOT} \
   --output-manifest-root-src-tgt ${AUDIO_MANIFEST_ROOT}
fi
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    nj=40
    AUDIO_MANIFEST_ROOT=/opt/tiger/audio_manifest_src_tgt
    for x in tst dev; do
        mkdir -p $AUDIO_MANIFEST_ROOT/split_${nj}/${x}/log
        python3 -m examples.s2s_trans.preprocessing.split_file \
                --type split \
                --n ${nj} \
                --input-file $AUDIO_MANIFEST_ROOT/${x}.audio.tsv \
                --output-path $AUDIO_MANIFEST_ROOT/split_${nj}/${x}
        ./parallel.pl JOB=1:${nj}  $AUDIO_MANIFEST_ROOT/split_${nj}/$x/log/log.JOB\
            preprocessing/g2p.py --input-file $AUDIO_MANIFEST_ROOT/split_${nj}/${x}/${x}.audio.JOB.tsv \
                   --output-file $AUDIO_MANIFEST_ROOT/split_${nj}/${x}/${x}.audio_phone.JOB.tsv

        python3 -m examples.s2s_trans.preprocessing.split_file \
            --type contat \
            --file-part  $AUDIO_MANIFEST_ROOT/split_${nj}/${x}/${x}.audio_phone.*.tsv \
            --output-path $AUDIO_MANIFEST_ROOT
    done
fi
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    python3 -m examples.s2s_trans.preprocessing.get_feature_manifest_8k \
    --audio-manifest-root ${AUDIO_MANIFEST_ROOT} \
    --output-root ${FEATURE_MANIFEST_ROOT} \
    --ipa-vocab --use-g2p \
    --splits "fisher_dev2" "fisher_dev" "fisher_test" "dev" "test" "fisher_train" "train"
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    # setting for s2st experiment
    export CUDA_VISIBLE_DEVICES=${train_gpus}
    SAVE_DIR=${s2st_save_dir}/st_pretraining/
    python3 -m fairseq_cli.train ${FEATURE_MANIFEST_ROOT} --save-dir ${SAVE_DIR} \
    --user-dir /opt/tiger/s2st_fairseq/examples/s2s_trans \
    --config-yaml config.yaml --train-subset train_fisher_mls --valid-subset dev_fisher \
    --num-workers 4 --max-tokens ${s2st_max_tokens} --max-update ${s2st_max_update} \
    --task s2s_translation --criterion s2t_loss --arch s2t_transformer_hubert \
    --clip-norm ${s2st_clip_norm} --n-frames-per-step 4 \
    --dropout ${dropout} --attention-dropout 0.1 --activation-dropout 0.01 \
    --encoder-normalize-before --decoder-normalize-before \
    --optimizer adam --lr ${s2st_lr} --lr-scheduler inverse_sqrt --warmup-updates ${s2st_warmup_updates} \
    --seed 1 --update-freq ${update_freq} --eval-inference --best-checkpoint-metric mcd_loss \
    --use-hubert ${use_hubert} \
    --label-smoothing 0.1 --report-accuracy \
    --skip-invalid-size-inputs-valid-test \
    --log-file ${log_path} --log-format json --tensorboard-logdir ${s2st_save_dir}/st_pretraining/tensorboard \
    --max-source-positions ${max_source_positions} \
    --fp16 --find-unused-parameters \
    --validate-after-updates 300000 \
    --disable-validation \
    --keep-best-checkpoints 50 \
    --keep-last-epochs 50 \
    --save-interval-updates 1000
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    # setting for s2st experiment
    export CUDA_VISIBLE_DEVICES=${train_gpus}
    SAVE_DIR=${s2st_save_dir}
    python3 -m fairseq_cli.train ${FEATURE_MANIFEST_ROOT} --save-dir ${SAVE_DIR} \
    --user-dir /opt/tiger/s2st_fairseq/examples/s2s_trans \
    --config-yaml config.yaml --train-subset train_fisher_mls_upsample --valid-subset dev_fisher \
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
    --load-pretrained-encoder-from ${s2st_save_dir}/st_pretraining/checkpoint_last.pt \
    --load-pretrained-decoder-from ${s2st_save_dir}/st_pretraining/checkpoint_last.pt \
    --keep-best-checkpoints 50 \
    --keep-last-epochs 50 \
    --encoder-attention-heads ${encoder_attention_heads} \
    --decoder-attention-heads ${decoder_attention_heads} 
    --decoder-ffn-embed-dim ${decoder_ffn_embed_dim} \
    --asr-decoder-attention-heads ${asr_decoder_attention_heads} \
    --asr-decoder-ffn-embed-dim ${asr_decoder_ffn_embed_dim} \
    --st-decoder-attention-heads ${st_decoder_attention_heads} \
    --st-decoder-ffn-embed-dim ${st_decoder_ffn_embed_dim}
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    export CUDA_VISIBLE_DEVICES=${infer_gpus}
    SAVE_DIR=${s2st_save_dir}
    # SPLIT=test_fisher
    SPLIT=test_fisher
    CHECKPOINT_NAME=last_avg15_$(date "+%Y-%m-%d-%H-%M-%S")
    CHECKPOINT_PATH=${SAVE_DIR}/checkpoint_${CHECKPOINT_NAME}.pt
    python3 ${fairseq_dir}/scripts/average_checkpoints.py --inputs ${SAVE_DIR} \
    --num-epoch-checkpoints 15 \
    --output ${CHECKPOINT_PATH}
    
    python3 /opt/tiger/s2st_fairseq/examples/s2s_trans/convert_pt_to512.py ${CHECKPOINT_PATH} ${SAVE_DIR}/checkpoint_${CHECKPOINT_NAME}_to512.pt
    mv ${SAVE_DIR}/checkpoint_${CHECKPOINT_NAME}.pt ${SAVE_DIR}/checkpoint_${CHECKPOINT_NAME}_raw.pt
    mv ${SAVE_DIR}/checkpoint_${CHECKPOINT_NAME}_to512.pt ${SAVE_DIR}/checkpoint_${CHECKPOINT_NAME}.pt
    cp ${SAVE_DIR}/checkpoint_${CHECKPOINT_NAME}.pt ${SAVE_DIR}/checkpoint_last_avg15.pt

fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    export CUDA_VISIBLE_DEVICES=${infer_gpus}
    # SAVE_DIR=/opt/tiger/exp/mls_spanish_pretrain_encoder_ce_loss_1_enc-layer4_dec-layer9_1221_wohubert/
    SAVE_DIR=${s2st_save_dir}
    SPLIT=test_fisher
    CHECKPOINT_FILENAME=checkpoint_last_avg15.pt
    CHECKPOINT_PATH=${SAVE_DIR}/${CHECKPOINT_FILENAME}
    python3 -m examples.s2s_trans.generate_waveform ${FEATURE_MANIFEST_ROOT} \
    --user-dir /opt/tiger/s2st_fairseq/examples/s2s_trans \
    --config-yaml config.yaml --gen-subset ${SPLIT} --task s2s_translation \
    --use-hubert False \
    --path ${CHECKPOINT_PATH} --max-tokens 60000 --spec-bwd-max-iter 64 \
    --n-frames-per-step 4  \
    --dump-waveforms --dump-attentions --dump-features --dump-plots --dump-target \
    --results-path ${SAVE_DIR}/dump_wav_${SPLIT}_${CHECKPOINT_FILENAME}
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    export CUDA_VISIBLE_DEVICES=${infer_gpus}
    SAVE_DIR=${s2st_save_dir}
    SPLIT=test_fisher
    CHECKPOINT_FILENAME=checkpoint_last_avg15.pt
    python3 ${fairseq_dir}/examples/s2s_trans/evalute_s2s_bleu.py --audio_manifest_file   ${AUDIO_MANIFEST_ROOT}/fisher_test.audio.tsv \
                                    --decode_save_path ${SAVE_DIR}/dump_wav_${SPLIT}_${CHECKPOINT_FILENAME} \
                                    --out_result_file ${SAVE_DIR}/dump_wav_${SPLIT}_${CHECKPOINT_FILENAME}/decode.txt \
                                    --scoring sacrebleu
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    echo "Evaluate multi-references BLEU"
    SAVE_DIR=${s2st_save_dir}
    SPLIT=test_fisher
    CHECKPOINT_FILENAME=checkpoint_last_avg15.pt
    cut -f1 ${SAVE_DIR}/dump_wav_${SPLIT}_${CHECKPOINT_FILENAME}/decode.txt > ${SAVE_DIR}/dump_wav_${SPLIT}_${CHECKPOINT_FILENAME}/decode.txt.prediction
    # put  reference text (en.0.norm.lc.rm en.1.norm.lc.rm en.2.norm.lc.rm en.3.norm.lc.rm) under current dir.
    sacrebleu en.0.norm.lc.rm en.1.norm.lc.rm en.2.norm.lc.rm en.3.norm.lc.rm -i ${SAVE_DIR}/dump_wav_${SPLIT}_${CHECKPOINT_FILENAME}/decode.txt.prediction -lc
fi

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
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
fi

if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
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
fi
