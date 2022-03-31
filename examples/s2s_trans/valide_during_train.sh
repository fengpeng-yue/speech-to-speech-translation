#! /usr/bin/env bash

exp_name=fisher_ce_asr_0.3_st_0.3_decoder6_dropout0.1_lr1.5e-3_small_auxdecoder_2022-01-11-17-32-04
fairseq_dir=/opt/tiger/s2st_fairseq
SAVE_DIR=/opt/tiger/${exp_name}
CHECKPOINT_NAME=last_avg15
CHECKPOINT_PATH=${SAVE_DIR}/checkpoint_${CHECKPOINT_NAME}.pt


. ${fairseq_dir}/examples/s2s_trans/parse_options.sh

python3 ${fairseq_dir}/scripts/average_checkpoints.py --inputs ${SAVE_DIR} \
    --num-epoch-checkpoints 15 \
    --output ${CHECKPOINT_PATH}

python3 /opt/tiger/s2st_fairseq/examples/speech_to_speech_translation/convert_pt_to512.py ${CHECKPOINT_PATH} ${SAVE_DIR}/checkpoint_${CHECKPOINT_NAME}_to512.pt
mv ${SAVE_DIR}/checkpoint_${CHECKPOINT_NAME}.pt ${SAVE_DIR}/checkpoint_${CHECKPOINT_NAME}_raw.pt
mv ${SAVE_DIR}/checkpoint_${CHECKPOINT_NAME}_to512.pt ${SAVE_DIR}/checkpoint_${CHECKPOINT_NAME}.pt

