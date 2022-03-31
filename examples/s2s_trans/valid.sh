#!/usr/bin/env bash
exp_name=$1
cd /opt/tiger/${exp_name}
while [ '1' -eq '1' ];
do
  vars=`ls -lR checkpoint[0-9]*.pt| grep "^-" | wc -l`
  if [ $vars -gt 15 ];then
    bash /opt/tiger/s2st_fairseq/run.sh --stage 6 \
    --stop_stage 11 \
    --mode debug \
    --s2st_exp_name ${exp_name} \
    --s2st_save_dir /opt/tiger/${exp_name} \
    --infer_gpus 0 >> /opt/tiger/${exp_name}/infer.log 2>&1
    bash /opt/tiger/s2st_fairseq/run.sh --stage 12 \
    --stop_stage 12 \
    --mode debug \
    --s2st_exp_name ${exp_name} \
    --s2st_save_dir /opt/tiger/${exp_name} >> /opt/tiger/${exp_name}/upload.log 2>&1 &
  fi
done