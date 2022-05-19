This is an implementation of the paper, based on the [Fairseq](https://github.com/pytorch/fairseq). 
If you have any questions, please email to us (11930381@mail.sustech.edu.cn, dongqianqian@bytedance.com).
# Data Preparation: 
train.tsv(format): 
id      src_audio       tgt_audio       src_n_frames    tgt_n_frames    src_text        speaker tgt_text 
0050908_182943_22_fsp-A-000055-000156  fisher_spanish/feature_manifest_src_tgt/src_fbank80.zip:15791794419:31808  fisher_spanish/feature_manifest_src_tgt/tgt_logmelspec80.zip:10869679859:18048       99      56      a l o | fisher_spanish  HH AH0 L OW1 | ?
