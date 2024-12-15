#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo $DIR

dataset="Ruiwen"
modalities="eeg, eye, au"
cls_num=2
checkpoint="2024-12-12_23-49-21"

/home/yihaoyuan/miniconda3/envs/torch/bin/python $DIR/../cross_main.py \
                            --data $dataset \
                            --using_modality="$modalities" \
                            --num_classes $cls_num \
                            --checkpoint "$checkpoint" &
cls_num=4
checkpoint="2024-12-12_23-49-26"
sleep 2

/home/yihaoyuan/miniconda3/envs/torch/bin/python $DIR/../cross_main.py \
                            --data $dataset \
                            --using_modality="$modalities" \
                            --num_classes $cls_num \
                            --checkpoint "$checkpoint" &

dataset="HCI"
modalities="eeg, eye, pps"
cls_num=3

sleep 2

checkpoint="2024-12-09_20-26-19"
label_type="arousal"
/home/yihaoyuan/miniconda3/envs/torch/bin/python $DIR/../cross_main.py \
                            --data $dataset \
                            --num_classes $cls_num \
                            --label_type $label_type \
                            --using_modality "$modalities" \
                            --checkpoint "$checkpoint" &

sleep 2
checkpoint="2024-12-09_20-26-21"
label_type="valence"
/home/yihaoyuan/miniconda3/envs/torch/bin/python $DIR/../cross_main.py \
                            --data $dataset \
                            --num_classes $cls_num \
                            --label_type $label_type \
                            --using_modality "$modalities" \
                            --checkpoint "$checkpoint" &





