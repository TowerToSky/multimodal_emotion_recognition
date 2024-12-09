#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo $DIR

dataset="HCI"
# modalities="eeg, eye, au"
modalities=("eeg, eye, pps")
# modalities=("eeg" "eye" "pps")
for label_type in "arousal" "valence"
do
    for cls_num in 3 #2 4
    do
        for dependent in 0 # 1
        do
            for modality in "${modalities[@]}"
            do
                sleep 2
                echo "Dataset: $dataset, Num Classes: $cls_num, Dependent: $dependent, Label Type: $label_type, Modality: $modality"
                /home/yihaoyuan/miniconda3/envs/torch/bin/python $DIR/../main.py \
                --data $dataset \
                --num_classes $cls_num \
                --label_type $label_type \
                --dependent $dependent \
                --using_modality "$modality" &
            done
        done
    done
done




