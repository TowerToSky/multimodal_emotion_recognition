#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo $DIR

dataset="HCI"
# dataset="Ruiwen"
# modalities="eeg, eye, au"
# modalities=("eeg, eye, pps")
modalities=("eye, pps")
# modalities=("eeg" "eye" "pps")
for label_type in  "arousal" "valence"  # "ruiwen"
do
    for cls_num in 3 #2 4 
    do
        for dependent in 0 #1
        do
            for modality in "${modalities[@]}"
            do
                for seq_len in 10 #10 6 8 12 16
                do
                    for num_layers in 6 #4 8 10 12 2
                    do
                        echo "Dataset: $dataset, Num Classes: $cls_num, Dependent: $dependent, Label Type: $label_type, Modality: $modality, Seq Len: $seq_len, Num Layers: $num_layers"
                        /home/yihaoyuan/miniconda3/envs/torch/bin/python $DIR/../main.py \
                        --data $dataset \
                        --num_classes $cls_num \
                        --label_type $label_type \
                        --dependent $dependent \
                        --using_modality "$modality" \
                        --seq_len $seq_len \
                        --num_layers $num_layers &
                        sleep 2
                    done
                done
            done
        done
    done
done




