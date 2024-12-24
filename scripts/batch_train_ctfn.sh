#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo $DIR

# dataset="HCI"
dataset="Ruiwen"
modalities=("eeg, eye, au")

if [ "$dataset" == "HCI" ]; then
    modalities=("eeg, eye, pps")
    label_types=("arousal" "valence")
    cls_nums=(3)
elif [ "$dataset" == "Ruiwen" ]; then
    modalities=("eeg, eye, au")
    label_types=("ruiwen")
    cls_nums=(2 4)
fi

# modalities="eeg, eye, au"
# modalities=("eeg, eye, pps")
# modalities=("eeg, eye, au")
# modalities=("eeg" "eye" "pps")
for label_type in "${label_types[@]}"
do
    for cls_num in "${cls_nums[@]}"
    do
        for dependent in 0 #1
        do
            for modality in "${modalities[@]}"
            do
                for seq_len in 10 #10 6 8 12 16
                do
                    echo "Dataset: $dataset, Num Classes: $cls_num, Dependent: $dependent, Label Type: $label_type, Modality: $modality, Seq Len: $seq_len"
                    /home/yihaoyuan/miniconda3/envs/torch/bin/python $DIR/../ctfn_main_new.py \
                    --data $dataset \
                    --num_classes $cls_num \
                    --label_type $label_type \
                    --dependent $dependent \
                    --using_modality "$modality" &
                    sleep 2                    
                done
            done
        done
    done
done




