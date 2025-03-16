#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo $DIR

# dataset="HCI"
dataset="Ruiwen"
modalities=("eeg, eye, au")

if [ "$dataset" == "HCI" ]; then
    modalities=("eye, pps")
    label_types=("valence")
    cls_nums=(3)
elif [ "$dataset" == "Ruiwen" ]; then
    modalities=("eye, au")
    label_types=("ruiwen")
    cls_nums=(2 4)
fi
sleep 2


checkpoints=("2024-12-24_00-46-27" "2024-12-24_00-46-29")

for index in "${!checkpoints[@]}";
do
    checkpoint=${checkpoints[$index]}
    label_type="ruiwen"
    cls_num=${cls_nums[$index]}
    echo "Dataset: $dataset, Num Classes: $cls_num, Label Type: $label_type, Checkpoint: $checkpoint, Modalities: $modalities"

    /home/yihaoyuan/miniconda3/envs/torch/bin/python $DIR/../ctfn_main_new.py \
        --data $dataset \
        --num_classes $cls_num \
        --label_type $label_type \
        --using_modality "$modalities" \
        --checkpoint "$checkpoint" &
    sleep 2 
done







