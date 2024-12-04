#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo $DIR

# 递归生成所有可能的组合
generate_combinations() {
    local prefix="$1"
    shift

    if [ $# -eq 0 ]; then
        echo "$prefix"
        if [ -n "$prefix" ]; then
            for cls_num in 2 4
            do
                for dependent in 0 #1
                do
                    sleep 2
                    echo "Modality: $prefix, Num Classes: $cls_num, Dependent: $dependent"
                    /home/yihaoyuan/miniconda3/envs/torch/bin/python $DIR/../main.py \
                    --data "Ruiwen" \
                    --num_classes $cls_num \
                    --dependent $dependent \
                    --using_modality "$prefix" &
                done
            done
            # python3 test.py --using_modality "$prefix"
        fi
        # python3 test.py --using_modality "$prefix"
    else
        local first="$1"
        shift

        generate_combinations "$prefix$first, " "$@"
        generate_combinations "$prefix" "$@"
    fi
}

# 定义模态数组
modalities=("eeg" "eye" "au")
# modalities=("eeg" "au")

# 生成并输出所有可能的组合
generate_combinations "" "${modalities[@]}"
