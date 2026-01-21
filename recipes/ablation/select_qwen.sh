export CUDA_VISIBLE_DEVICES="0,1"
export HF_ENDPOINT=https://hf-mirror.com  

DATA_NAME=(aime24 aime25 amc23 gpqa_d gaokao2023en math500) 
MODEL_PATH="outputs/models/DeepSeek-R1-Distill-Qwen-7B_ip/checkpoints/global_step_405"


# SAVE_DIR="outputs/ablation/wothink-qwen"

# Without Think Token
# for DATA in "${DATA_NAME[@]}"; do

#     if [[ "$DATA" == "aime24" || "$DATA" == "aime25" || "$DATA" == "amc23" ]]; then
#         KEPT_WINDOW=2000
#         PRUNE_KVLENGTH=5000
#         RATIO=0.4
#     elif [[ "$DATA" == "gpqa_d" ]]; then
#         KEPT_WINDOW=2000
#         PRUNE_KVLENGTH=5000
#         RATIO=0.4
#     elif [[ "$DATA" == "gaokao2023en" || "$DATA" == "math500" ]]; then
#         KEPT_WINDOW=1500
#         PRUNE_KVLENGTH=3000
#         RATIO=0.4
#     else
#         echo "⚠️ 未识别的数据集: $DATA，使用默认参数"
#         KEPT_WINDOW=2000
#         PRUNE_KVLENGTH=5000
#         RATIO=0.4
#     fi

#     python eval.py \
#         --method="window" \
#         --sink_type="question" \
#         --model_name_or_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
#         --data_dir="datas/dataset/$DATA" \
#         --save_dir=$SAVE_DIR \
#         --batch_size=20 \
#         --device_parallel_size=2 \
#         --num_samples=5 \
#         --max_new_tokens=16384\
#         --prune_step=2000 \
#         --kept_window=$KEPT_WINDOW \
#         --prune_signal='kvlength' \
#         --prune_kvlength=$PRUNE_KVLENGTH \
#         --end=-1 
#     done

# done


# # DynTS-Random
# SAVE_DIR="outputs/ablation/qwen/random"
# for DATA in "${DATA_NAME[@]}"; do

#     if [[ "$DATA" == "aime24" || "$DATA" == "aime25" || "$DATA" == "amc23" ]]; then
#         KEPT_WINDOW=2000
#         PRUNE_KVLENGTH=5000
#         RATIO=0.4
#     elif [[ "$DATA" == "gpqa_d" ]]; then
#         KEPT_WINDOW=2000
#         PRUNE_KVLENGTH=5000
#         RATIO=0.4
#     elif [[ "$DATA" == "gaokao2023en" || "$DATA" == "math500" ]]; then
#         KEPT_WINDOW=1500
#         PRUNE_KVLENGTH=3000
#         RATIO=0.4
#     else
#         echo "⚠️ 未识别的数据集: $DATA，使用默认参数"
#         KEPT_WINDOW=2000
#         PRUNE_KVLENGTH=5000
#         RATIO=0.4
#     fi

#     python eval.py \
#         --method="dynts-random" \
#         --ratio=$RATIO \
#         --head_type="regression" \
#         --model_name_or_path="$MODEL_PATH" \
#         --data_dir="datas/dataset/$DATA" \
#         --save_dir=$SAVE_DIR \
#         --batch_size=20 \
#         --device_parallel_size=2 \
#         --num_samples=5 \
#         --max_new_tokens=16384\
#         --prune_step=2000 \
#         --kept_window=$KEPT_WINDOW \
#         --prune_signal='kvlength' \
#         --prune_kvlength=$PRUNE_KVLENGTH 
# done


# # DynTS-Bottom
# SAVE_DIR="outputs/ablation/qwen/bottom"
# for DATA in "${DATA_NAME[@]}"; do

#     if [[ "$DATA" == "aime24" || "$DATA" == "aime25" || "$DATA" == "amc23" ]]; then
#         KEPT_WINDOW=2000
#         PRUNE_KVLENGTH=5000
#         RATIO=0.4
#     elif [[ "$DATA" == "gpqa_d" ]]; then
#         KEPT_WINDOW=2000
#         PRUNE_KVLENGTH=5000
#         RATIO=0.4
#     elif [[ "$DATA" == "gaokao2023en" || "$DATA" == "math500" ]]; then
#         KEPT_WINDOW=1500
#         PRUNE_KVLENGTH=3000
#         RATIO=0.4
#     else
#         echo "⚠️ 未识别的数据集: $DATA，使用默认参数"
#         KEPT_WINDOW=2000
#         PRUNE_KVLENGTH=5000
#         RATIO=0.4
#     fi

#     python eval.py \
#         --method="dynts-bottom" \
#         --ratio=$RATIO \
#         --head_type="regression" \
#         --model_name_or_path="$MODEL_PATH" \
#         --data_dir="datas/dataset/$DATA" \
#         --save_dir=$SAVE_DIR \
#         --batch_size=20 \
#         --device_parallel_size=2 \
#         --num_samples=5 \
#         --max_new_tokens=16384\
#         --prune_step=2000 \
#         --kept_window=$KEPT_WINDOW \
#         --prune_signal='kvlength' \
#         --prune_kvlength=$PRUNE_KVLENGTH 
# done


# DynTS-noque
SAVE_DIR="outputs/ablation/qwen/noque"
for DATA in "${DATA_NAME[@]}"; do

    if [[ "$DATA" == "aime24" || "$DATA" == "aime25" || "$DATA" == "amc23" ]]; then
        KEPT_WINDOW=2000
        PRUNE_KVLENGTH=5000
        RATIO=0.4
    elif [[ "$DATA" == "gpqa_d" ]]; then
        KEPT_WINDOW=2000
        PRUNE_KVLENGTH=5000
        RATIO=0.4
    elif [[ "$DATA" == "gaokao2023en" || "$DATA" == "math500" ]]; then
        KEPT_WINDOW=1500
        PRUNE_KVLENGTH=3000
        RATIO=0.4
    else
        echo "⚠️ 未识别的数据集: $DATA，使用默认参数"
        KEPT_WINDOW=2000
        PRUNE_KVLENGTH=5000
        RATIO=0.4
    fi

    python eval.py \
        --method="dynts-noque" \
        --ratio=$RATIO \
        --head_type="regression" \
        --model_name_or_path="$MODEL_PATH" \
        --data_dir="datas/dataset/$DATA" \
        --save_dir=$SAVE_DIR \
        --batch_size=20 \
        --device_parallel_size=2 \
        --num_samples=5 \
        --max_new_tokens=16384\
        --prune_step=2000 \
        --kept_window=$KEPT_WINDOW \
        --prune_signal='kvlength' \
        --prune_kvlength=$PRUNE_KVLENGTH 
done