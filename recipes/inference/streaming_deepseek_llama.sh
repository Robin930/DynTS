export CUDA_VISIBLE_DEVICES="0,1"
export HF_ENDPOINT=https://hf-mirror.com  

# DATA_NAME=(aime24 aime25 amc23 gaokao2023en gpqa_d math500)
DATA_NAME=(aime24)

SAVE_DIR="outputs/inference/streaming-llama"

for DATA in "${DATA_NAME[@]}"; do

    if [[ "$DATA" == "aime24" || "$DATA" == "aime25" || "$DATA" == "amc23" ]]; then
        KEPT_WINDOW=2900 # 2000 + (5000-2000) * 0.3 = 2900
        PRUNE_KVLENGTH=5000
    elif [[ "$DATA" == "gpqa_d" ]]; then
        KEPT_WINDOW=3250 # 3000 + (5000-2500) * 0.3 = 3250
        PRUNE_KVLENGTH=5000
    elif [[ "$DATA" == "gaokao2023en" || "$DATA" == "math500" ]]; then
        KEPT_WINDOW=1600 # 1000 + (3000-1000) * 0.3 = 1600
        PRUNE_KVLENGTH=3000
    else
        echo "⚠️ 未识别的数据集: $DATA，使用默认参数"
        KEPT_WINDOW=2000
        PRUNE_KVLENGTH=5000
    fi

    python eval.py \
        --method="streaming" \
        --sink_type="native" \
        --sink_size=10 \
        --model_name_or_path="deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
        --data_dir="datas/dataset/$DATA" \
        --save_dir=$SAVE_DIR \
        --batch_size=20 \
        --device_parallel_size=2 \
        --num_samples=5 \
        --max_new_tokens=16384\
        --prune_step=2000 \
        --kept_window=$KEPT_WINDOW \
        --prune_signal='kvlength' \
        --prune_kvlength=$PRUNE_KVLENGTH \
        --end=-1 
    done