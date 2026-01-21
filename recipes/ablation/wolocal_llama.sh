# w.o. local tokens
export CUDA_VISIBLE_DEVICES="0,1"
export HF_ENDPOINT=https://hf-mirror.com  

DATA_NAME=(aime24 aime25 amc23 gaokao2023en gpqa_d math500)
MODEL_PATH="outputs/models/DeepSeek-R1-Distill-Llama-8B_ip/checkpoints/global_step_405"

SAVE_DIR="outputs/ablation/nolocal-llama"

RATIO=0.3
LOCAL_WINDOW=1
PRUNE_KVLENGTH=3000

for DATA in "${DATA_NAME[@]}"; do
    echo "评估数据集: $DATA, 剪枝比例: $RATIO, 保留窗口: $LOCAL_WINDOW, 剪枝KV长度: $PRUNE_KVLENGTH"
    python eval.py \
        --method="dynts" \
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
        --kept_window=$LOCAL_WINDOW \
        --prune_signal='kvlength' \
        --prune_kvlength=$PRUNE_KVLENGTH
done