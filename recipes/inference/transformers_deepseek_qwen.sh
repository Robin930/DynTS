export CUDA_VISIBLE_DEVICES="0,1"
export HF_ENDPOINT=https://hf-mirror.com  

# DATA_NAME=(aime24 aime25 amc23 gaokao2023en gpqa_d math500)
DATA_NAME=(aime24)

SAVE_DIR="outputs/inference/transformers-qwen"

for DATA in "${DATA_NAME[@]}"; do
    python eval.py \
        --method="transformers" \
        --model_name_or_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
        --data_dir="datas/dataset/$DATA" \
        --save_dir=$SAVE_DIR \
        --batch_size=20 \
        --device_parallel_size=2 \
        --num_samples=5 \
        --max_new_tokens=16384\
        --end=-1
    done