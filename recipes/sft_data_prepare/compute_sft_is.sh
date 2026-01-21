export CUDA_VISIBLE_DEVICES="0,1,2,3"
export HF_ENDPOINT=https://hf-mirror.com  


MODEL_NAME_OR_PATHS=(   
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" 
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" 
)

DATA_NAMES=(  
    # "aime24"
    # "aime25"
    # "amc23"
    # "gpqa_d"
    # "gaokao2023en"
    "math"
)


DATA_DIR="outputs/inference/vllm_train"
SAVE_DIR="./datas/train"
SP=4
MAX_TOKEN_NUM=16384


for MODEL_NAME_OR_PATH in "${MODEL_NAME_OR_PATHS[@]}"; do
    for DATA_NAME in "${DATA_NAMES[@]}"; do
        echo "Processing Model: $MODEL_NAME_OR_PATH on Dataset: $DATA_NAME"
        
        torchrun --nproc_per_node=$SP --master_port=29511 compute_is_sp.py \
                    --sp_size=$SP \
                    --data_dir=$DATA_DIR \
                    --data_name=$DATA_NAME \
                    --model_name_or_path=$MODEL_NAME_OR_PATH \
                    --save_dir=$SAVE_DIR \
                    --max_token_num=$MAX_TOKEN_NUM \
                    --filter_token_num \
                    --train_data

    done
done