export CUDA_VISIBLE_DEVICES="0,1"
export HF_ENDPOINT=https://hf-mirror.com  
export VLLM_LOGGING_LEVEL=ERROR

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

# METHODS=("is-top" "is-bottom" "is-random")
METHODS=("is-random")
RATIOS=(0.02)

DATA_DIR="datas/importance_score_v2"
SAVE_DIR="outputs/observation-v2"
SP=2
MAX_TOKEN_NUM=5000

for MODEL_NAME_OR_PATH in "${MODEL_NAME_OR_PATHS[@]}"; do
    for DATA_NAME in "${DATA_NAMES[@]}"; do
        for METHOD in "${METHODS[@]}"; do
            for RATIO in "${RATIOS[@]}"; do
                echo "Processing Model: $MODEL_NAME_OR_PATH on Dataset: $DATA_NAME with Method: $METHOD and Ratio: $RATIO"
                
                python inference_vllm_pruning.py \
                    --method="$METHOD" \
                    --ratio=$RATIO \
                    --model_name_or_path="$MODEL_NAME_OR_PATH" \
                    --data_name="$DATA_NAME" \
                    --data_dir="$DATA_DIR" \
                    --save_dir="$SAVE_DIR" \
                    --parallel_size=$SP \
                    --max_tokens=$MAX_TOKEN_NUM \
                    --num_samples=5 \
                    # --filter

            done
        done
    done
done
