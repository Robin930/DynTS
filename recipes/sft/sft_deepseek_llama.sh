export CUDA_VISIBLE_DEVICES="0,1,2,3"
export HF_ENDPOINT=https://hf-mirror.com  
export WANDB_MODE=offline

DATA_NAME=math
MODEL_NAME=DeepSeek-R1-Distill-Llama-8B

MODEL_NAME_OR_PATH="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
     
# 32768+500
MAX_LENGTH=18000
PARALLEL_SIZE=4
EPOCH=15
LR=5e-4

torchrun --nnodes=1 --master_port=29677 --nproc_per_node=${PARALLEL_SIZE} -m verl_sft_fsdp \
    data.train_files=datas/train/${MODEL_NAME}_${DATA_NAME}_is.jsonl \
    data.val_files=datas/train/${MODEL_NAME}_${DATA_NAME}_is.jsonl \
    data.max_length=${MAX_LENGTH} \
    model.model_name_or_path=${MODEL_NAME_OR_PATH} \
    trainer.experiment_name=${MODEL_NAME}_ip \
    trainer.alpha=1 \
    trainer.head_type="regression" \
    trainer.pruning_ratio=0 \
    trainer.total_epochs=${EPOCH} \
    optim.lr=${LR} \
    trainer.ktl=true