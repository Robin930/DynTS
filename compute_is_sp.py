import os
import json
import time
import torch
import argparse

from torch import nn
from tqdm import tqdm
from datasets import load_dataset,Dataset
from typing import List, Dict, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from flash_attn.bert_padding import index_first_axis, rearrange, unpad_input
from torch.distributed import init_device_mesh
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.protocol import DataProto
from verl.utils.distributed import initialize_global_process_group
from verl.utils.model import compute_position_id_with_mask, create_random_mask
from verl.utils.ulysses import (
    get_ulysses_sequence_parallel_world_size,
    get_ulysses_sequence_parallel_group,
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_world_size,
)
from transformers.models.qwen2.modeling_qwen2 import eager_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

from utils.utils import load_jsonl, seq_token_info, get_pad_token_id, get_think_token_ids,  get_think_index   

from transformers.utils.generic import TransformersKwargs
from transformers.models.exaone4.modular_exaone4 import Exaone4ForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from models.exaone import ExaoneForCausalLM

# model = LlamaForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
# class myTransformersKwargs(TransformersKwargs):
#     think_index_end: Optional[int]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="datas/train", type=str)
    parser.add_argument("--data_name", default="deepmath", type=str)
    parser.add_argument("--model_name_or_path", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", type=str)

    parser.add_argument("--sp_size", type=int, default=4)
    parser.add_argument("--save_dir", default="datas/train", type=str)

    parser.add_argument("--filter_token_num", action="store_true")
    parser.add_argument("--max_token_num", type=int, default=16384)

    parser.add_argument("--train_data", action="store_true")
    args = parser.parse_args()
    return args


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    cache_position: torch.Tensor,
    batch_size: int,
):

    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        min_dtype = torch.finfo(dtype).min
        causal_mask = torch.full(
            (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
        )
        diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)

        causal_mask *= diagonal_attend_mask
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            if attention_mask.shape[-1] > target_length:
                attention_mask = attention_mask[:, :target_length]
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                causal_mask.device
            )
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )
    return causal_mask

def update_causal_mask(
    attention_mask: torch.Tensor,
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor,
):

    # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
    # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
    # to infer the attention mask.
    past_seen_tokens = 0

    dtype, device = torch.bfloat16, torch.cuda.current_device()
    sequence_length = input_tensor.shape[1]

    # DynamicCache or no cache
    target_length = (
        attention_mask.shape[-1]
        if isinstance(attention_mask, torch.Tensor)
        else past_seen_tokens + sequence_length + 1
    )

    # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
    causal_mask = prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask,
        sequence_length=sequence_length,
        target_length=target_length,
        dtype=dtype,
        device=device,
        cache_position=cache_position,
        batch_size=input_tensor.shape[0]
    )

    return causal_mask


def ulysses_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """Insert all-to-all before and after flash attention.
    DeepSpeed-Ulysses: https://arxiv.org/pdf/2309.14509

    Args:
        query_states (torch.Tensor): (batch_size, nheads, seqlen/sp_size, head_dim)
        key_states (torch.Tensor): (batch_size, nheads, seqlen/sp_size, head_dim)
        value_states (torch.Tensor): (batch_size, nheads, seqlen/sp_size, head_dim)
    Returns:
        torch.Tensor: (batch_size, seqlen/sp_size, nheads, head_dim)
    """
    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()
    # import pdb; pdb.set_trace()
    if ulysses_sp_size > 1:
        
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)

        query_states = gather_seq_scatter_heads(query, seq_dim=2, head_dim=1)
        key_states = gather_seq_scatter_heads(key_states, seq_dim=2, head_dim=1)
        value_states = gather_seq_scatter_heads(value_states, seq_dim=2, head_dim=1)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling

        # my_attn_weights = torch.mean(attn_weights, dim=(0,1))
        # my_mask = torch.triu(torch.ones_like(my_attn_weights), diagonal=1).to(torch.bool)
        # my_attn_weights = my_attn_weights.masked_fill(my_mask, torch.tensor(0))

        # truncate_idx = kwargs['think_index_end'] + 1
        # my_attn_weights = torch.mean(my_attn_weights[truncate_idx:,:], dim=(0))

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            # import rpdb;rpdb.set_trace()
            attn_weights = attn_weights + causal_mask
        
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = gather_heads_scatter_seq(attn_output, seq_dim=1, head_dim=2)


        # import rpdb; rpdb.set_trace()
        # torch.distributed.barrier()

        # # import rpdb; rpdb.set_trace()
        attn_weights.diagonal(dim1=-2,dim2=-1).zero_()
        truncate_idx = kwargs['think_index_end'] + 1
        attn_weights = torch.mean(attn_weights[:,:,truncate_idx:,:], dim=(0,1,2))


        return attn_output, attn_weights



def load_model_and_tokenizer(model_name_or_path, unquant):

    if unquant == True:
        quant_config = None
    else:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,              # 在这里指定即可
            bnb_4bit_compute_dtype="float16",  # 通常使用 float16/bfloat16
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"         # 也可以是 'fp4'
        )
    if model_name_or_path == "LGAI-EXAONE/EXAONE-Deep-7.8B":
        model = ExaoneForCausalLM.from_pretrained(model_name_or_path,
                                                device_map="cuda",
                                                # torch_dtype=torch.bfloat16,
                                                # quantization_config=quant_config,
                                                # output_attentions=True,
                                                return_dict_in_generate=True,
                                                # attn_implementation="eager",
                                                attn_implementation="ulysses_eager",
                                                # attn_implementation="eager",
                                                use_cache=False,
                                                trust_remote_code=True,
                                                )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                            device_map="cuda",
                                            # torch_dtype=torch.bfloat16,
                                            # quantization_config=quant_config,
                                            # output_attentions=True,
                                            return_dict_in_generate=True,
                                            # attn_implementation="eager",
                                            attn_implementation="ulysses_eager",
                                            use_cache=False,
                                            )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    if model_name_or_path in ["Nanbeige/Nanbeige4-3B-Thinking-2511"]:
        tokenizer.add_special_tokens({
            "additional_special_tokens": ["<think>", "</think>"]
        })
    model.eval()
    return model, tokenizer

def sync_model_parameters_global(layer):
    # synchronize weights
    for p in layer.parameters():
        torch.distributed.broadcast(tensor=p.data, src=0)

def chunk_add_pad(input_ids, attention_mask, chunk_size, pad_token_id):
    total_len = input_ids.shape[1]
    pad_len = (chunk_size - total_len % chunk_size) % chunk_size
    padded_input = torch.cat([
        input_ids,
        torch.full((1,pad_len), pad_token_id, dtype=input_ids.dtype)
    ],dim=1)
    padded_attention_mask = torch.cat([
        attention_mask,
        torch.full((1,pad_len), 0, dtype=attention_mask.dtype)
    ],dim=1)
    return padded_input, padded_attention_mask, total_len

def all_gather_tensor(local_tensor, dim, group=None, stack=False):
    group = get_ulysses_sequence_parallel_group() if group is None else group
    sp_world_size = torch.distributed.get_world_size(group=group)
    gather_tensor_list = [torch.empty_like(local_tensor) for _ in range(sp_world_size)]
    torch.distributed.all_gather(gather_tensor_list,local_tensor,group)
    if stack == True:
        gather_tensor = torch.stack(gather_tensor_list,dim=dim)
    else:
        gather_tensor = torch.cat(gather_tensor_list,dim=dim)
    return gather_tensor

def compute_importance_score(args, data, model, tokenizer, sharding_manager, ulysses_device_mesh):

    rank = torch.distributed.get_rank()
    
    # Loop through the samples to compute the important score.
    for data_i, data in enumerate(tqdm(data, desc="Processing data for important score")):
        text = data["texts"]

        inputs = tokenizer(text, return_tensors='pt', add_special_tokens=False)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        if args.model_name_or_path in ["LGAI-EXAONE/EXAONE-Deep-7.8B"]:
            think_start_index, think_end_index = get_think_index(text, tokenizer, args.model_name_or_path)
        else:
            _, _, think_start_index, think_end_index = seq_token_info(text, tokenizer)
        
        # import rpdb; rpdb.set_trace()
        # print(think_start_index, think_end_index)

        chunk_size = ulysses_device_mesh['sp'].size()
        if tokenizer.pad_token is None:
            print("Adding pad token to tokenizer...")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        pad_token_id = tokenizer.pad_token_id
        # pad_token_id = 151645
        padded_input, padded_attention_mask, nopad_len = chunk_add_pad(input_ids, attention_mask, chunk_size, pad_token_id)
        position_ids = compute_position_id_with_mask(padded_attention_mask)
        
        local_padded_input = padded_input.chunk(chunks=chunk_size, dim=1)[rank].cuda()
        local_position_ids = position_ids.chunk(chunks=chunk_size, dim=1)[rank].cuda()
        local_padded_attention_mask = padded_attention_mask.cuda()

        cache_position = all_gather_tensor(local_position_ids,dim=1).squeeze(0)
        causal_mask = update_causal_mask(
            attention_mask=local_padded_attention_mask,
            input_tensor=padded_input,
            cache_position=cache_position,
        )
        with torch.inference_mode():
        # with torch.no_grad():
            # kwargs = myTransformersKwargs(think_index_end=think_end_index)
            with sharding_manager:
                outputs = model(
                    input_ids=local_padded_input,
                    attention_mask=causal_mask,
                    position_ids=local_position_ids,
                    use_cache=False,
                    output_attentions=True,
                    think_index_end=think_end_index,
                )

                # import rpdb; rpdb.set_trace()
                importance_score = torch.stack(outputs.attentions,dim=0)
                importance_score = torch.mean(importance_score, dim=0) 
                importance_score = all_gather_tensor(importance_score, dim=0, stack=True)  
                importance_score = torch.mean(importance_score, dim=0)[:nopad_len]
                # importance_score = torch.mean(importance_score, dim=0)[:,:nopad_len] # for two attn weight
                # import rpdb; rpdb.set_trace()
                # import rpdb; rpdb.set_trace()
                # torch.distributed.barrier()
                data['importance_score'] = importance_score.tolist()  # Convert to list for JSON serialization

                torch.cuda.empty_cache()

        if rank == 0:
            print('start think index:', think_start_index, ' end think index:', think_end_index)

            if args.train_data:
                save_path = args.save_dir + "/" + args.model_name_or_path.split("/")[-1] + "_" + args.data_name + "_is-new.jsonl"
                save_dir = os.path.dirname(save_path)
                os.makedirs(save_dir, exist_ok=True)
                with open(save_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
            else:
                save_path = args.save_dir + "/" + args.model_name_or_path.split("/")[-1] + "_" + args.data_name + "_is-FCTrue.jsonl"
                save_path2 = args.save_dir + "/" + args.model_name_or_path.split("/")[-1] + "_" + args.data_name + "_is-FCFalse.jsonl"
                save_dir = os.path.dirname(save_path)
                os.makedirs(save_dir, exist_ok=True)

                if args.filter_token_num == True and data['score']==True:
                    with open(save_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(data, ensure_ascii=False) + "\n")

                with open(save_path2, "a", encoding="utf-8") as f:
                        f.write(json.dumps(data, ensure_ascii=False) + "\n")
            

# ulysses sequence parallel attention forward
ALL_ATTENTION_FUNCTIONS['ulysses_eager'] = ulysses_eager_attention_forward
# ALL_MASK_ATTENTION_FUNCTIONS['ulysses_eager'] = ALL_MASK_ATTENTION_FUNCTIONS['eager']
ALL_MASK_ATTENTION_FUNCTIONS._global_mapping['ulysses_eager'] = ALL_MASK_ATTENTION_FUNCTIONS['eager']

if __name__ == "__main__":
    args = parse_args()
    print(args)

    # load datasets
    file_path = args.data_dir + "/" + args.model_name_or_path.split("/")[-1] + "_" + args.data_name + '.jsonl'
    reasoning_data = load_jsonl(file_path)

    filtered_data = []
    added_qid = []

    if args.model_name_or_path == "LGAI-EXAONE/EXAONE-Deep-7.8B":
        think_strat = "<thought>"
        think_end = "</thought>"
    else:
        think_strat = "<think>"
        think_end = "</think>"

    for data in reasoning_data:
        if args.model_name_or_path == "LGAI-EXAONE/EXAONE-Deep-7.8B":
            text = data["texts"].lower().replace(" ", "")
        if len(data['texts'].split(think_strat)) == 2 and len(data['texts'].split(think_end)) == 2:
            if args.train_data:
                if data['question_id'] not in added_qid:
                    if data['score']==True:
                        added_qid.append(data['question_id'])
                        filtered_data.append(data)
            else:
                if data['question_id'] not in added_qid:
                    added_qid.append(data['question_id'])
                    filtered_data.append(data)

    print(f"Total samples: {len(reasoning_data)}, Filtered samples: {len(filtered_data)}")
    # import pdb; pdb.set_trace()

    # initialize device mesh
    dp_size=1
    sp_size=args.sp_size
    ulysses_device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(dp_size, sp_size), mesh_dim_names=("dp", "sp"))
    sharding_manager = FSDPUlyssesShardingManager(ulysses_device_mesh)
    
    # load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path, False)
    print("Model and tokenizer loaded.")
    # print(model.config)
    sync_model_parameters_global(model)

    # compute important score
    print("Computing attention maps for training data...")
    train_data = compute_importance_score(args, filtered_data, model, tokenizer, sharding_manager, ulysses_device_mesh)

