import os
import gc
import json
import time
import copy
import torch
import argparse

from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import set_start_method
from transformers import pipeline

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import TopPLogitsWarper, TemperatureLogitsWarper, TopKLogitsWarper
from transformers.cache_utils import DynamicCache, StaticCache

from utils.utils import set_seed
from qw_evaluation.parser import extract_answer, parse_ground_truth, parse_question
from qw_evaluation.grader import math_equal

class Inference:
    def __init__(self, temperature=0.6, max_new_tokens=16384, top_p=0.95, top_k=20, **kwargs):

        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.top_p_warper = TopPLogitsWarper(top_p=self.top_p)
        self.top_k_warper = TopKLogitsWarper(top_k=self.top_k)
        self.temperature_warper = TemperatureLogitsWarper(temperature=self.temperature)
        self.max_new_tokens = max_new_tokens

        self.device = kwargs.pop("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.finsh_type = kwargs.pop("finsh_type", "step") # step: decode step, time: decode time, length: kvcache length
        self.max_time = kwargs.pop("max_time", None) # seconds
        self.max_kvlength = kwargs.pop("max_kvlength", None) # max kv length
        self.kwargs = kwargs


    def is_unfinished(self, num_unfinished_sequences, step, time, length):
        if self.finsh_type == "step":
            return step < self.max_new_tokens and num_unfinished_sequences != 0
        elif self.finsh_type == "time" and self.max_time is not None:
            return time < self.max_time and num_unfinished_sequences != 0
        elif self.finsh_type == "length" and self.max_kvlength is not None:
            return length < self.max_kvlength and num_unfinished_sequences != 0
        else:
            raise ValueError("finsh_type must be one of ['step', 'time', 'length']")

    def infer(self, model, tokenizer, batched_prompts=None):

        inputs = tokenizer(batched_prompts, return_tensors="pt", add_special_tokens=False, padding=True, padding_side="left")
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        bs = input_ids.shape[0]

        ''' prefill stage '''
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                logits_to_keep=1,
                use_cache=True,
                return_dict=True,
            )

        # init decoding stage: next_token_ids, position_ids, cache_position
        next_token_ids = self.logits_to_tokens(outputs.logits[:, -1, :])
        kvcache = outputs.past_key_values
        prefill_tokens = input_ids.shape[1]
        cache_position = torch.arange(prefill_tokens, prefill_tokens + next_token_ids.shape[1], device=next_token_ids.device)
        position_ids = position_ids[:,-1:] + 1 # [B,]

        # setting stop criteria
        unfinished_sequences = torch.ones_like(next_token_ids, dtype=torch.long)

        # update attention mask. if the sequences not stop mask = 1, else mask = 0
        attention_mask = torch.cat([attention_mask, unfinished_sequences], dim=1)

        # initialize deoce step, time, length
        decode_step = 0
        decode_time = 0
        kv_length = 0

        # store generated token ids
        decode_token_ids = torch.empty((bs, 0), dtype=torch.int64).to(self.device)

        # compute the number of unfinished sequences
        num_unfinished_sequences = unfinished_sequences.sum().item()

        # save info 
        info_time_per_decode_step = []
        info_kvmem_per_decode_step = []
        info_peak_kvcache = 0
        info_num_decode_tokens = torch.full((bs,), self.max_new_tokens, dtype=torch.int32)

        # import rpdb; rpdb.set_trace()

        ''' decode stage '''
        with torch.inference_mode():
            while self.is_unfinished(num_unfinished_sequences, decode_step, decode_time, kv_length):
                # decode the next token
                torch.cuda.synchronize()
                start_time_per_token = time.perf_counter()
                outputs = model(
                    input_ids=next_token_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    logits_to_keep=1,
                    use_cache=True,
                    past_key_values=kvcache, 
                    # cache_position=cache_position
                )
                torch.cuda.synchronize()
                time_decode_one_step = time.perf_counter() - start_time_per_token
                # import rpdb; rpdb.set_trace()
                # torch.distributed.barrier()
                # update next_token_ids, position_ids
                next_token_ids = self.logits_to_tokens(outputs.logits[:, -1, :])
                position_ids = position_ids + 1
                cache_position = cache_position + 1

                # update unfinished sequences and next_token_ids
                next_token_ids = next_token_ids * unfinished_sequences + tokenizer.eos_token_id * (1-unfinished_sequences)
                unfinished_sequences = unfinished_sequences & ~(next_token_ids == tokenizer.eos_token_id)

                # update attention mask
                attention_mask = torch.cat([attention_mask, unfinished_sequences], dim=1)
                
                # add decode token ids
                decode_token_ids = torch.cat([decode_token_ids, next_token_ids], dim=1)

                # update decode step, time, kv_length
                decode_step += 1
                decode_time += time_decode_one_step
                kv_length = kvcache.get_seq_length()

                # save info
                info_time_per_decode_step.append(time_decode_one_step)
                if kv_length > info_peak_kvcache:
                    info_peak_kvcache = kv_length

                memory_bytes = (kvcache.layers[0].keys.numel() * kvcache.layers[0].keys.element_size() + \
                            kvcache.layers[0].values.numel() * kvcache.layers[0].values.element_size()) * len(kvcache.layers)
                info_kvmem_per_decode_step.append(memory_bytes / 1024 / 1024) # MB

                # record finished sequences
                if unfinished_sequences.sum().item() < num_unfinished_sequences:
                    num_unfinished_sequences = unfinished_sequences.sum().item()
                    finished_seq_idx = torch.nonzero(unfinished_sequences==0, as_tuple=True)[0]
                    for fsi in finished_seq_idx.tolist():
                        info_num_decode_tokens[fsi] = decode_step
                
                # debug
                if self.kwargs.get("debug", False):
                    print(f"decode step: {decode_step}, decode time: {time_decode_one_step:.2f}s, kv_length: {kv_length}, unfinished sequences: {num_unfinished_sequences}")
                
                # del outputs
                # gc.collect()
                # torch.cuda.empty_cache()
                
        return decode_token_ids.tolist(), {
            "time_per_decode_step": info_time_per_decode_step,
            "kvmem_per_decode_step": info_kvmem_per_decode_step,
            "peak_kvcache": info_peak_kvcache,
            "num_decode_tokens": info_num_decode_tokens.tolist()
        }

    def logits_to_tokens(self, logits):
        # probs = logits / self.temperature
        probs = self.temperature_warper(input_ids=None, scores=logits)
        probs = self.top_p_warper(input_ids=None, scores=probs)
        probs = self.top_k_warper(input_ids=None, scores=probs)
        probs = torch.softmax(probs, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)
        return next_token_id