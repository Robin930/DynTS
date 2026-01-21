import time
import torch

from utils.attention import h2o_eager_attention_forward
from inference.dynts import DynTS
import torch.nn.functional as F

class H2O(DynTS):
    def __init__(self, temperature=0.6, max_new_tokens=16384, top_p=0.95, top_k=20, **kwargs):
        super().__init__(temperature, max_new_tokens, top_p, top_k, **kwargs)
        # H2O specific args
        self.ratio = kwargs.pop("ratio", None) # number of attention heads / number of key/value heads
        
        if self.ratio is None:
            raise ValueError("Please specify the ratio of attention heads to key/value heads for H2O models.")
    
    def pruning(self, kvcache, attention_mask, importance_score):
        if kvcache.get_seq_length() > self.kept_window:
            p_is = importance_score[:, :-self.kept_window]
            k_is = importance_score[:, -self.kept_window:]
            k = int(p_is.shape[-1] * self.ratio)
            print(k)
            topk_val, topk_idx = torch.topk(p_is, k, dim=-1)  # (B, k)
            topk_idx_sort = torch.sort(topk_idx, dim=-1).values
            topk_idx_sort_exp = topk_idx_sort[:, None, :, None].expand(-1, kvcache.layers[0].keys.shape[1], -1, kvcache.layers[0].keys.shape[-1])

            for layer in kvcache.layers:
                layer.keys = torch.cat([torch.gather(layer.keys[:, :, :-self.kept_window, :], dim=2, index=topk_idx_sort_exp), layer.keys[:, :, -self.kept_window:, :]], dim=2).contiguous()
                layer.values = torch.cat([torch.gather(layer.values[:, :, :-self.kept_window, :], dim=2, index=topk_idx_sort_exp), layer.values[:, :, -self.kept_window:, :]], dim=2).contiguous()
            attention_mask = torch.cat([torch.gather(attention_mask[:, :-self.kept_window], dim=1, index=topk_idx_sort), attention_mask[:, -self.kept_window:]], dim=1).contiguous()
            importance_score = torch.cat([torch.gather(importance_score[:, :-self.kept_window], dim=1, index=topk_idx_sort), importance_score[:, -self.kept_window:]], dim=1).contiguous()
        return attention_mask, importance_score

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
                output_attentions=True,
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

        # importance score
        importance_score = torch.mean((torch.stack(outputs.attentions,dim=0)), dim=0)

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
                    output_attentions=True,
                    # cache_position=cache_position
                )
                torch.cuda.synchronize()
                time_decode_one_step = time.perf_counter() - start_time_per_token
                # import rpdb; rpdb.set_trace()
                # torch.distributed.barrier()
                # Pruning Start
                cur_importance_score = torch.mean((torch.stack(outputs.attentions,dim=0)), dim=0)
                importance_score = F.pad(importance_score, (0,1), mode="constant", value=0) + cur_importance_score

                importance_score = torch.stack(outputs.attentions,dim=0)
                importance_score = torch.mean(importance_score, dim=0)
                if self.is_pruning_step(decode_step, kv_length):
                    attention_mask, importance_score = self.pruning(kvcache, attention_mask, importance_score)
                    # import rpdb; rpdb.set_trace()
                    # torch.distributed.barrier()
                # Pruning End

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

        return decode_token_ids.tolist(), {
            "time_per_decode_step": info_time_per_decode_step,
            "kvmem_per_decode_step": info_kvmem_per_decode_step,
            "peak_kvcache": info_peak_kvcache,
            "num_decode_tokens": info_num_decode_tokens.tolist()
        }