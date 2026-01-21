import time
import torch
from inference.base import Inference

class DynTS(Inference):
    def __init__(self, temperature=0.6, max_new_tokens=16384, top_p=0.95, top_k=20, **kwargs):
        super().__init__(temperature, max_new_tokens, top_p, top_k, **kwargs)
        # DynTS specific args
        self.prune_signal = kwargs.pop("prune_signal", None)
        self.prune_step = kwargs.pop("prune_step", None)
        self.prune_kvlength = kwargs.pop("prune_kvlength", None)
        self.kept_window = kwargs.pop("kept_window", None)
        self.think_end_token_id = kwargs.pop("think_end_token_id", None)
        self.method = kwargs.pop("method", None)
        
        # re
        self.head_type = kwargs.pop("head_type", None)
        self.ratio = kwargs.pop("ratio", None)
        self.pruning_size = kwargs.pop("pruning_size", None)

        # check args
        self.model_name = kwargs.pop("model_name", None)
        self.device = kwargs.get("device", None)

        print("Inference Method:", self.method)

    def cut_tensor_by_mask(self, tensor, mask, am=False):
        """
            tensor: [batch_size, head, seq_len, dim]
            mask: [batch_size, seq_len], 1 for keep, 0 for remove
            return: [batch_size, head, seq_len, dim]
        """
        if am == True:
            B, T = tensor.shape
        else:
            B,C,T,D = tensor.shape

        keep_lens = mask.sum(dim=1)
        T_max = int(keep_lens.max().item())
        cut_len = T - T_max
        
        idx = torch.arange(T).expand(B,T).to(tensor.device)  # [B, T]
        key = (mask).to(torch.int64) * T + idx
        perm = key.argsort(dim=1)

        new_idx = torch.gather(idx, dim=1, index=perm) # 把所有0排到前面
        cut_idx = new_idx[:, cut_len:]  # 去掉前面cut_len个0
        cut_idx_sorted, _ = cut_idx.sort(dim=1) # 恢复原始顺序

        new_mask = torch.gather(mask, dim=1, index=cut_idx_sorted)

        if am == True:
            new_tensor = torch.gather(tensor, dim=1, index=cut_idx_sorted)
        else:
            new_tensor = torch.gather(tensor, dim=2, index=cut_idx_sorted[:, None,:, None].expand(B,C,T_max,D))

        return new_tensor, new_mask

    def is_pruning_step(self, step, kvlength):
        if self.prune_signal == "step" and self.prune_step is not None:
            return step % self.prune_step == 0 and step > 0
        elif self.prune_signal == "kvlength" and self.prune_kvlength is not None:
            return kvlength >= self.prune_kvlength
        else:
            return False

    def pruning(self, kvcache, attention_mask, pruning_mask):
        """
            kvcache: past_key_values
            attention_mask: [B, seq_len]
            pruning_mask: [B, seq_len], 1 for keep, 0 for remove
            return: pruned attention_mask, pruned pruning_mask
        """
        process_pruning_mask = pruning_mask[:, :-self.kept_window]
        keep_pruning_mask = pruning_mask[:, -self.kept_window:]

        # prune kvcache
        for layer in kvcache.layers:
            layer.keys = torch.cat([self.cut_tensor_by_mask(layer.keys[:,:,:-self.kept_window,:], process_pruning_mask)[0], \
                                    layer.keys[:,:,-self.kept_window:,:]], dim=2).contiguous()
            layer.values = torch.cat([self.cut_tensor_by_mask(layer.values[:,:,:-self.kept_window,:], process_pruning_mask)[0], \
                                    layer.values[:,:,-self.kept_window:,:]], dim=2).contiguous()
        
        # prune attention_mask
        process_attention_mask = attention_mask[:, :-self.kept_window]
        keep_attention_mask = attention_mask[:, -self.kept_window:]
        
        attention_mask_p, process_pruning_mask_p = self.cut_tensor_by_mask(process_attention_mask, process_pruning_mask, am=True)
        attention_mask = torch.cat([attention_mask_p, keep_attention_mask], dim=1).contiguous()

        # prune pruning_mask
        pruning_mask = torch.cat([process_pruning_mask_p, keep_pruning_mask], dim=1).contiguous()
        torch.cuda.empty_cache()

        return attention_mask, pruning_mask

    def pruning2(self, kvcache, attention_mask, importance_score):
        kv_len_before_pruning = kvcache.get_seq_length()
        if kvcache.get_seq_length() > self.kept_window:
            p_is = importance_score[:, :-self.kept_window]
            k_is = importance_score[:, -self.kept_window:]
            if self.pruning_size is None:
                k = int(p_is.shape[-1] * self.ratio)
            else:
                k = self.prune_kvlength - self.kept_window - self.pruning_size
            if self.method == "dynts-random":
                B, L = p_is.shape
                topk_idx = torch.stack([torch.randperm(L, device=p_is.device)[:k] for _ in range(B)], dim=0)
            elif self.method == "dynts-bottom":
                topk_val, topk_idx = torch.topk(p_is, k, dim=-1, largest=False)  # (B, k)
            elif self.method == "dynts" or self.method == "dynts-noque":
                topk_val, topk_idx = torch.topk(p_is, k, dim=-1)  # (B, k)
            else:
                print(f"Method {self.method} not supported for DynTS pruning2.")
            topk_idx_sort = torch.sort(topk_idx,dim=-1).values
            topk_idx_sort_exp = topk_idx_sort[:, None, :, None].expand(-1, kvcache.layers[0].keys.shape[1], -1, kvcache.layers[0].keys.shape[-1])

            for layer in kvcache.layers:
                layer.keys = torch.cat([torch.gather(layer.keys[:, :, :-self.kept_window, :], dim=2, index=topk_idx_sort_exp), layer.keys[:, :, -self.kept_window:, :]], dim=2).contiguous()
                layer.values = torch.cat([torch.gather(layer.values[:, :, :-self.kept_window, :], dim=2, index=topk_idx_sort_exp), layer.values[:, :, -self.kept_window:, :]], dim=2).contiguous()
            attention_mask = torch.cat([torch.gather(attention_mask[:, :-self.kept_window], dim=1, index=topk_idx_sort), attention_mask[:, -self.kept_window:]], dim=1).contiguous()
            importance_score = torch.cat([torch.gather(importance_score[:, :-self.kept_window], dim=1, index=topk_idx_sort), importance_score[:, -self.kept_window:]], dim=1).contiguous()
        print(f"[{self.device}] Pruning KV Cache from {kv_len_before_pruning} to {kvcache.get_seq_length()}", flush=True)
        return attention_mask, importance_score

    def match_think_end_token(self, decode_token_ids, unfinished_think):

        if decode_token_ids.shape[1] < 3:
            return unfinished_think # not enough tokens to match
        else:
            if self.model_name == 'EXAONE-Deep-7.8B':
                pattern = torch.tensor([2240, 52040, 391], device=unfinished_think.device)
                # pattern = torch.tensor([389, 52040, 391], device=unfinished_think.device)
            elif self.model_name == 'Nanbeige4-3B-Thinking-2511':
                pattern = torch.tensor([897, 20993, 152426], device=unfinished_think.device)
            else: 
                raise NotImplementedError(f"Model {self.model_name} not supported for think end token matching.")
            
            decode_token_ids_last3 = decode_token_ids[:,-3:]
            # import rpdb; rpdb.set_trace()
            match_last3 = (decode_token_ids_last3 == pattern).all(dim=1).unsqueeze(-1)  # [B, 1]
            unfinished_think = unfinished_think & ~match_last3  # [B,]

            return unfinished_think

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

        # create pruning mask 
        pruning_mask = attention_mask[:, :-1].clone().bool() # [B, seq_len], 1 for keep, 0 for remove
        # setting think end criteria
        unfinished_think = torch.ones_like(next_token_ids, dtype=torch.bool)


        # initialize deoce step, time, length
        decode_step = 0
        decode_time = 0
        kv_length = 0

        # store generated token ids
        decode_token_ids = torch.empty((bs, 0), dtype=torch.int64).to(self.device)
        # decode_token_ids = input_ids[:, :].clone()

        # compute the number of unfinished sequences
        num_unfinished_sequences = unfinished_sequences.sum().item()

        # save info 
        info_time_per_decode_step = []
        info_kvmem_per_decode_step = []
        info_peak_kvcache = 0
        info_num_decode_tokens = torch.full((bs,), self.max_new_tokens, dtype=torch.int32)


        # print(self.head_type)
        if self.head_type == "regression":
            print("Regression head type pruning")
            importance_score = torch.where(pruning_mask==True, torch.tensor(torch.inf).to(self.device), torch.tensor(0.).to(self.device))
            if self.method == "dynts-noque":
                print("DynTS No Question Mode: Setting all initial importance scores to 0")
                importance_score = torch.full_like(importance_score, torch.tensor(0.).to(self.device))

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
                # update pruning mask and unfinished_think
                if self.head_type == "regression":
                    if self.model_name in ['EXAONE-Deep-7.8B']:
                        
                        # if decode_step == 4604:
                        #     import rpdb; rpdb.set_trace()
                            
                        unfinished_think = self.match_think_end_token(decode_token_ids, unfinished_think)
                    else:
                        unfinished_think = unfinished_think & ~(next_token_ids == self.think_end_token_id)

                    if decode_step == 0 and self.model_name in ['Nanbeige4-3B-Thinking-2511']: #TODO: 添加think start token判断
                        new_importance_scores = torch.tensor(torch.inf).to(self.device)
                        
                        # import rpdb; rpdb.set_trace()
                        # torch.distributed.barrier()
                    else:
                        new_importance_scores = outputs.importance_scores.squeeze(-1)  # [B, 1]


                    # import rpdb; rpdb.set_trace()
                    # torch.distributed.barrier() 
                    importance_score_next = torch.where(unfinished_think==True, new_importance_scores, torch.tensor(torch.inf).to(self.device))
                    # print(importance_score_next.shape, importance_score.shape, outputs.importance_scores.shape, unfinished_think.shape)
                    importance_score = torch.cat([importance_score, importance_score_next], dim=1)

                    # Pruning Start
                    if self.is_pruning_step(decode_step, kv_length):
                        if self.kwargs.get("debug", False):
                            print('Pruning step:')
                        attention_mask, importance_score = self.pruning2(kvcache, attention_mask, importance_score)

                elif self.head_type == "classification":
                    pred_mask = torch.argmax(torch.softmax(outputs.importance_scores, dim=2),dim=2).bool()  # [1,]
                    unfinished_think = unfinished_think & ~(next_token_ids == self.think_end_token_id)  # [B,]
                    token_pruning_mask = torch.where(unfinished_think==True, pred_mask, unfinished_think)
                    pruning_mask = torch.cat([pruning_mask, token_pruning_mask],dim=1)

                    # Pruning Start
                    if self.is_pruning_step(decode_step, kv_length):
                        if self.kwargs.get("debug", False):
                            print('Pruning step:')
                        attention_mask, pruning_mask = self.pruning(kvcache, attention_mask, pruning_mask)
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
                    print(f"unfinished_think: {unfinished_think.sum().item()}")

                    # if num_unfinished_sequences == 19:
                    #     import rpdb; rpdb.set_trace()
                    #     torch.distributed.barrier()

        return decode_token_ids.tolist(), {
            "time_per_decode_step": info_time_per_decode_step,
            "kvmem_per_decode_step": info_kvmem_per_decode_step,
            "peak_kvcache": info_peak_kvcache,
            "num_decode_tokens": info_num_decode_tokens.tolist()
        }