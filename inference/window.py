import time
import torch
from inference.dynts import DynTS

class Window(DynTS):
    def __init__(self, temperature=0.6, max_new_tokens=16384, top_p=0.95, top_k=20, **kwargs):
        super().__init__(temperature, max_new_tokens, top_p, top_k, **kwargs)
        # Window specific args
        self.sink_type = kwargs.pop("sink_type", "none") # none, native, question
        self.sink_size = kwargs.pop("sink_size", 10) # number of tokens to keep in the sink

        self.flag_first_pruning = True

    def cut_sink(self, tensor, attention_mask, am=False):
        if am == True:
            B, L = tensor.shape
        else:
            B, C, L, D = tensor.shape
        
        K = self.sink_size

        valid_len = attention_mask.sum(dim=1)
        start_index = L - valid_len

        rangeK = torch.arange(K, device=attention_mask.device)  # (K,)
        index = start_index[:, None] + rangeK[None, :]  # (B, K)

        if am == True:
            tensor = torch.gather(tensor, dim=1, index=index)
            return tensor
        else:
            index_exp = index[:, None, :, None].expand(-1, C, -1, D)
            tensor = torch.gather(tensor, dim=2, index=index_exp)
            return tensor
        
        
    def process_first_pruning(self, kvcache, attention_mask, num_prefill_tokens):
        # attention_mask
        p_am = attention_mask[:, :num_prefill_tokens]
        k_am = attention_mask[:, num_prefill_tokens:]
        
        for layer in kvcache.layers:
            layer.keys = torch.cat([self.cut_sink(layer.keys[:,:,:num_prefill_tokens,:], p_am, am=False), layer.keys[:,:,num_prefill_tokens:,:]], dim=2).contiguous()
            layer.values = torch.cat([self.cut_sink(layer.values[:,:,:num_prefill_tokens,:], p_am, am=False), layer.values[:,:,num_prefill_tokens:,:]], dim=2).contiguous()
        
        attention_mask = torch.cat([self.cut_sink(attention_mask[:,:num_prefill_tokens], p_am, am=True), k_am], dim=1).contiguous()

        return attention_mask


    def pruning(self, kvcache, attention_mask, num_prefill_tokens):
        
        if self.sink_type == "none":
            if kvcache.get_seq_length() > self.kept_window:
                for layer in kvcache.layers:
                    layer.keys = layer.keys[:, :, -self.kept_window:, :].contiguous()
                    layer.values = layer.values[:, :, -self.kept_window:, :].contiguous()
                attention_mask = attention_mask[:, -self.kept_window:].contiguous()
        elif self.sink_type == "native":
            if kvcache.get_seq_length() > self.kept_window + self.sink_size:
                if self.flag_first_pruning: # first pruning need to handle the prefll tokens
                    print("First pruning with sink")
                    attention_mask = self.process_first_pruning(kvcache, attention_mask, num_prefill_tokens)
                    self.flag_first_pruning = False
                    print(f"First pruning done, attention_mask shape: {attention_mask.shape}, kvcache length: {kvcache.get_seq_length()}")
                if kvcache.get_seq_length() > self.kept_window + self.sink_size:
                    for layer in kvcache.layers:
                        layer.keys = torch.cat([layer.keys[:, :, :self.sink_size, :], layer.keys[:, :, -self.kept_window:, :]], dim=2).contiguous()
                        layer.values = torch.cat([layer.values[:, :, :self.sink_size, :], layer.values[:, :, -self.kept_window:, :]], dim=2).contiguous()
                    attention_mask = torch.cat([attention_mask[:, :self.sink_size], attention_mask[:, -self.kept_window:]], dim=1).contiguous()
        elif self.sink_type == "question":
            if kvcache.get_seq_length() > self.kept_window + num_prefill_tokens:
                for layer in kvcache.layers:
                    layer.keys = torch.cat([layer.keys[:, :, :num_prefill_tokens, :], layer.keys[:, :, -self.kept_window:, :]], dim=2).contiguous()
                    layer.values = torch.cat([layer.values[:, :, :num_prefill_tokens, :], layer.values[:, :, -self.kept_window:, :]], dim=2).contiguous()
                attention_mask = torch.cat([attention_mask[:, :num_prefill_tokens], attention_mask[:, -self.kept_window:]], dim=1).contiguous()
        return attention_mask

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

                # Pruning Start
                if self.is_pruning_step(decode_step, kv_length):
                    attention_mask = self.pruning(kvcache, attention_mask, prefill_tokens)
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