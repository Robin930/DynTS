import os
import gc
import torch
import json
from tqdm import tqdm
import random
import argparse

from vllm import SamplingParams,LLM
from transformers import AutoTokenizer


from utils.utils import seq_token_info, get_think_token_ids, get_passk, get_question_score
from qw_evaluation.parser import extract_answer
from qw_evaluation.grader import math_equal

from models.qwen import myQwen2ForCausalLM
from models.llama import myLlamaForCausalLM

def load_jsonl(path):
    datas = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            datas.append(item)
    return datas

def find_think_end_index(tokens):
    think_end_index = -1
    for i, token in enumerate(tokens):
        if token == '</think>':
            think_end_index = i
            break
    return think_end_index

def extract_text_from_results(results, data_name, datas):
    outputs_texts = []
    preds = []
    scores = []
    for i, result in enumerate(results):
        for res in result.outputs:
            generate_text = res.text
            data = datas[i]
            outputs_texts.append(generate_text)
            pred_ans = extract_answer(generate_text.split("</think>")[-1], data_name=data_name)
            preds.append(pred_ans)
            scores.append(math_equal(pred_ans, data['ground_truth']))  # Replace "gt" with actual ground truth if available
    return outputs_texts, preds, scores


def dynts_prune(datas, args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint_path)
    model = myQwen2ForCausalLM.from_pretrained(args.model_checkpoint_path,
                                               torch_dtype=torch.float16,
                                               head_type="classification",
                                               attn_implementation="flash_attention_2",
                                               device_map="auto")    
    model.eval()
    # pruning datas
    pruned_prompt_token_ids = []
    with torch.inference_mode():
        for data in tqdm(datas, desc="Pruning datas"):
            texts = data['texts']
            inputs = tokenizer(texts, return_tensors="pt", add_special_tokens=False).to(model.device)
            outputs = model(**inputs)
            # importance_scores = outputs.importance_scores.squeeze(0).squeeze(-1)
            is_mask = torch.argmax(torch.softmax(outputs.importance_scores, dim=-1), dim=-1).squeeze(0)
            inputs_ids = inputs['input_ids'][0]

            # sub_step, type_sub_step, think_index_start, think_index_end = seq_token_info(texts, tokenizer)
            think_start_token_id, think_end_token_id = get_think_token_ids(tokenizer)
            think_start_index = (inputs_ids == think_start_token_id).nonzero(as_tuple=True)[0][0].item()
            think_end_index = (inputs_ids == think_end_token_id).nonzero(as_tuple=True)[0][0].item()
            assert think_start_index < think_end_index, "think token indices error"
            if args.debug:
                print(f"think_start_index: {think_start_index}, think_end_index: {think_end_index}")

            question_tokens_ids = inputs_ids[:think_start_index+1]
            think_tokens_ids = inputs_ids[think_start_index+1:think_end_index]
            thinkend_tokens_id = inputs_ids[think_end_index].unsqueeze(0)

            think_is_mask = is_mask[think_start_index+1:think_end_index]

            pruned_think_token_ids = think_tokens_ids[think_is_mask==1]

            pruned_tokens_ids = torch.cat([question_tokens_ids, pruned_think_token_ids, thinkend_tokens_id], dim=0)
            pruned_prompt_token_ids.append(pruned_tokens_ids.tolist())

    return pruned_prompt_token_ids

def standardize_shift_to_positive(X, target_mean=1.0, target_std=0.5, eps=1e-18):

        # X = np.asarray(X, dtype=np.float64)
        
        # 1️⃣ 标准化
        mu = X.mean()
        sigma = X.std() + eps
        X_std = (X - mu) / sigma
        
        # 2️⃣ 平移为正
        X_shift = X_std - X_std.min() + eps
        
        # 3️⃣ 调整到目标均值与方差
        shift_mean = X_shift.mean()
        shift_std = X_shift.std() + eps
        X_scaled = (X_shift - shift_mean) / shift_std
        X_final = target_mean + X_scaled * target_std
        
        # 保证全为正数（理论上不需再裁剪，但可安全起见）
        X_final = torch.clip(X_final, eps, None)
    
        return X_final

def dynts_prune_r(datas, args):
    print("DYNTS-R Pruning")
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint_path)
    
    model = myQwen2ForCausalLM.from_pretrained(args.model_checkpoint_path,
                                               torch_dtype=torch.float16,
                                               head_type="regression",
                                               attn_implementation="flash_attention_2",
                                               device_map="auto")    
    # model = myLlamaForCausalLM.from_pretrained(args.model_checkpoint_path,
    #                                            torch_dtype=torch.float16,
    #                                            head_type="regression",
    #                                            attn_implementation="flash_attention_2",
    #                                            device_map="auto")

    loss_fn = torch.nn.MSELoss()
    model.eval()
    # pruning datas
    pruned_prompt_token_ids = []
    with torch.inference_mode():
        for data in tqdm(datas, desc="Pruning datas"):
            texts = data['texts']
            labels = data["importance_score"]
            inputs = tokenizer(texts, return_tensors="pt", add_special_tokens=False).to(model.device)
            outputs = model(**inputs)

            importance_scores = outputs.importance_scores.squeeze(0).squeeze(-1)

            inputs_ids = inputs['input_ids'][0]

            labels = standardize_shift_to_positive(torch.tensor(labels).to(model.device), target_mean=0.5, target_std=0.7)
            # labels = torch.tensor(labels).to(model.device) * 10000

            # sub_step, type_sub_step, think_index_start, think_index_end = seq_token_info(texts, tokenizer)
            think_start_token_id, think_end_token_id = get_think_token_ids(tokenizer)
            think_start_index = (inputs_ids == think_start_token_id).nonzero(as_tuple=True)[0][0].item()
            think_end_index = (inputs_ids == think_end_token_id).nonzero(as_tuple=True)[0][0].item()
            assert think_start_index < think_end_index, "think token indices error"
            if args.debug:
                print(f"think_start_index: {think_start_index}, think_end_index: {think_end_index}")

            question_tokens_ids = inputs_ids[:think_start_index+1]
            think_tokens_ids = inputs_ids[think_start_index+1:think_end_index]
            thinkend_tokens_id = inputs_ids[think_end_index].unsqueeze(0)

            think_is = importance_scores[think_start_index+1:think_end_index]
            think_labels = labels[think_start_index+1:think_end_index]
            
            order = torch.argsort(think_labels, descending=True)
            
            
            think_labels_prem = torch.gather(think_labels,-1, order)
            think_is_prem = torch.gather(think_is,-1, order)

            topk = int(len(think_labels) * args.ratio)
            think_is_topk = think_is_prem[:topk]
            think_labels_topk = think_labels_prem[:topk]

            loss_topk = loss_fn(think_is_topk, think_labels_topk)
            loss_all = loss_fn(think_is_prem, think_labels_prem)
            loss_other = loss_fn(think_is_prem[topk:], think_labels_prem[topk:])

            print(f"loss_topk: {loss_topk.item():.6f}, loss_other: {loss_other.item():.6f}, loss_all: {loss_all.item():.6f}")

            # import pdb; pdb.set_trace()
    #         think_is_value, think_is_index = torch.sort(think_is, descending=True)
    #         k = int(len(think_is_value) * args.ratio)
    #         think_is_topk_index_sorted = torch.sort(think_is_index[:k]).values
            
    #         pruned_think_token_ids = think_tokens_ids[think_is_topk_index_sorted]

    #         pruned_tokens_ids = torch.cat([question_tokens_ids, pruned_think_token_ids, thinkend_tokens_id], dim=0)
    #         pruned_prompt_token_ids.append(pruned_tokens_ids.tolist())

    # return pruned_prompt_token_ids

def is_prune(datas, args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # pruning datas
    pruned_prompt_token_ids = []
    for data in tqdm(datas, desc="Pruning datas"):
        texts = data['texts']
        inputs = tokenizer(texts, return_tensors="pt", add_special_tokens=False)
        inputs_ids = inputs['input_ids'][0]
        iss = data['importance_score']

        if len(iss) != len(inputs_ids):
            print("Error: importance score length does not match input tokens length")
            print(len(iss), len(inputs_ids))

        sub_step, type_sub_step, think_start_index, think_end_index = seq_token_info(texts, tokenizer)

        question_tokens_ids = inputs_ids[:think_start_index+1]
        think_tokens_ids = inputs_ids[think_start_index+1:think_end_index]
        thinkend_tokens_id = inputs_ids[think_end_index].unsqueeze(0)

        think_iss = iss[think_start_index+1:think_end_index]
        
        k = int(len(think_iss) * args.ratio)
        if args.method == "is-top":
            topk_indices = torch.topk(torch.tensor(think_iss), k=k, largest=True).indices
            pruned_think_token_ids = think_tokens_ids[topk_indices.sort().values]
        elif args.method == "is-bottom":
            bottomk_indices = torch.topk(torch.tensor(think_iss), k=k, largest=False).indices
            pruned_think_token_ids = think_tokens_ids[bottomk_indices.sort().values]
        elif args.method == "is-random":
            total_indices = list(range(len(think_iss)))
            if k == 0:
                k = 1   
            random_indices = random.sample(total_indices, k)
            pruned_think_token_ids = think_tokens_ids[torch.tensor(sorted(random_indices))]
        elif args.method == "is-none":
            pruned_think_token_ids = think_tokens_ids.new_empty((0,), dtype=think_tokens_ids.dtype)
        else:
            raise NotImplementedError(f"Pruning method {args.method} not implemented")
        # import pdb; pdb.set_trace()
        pruned_tokens_ids = torch.cat([question_tokens_ids, pruned_think_token_ids, thinkend_tokens_id], dim=0)
        pruned_prompt_token_ids.append(pruned_tokens_ids.tolist())

    return pruned_prompt_token_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="is")
    parser.add_argument("--ratio", type=float, default=0.8)
    parser.add_argument('--model_name_or_path', type=str, default='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
    parser.add_argument('--model_checkpoint_path', type=str, default='')
    parser.add_argument('--data_name', type=str, default='math')
    parser.add_argument('--data_dir', type=str, default='datas/train')
    parser.add_argument('--save_dir', type=str, default='outputs/inference_vllm_pruning')

    # vllm setting
    parser.add_argument("--parallel_size", type=int, default=2)
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--num_samples", type=int, default=5)

    # filter
    parser.add_argument("--filter", action='store_true')

    parser.add_argument("--debug", action='store_true', help="Run in debug mode with a smaller dataset")

    args = parser.parse_args()

    # load datas
    model_name = args.model_name_or_path.split("/")[-1]
    data_path = args.data_dir + "/" + model_name + "_" + args.data_name + '_is-om.jsonl'
    print(data_path)
    if args.filter:
        print('data filtering enabled')
        data = load_jsonl(data_path)
        q_id = []
        datas = []
        for d in data:
            if d['question_id'] not in q_id:
                q_id.append(d['question_id'])
                datas.append(d)
        print(f"Raw data size: {len(data)}; Filtered data size: {len(datas)}")
    else:
        datas = load_jsonl(data_path)

    # pruning datas
    if args.method in ["is-top", "is-bottom", "is-random", "is-none"]:
        prompt_token_ids = is_prune(datas, args)
    elif args.method == "dynts":
        prompt_token_ids = dynts_prune(datas, args)
    elif args.method == "dynts-r":
        prompt_token_ids = dynts_prune_r(datas, args)
    else:
        raise NotImplementedError(f"Pruning method {args.method} not implemented")
    
    gc.collect()
    torch.cuda.empty_cache()

    # vllm inference
    sampling_params = SamplingParams(n=args.num_samples,
                                     max_tokens=args.max_tokens,
                                     temperature=args.temperature,
                                     top_p=args.top_p,
                                     top_k=args.top_k)
    llm = LLM(model=args.model_name_or_path, 
              tokenizer=args.model_name_or_path,
              tensor_parallel_size=args.parallel_size,
              gpu_memory_utilization=args.gpu_memory_utilization,
              seed=42)

    results = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)
    outputs_texts, preds, scores = extract_text_from_results(results, data_name=args.data_name, datas=datas)
    k = args.num_samples
    q_score = {i // k: scores[i : i + k] for i in range(0, len(scores), k)}
    pass_k, major_k = [], []
    for qid, score in q_score.items():
        pass_k.append(get_passk(score, k=args.num_samples))
        major_k.append(max(set(score), key=score.count))
    # import pdb; pdb.set_trace()
    raw_scores = [data['score'] for data in datas]

    save_path = f"{args.save_dir}/vllm_pruning.jsonl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "a", encoding="utf-8") as f:
        metric = {
            "method": args.method,
            "model_name": args.model_name_or_path.split("/")[-1],
            "data_name": args.data_name,
            "ratio": args.ratio,
            "pruning_accuracy": sum(scores)/len(scores),
            "raw_accuracy": sum(raw_scores)/len(raw_scores),
            'num_samples_avg': len(scores),
            'correct_samples_avg': sum(scores),
            'num_samples': len(q_score),
            'num_passk': sum(pass_k),
            'num_majork': sum(major_k)
        }
        f.write(json.dumps(metric, ensure_ascii=False) + "\n")
    # print(f"Pruning ratio: {args.ratio}, Pruning accuracy: {sum(scores)/len(scores)}, Raw accuracy: {sum(raw_scores)/len(raw_scores)}, Num samples: {len(scores)}, Correct samples: {sum(scores)}")
    print(metric)
