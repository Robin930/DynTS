
import os
import gc
import copy
import torch
import argparse
import signal
import datetime

from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import set_start_method
from transformers import AutoTokenizer

from qw_evaluation.parser import extract_answer, parse_ground_truth, parse_question
from qw_evaluation.grader import math_equal

from utils.utils import load_jsonl, save_jsonl, set_seed, load_model_and_tokenizer, prepare_data, get_model_name, get_question_score, save_jsonl_append, get_key_token_ids, get_passk
from inference.base import Inference
from inference.dynts import DynTS
from inference.window import Window
from inference.h2o import H2O
from inference.sep import Sep
from inference.snapkv import SnapKV
from inference.rkv import RKV

# 定义超时后的处理函数
def handler(signum, frame):
    raise TimeoutError("Function call timed out!")



def handler_kill_signal(signum, frame):
    pid = os.getpid()
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] PID {pid} Received termination signal. Cleaning up...")
    gc.collect()
    torch.cuda.empty_cache()
    exit(0)


def split_batches_to_devices(datas, batch_size=16, ndev=4):
    N = len(datas)
    # 先切 batch：[(0,16), (16,32), ..., (992,1000)]
    batches = [(s, min(s+batch_size, N)) for s in range(0, N, batch_size)]
    # 再分配到设备：第 k 个 batch -> 设备 (k % ndev)
    per_dev_datas = [[] for _ in range(ndev)]
    for k, be in enumerate(batches):
        device_datas = datas[be[0]:be[1]]
        per_dev_datas[k % ndev].extend(device_datas)
    return per_dev_datas

def get_inferencer(method, think_end_token_ids, model_name, args):

    if method == "transformers":
        inferencer = Inference(
            temperature=args['temperature'],
            max_new_tokens=args['max_new_tokens'],
            top_p=args['top_p'],
            top_k=args['top_k'],
            device=args['device'],
            finsh_type=args['finsh_type'],
            max_time=args['max_time'],
            max_kvlength=args['max_kvlength'],
            debug=args['debug']
        )
    elif method == "dynts" or method == "dynts-random" or method == "dynts-bottom" or method == "dynts-noque":
        inferencer = DynTS(
            temperature=args['temperature'],
            max_new_tokens=args['max_new_tokens'],
            top_p=args['top_p'],
            top_k=args['top_k'],
            device=args['device'],
            finsh_type=args['finsh_type'],
            max_time=args['max_time'],
            max_kvlength=args['max_kvlength'],
            # dynts
            prune_signal=args['prune_signal'],
            prune_step=args['prune_step'],
            prune_kvlength=args['prune_kvlength'],
            kept_window=args['kept_window'],
            think_end_token_id=think_end_token_ids,
            debug=args['debug'],
            head_type=args['head_type'],
            ratio=args['ratio'],
            model_name=model_name,
            method=method,
            pruning_size=args['pruning_size'],
        )
    elif method == "h2o":
        print("Using H2O inference.")
        inferencer = H2O(
            temperature=args['temperature'],
            max_new_tokens=args['max_new_tokens'],
            top_p=args['top_p'],
            top_k=args['top_k'],
            device=args['device'],
            finsh_type=args['finsh_type'],
            max_time=args['max_time'],
            max_kvlength=args['max_kvlength'],
            # dynts
            prune_signal=args['prune_signal'],
            prune_step=args['prune_step'],
            prune_kvlength=args['prune_kvlength'],
            kept_window=args['kept_window'],
            think_end_token_id=think_end_token_ids,
            debug=args['debug'],
            # h2o
            ratio=args['ratio'],
        )
    elif method == "window" or method == "streaming":
        inferencer = Window(
            temperature=args['temperature'],
            max_new_tokens=args['max_new_tokens'],
            top_p=args['top_p'],
            top_k=args['top_k'],
            device=args['device'],
            finsh_type=args['finsh_type'],
            max_time=args['max_time'],
            max_kvlength=args['max_kvlength'],
            # dynts
            prune_signal=args['prune_signal'],
            prune_step=args['prune_step'],
            prune_kvlength=args['prune_kvlength'],
            kept_window=args['kept_window'],
            think_end_token_id=think_end_token_ids,
            debug=args['debug'],
            # window
            sink_type=args['sink_type'],
            sink_size=args['sink_size']
        )
    elif method == "sep":
        print("Using Sep inference.")
        inferencer = Sep(
            temperature=args['temperature'],
            max_new_tokens=args['max_new_tokens'],
            top_p=args['top_p'],
            top_k=args['top_k'],
            device=args['device'],
            finsh_type=args['finsh_type'],
            max_time=args['max_time'],
            max_kvlength=args['max_kvlength'],
            # dynts
            prune_signal=args['prune_signal'],
            prune_step=args['prune_step'],
            prune_kvlength=args['prune_kvlength'],
            kept_window=args['kept_window'],
            think_end_token_id=think_end_token_ids,
            debug=args['debug'],
            head_type=args['head_type'],
            ratio=args['ratio'],
            model_name=model_name
        )
    elif method == "snapkv":
        inferencer = SnapKV(
            temperature=args['temperature'],
            max_new_tokens=args['max_new_tokens'],
            top_p=args['top_p'],
            top_k=args['top_k'],
            device=args['device'],
            finsh_type=args['finsh_type'],
            max_time=args['max_time'],
            max_kvlength=args['max_kvlength'],
            # dynts
            prune_signal=args['prune_signal'],
            prune_step=args['prune_step'],
            prune_kvlength=args['prune_kvlength'],
            kept_window=args['kept_window'],
            think_end_token_id=think_end_token_ids,
            debug=args['debug'],
            head_type=args['head_type'],
            ratio=args['ratio'],
            ob_window=args['ob_window']
        )
    elif method == "rkv":
        inferencer = RKV(
            temperature=args['temperature'],
            max_new_tokens=args['max_new_tokens'],
            top_p=args['top_p'],
            top_k=args['top_k'],
            device=args['device'],
            finsh_type=args['finsh_type'],
            max_time=args['max_time'],
            max_kvlength=args['max_kvlength'],
            # dynts
            prune_signal=args['prune_signal'],
            prune_step=args['prune_step'],
            prune_kvlength=args['prune_kvlength'],
            kept_window=args['kept_window'],
            think_end_token_id=think_end_token_ids,
            debug=args['debug'],
            head_type=args['head_type'],
            ratio=args['ratio'],
            # rkv
            ob_window=args['ob_window'],
            sim_threshold=args['sim_threshold'],
            retain_ratio=args['retain_ratio'],
            mix_lambda=args['mix_lambda'],
            max_pool_kernel_size=args['max_pool_kernel_size']
        )
    else:
        raise ValueError("method must be one of ['transformers', 'dynts', 'h2o', 'window', 'sep', 'rkv', 'snapkv']")
    return inferencer

def get_metrics(method, model_name, data_name, statistics_datas, pass_k, major_k, avg_k, args):
    
    # "time_per_decode_step": info_time_per_decode_step,
    # "kvmem_per_decode_step": info_kvmem_per_decode_step,
    # "peak_kvcache": info_peak_kvcache,
    # "num_decode_tokens": info_num_decode_tokens

    latency, throughput_avg, throughput_last, memory, num_tokens = [], [], [], [], []
    peak_kv = 0
    for sd in statistics_datas:
        latency.append(sum(sd['time_per_decode_step']))
        throughput_avg.append(len(sd['time_per_decode_step']) / sum(sd['time_per_decode_step']) * args.batch_size)
        throughput_last.append((1/sd['time_per_decode_step'][-1]) * args.batch_size)
        memory.append(max(sd['kvmem_per_decode_step']))
        num_tokens.append(sum(sd['num_decode_tokens'])/len(sd['num_decode_tokens']))
        if sd['peak_kvcache'] > peak_kv:
            peak_kv = sd['peak_kvcache']


    if method == "transformers":
        metrics = {
            'model_name': model_name,
            'data_name': data_name,
            'num_samples': args.num_samples,
            'batch_size': args.batch_size,
            f'pass@{args.num_samples}': (sum(pass_k) / len(pass_k))*100,
            f'major@{args.num_samples}': (sum(major_k) / len(major_k))*100,
            f'avg@{args.num_samples}': (avg_k)*100,
            'throughput_avg': sum(throughput_avg) / len(throughput_avg),
            'throughput_last': sum(throughput_last) / len(throughput_last),
            'latency': sum(latency),
            'memory': max(memory),
            'num_tokens': sum(num_tokens) / len(num_tokens),
            'peak_kv': peak_kv
        }
    elif method == "dynts" or method=="dynts-random" or method=="dynts-bottom" or method=="dynts-noque" or method == "window" or method == "sep" or method == "snapkv" or method == "rkv":
        metrics = {
            'method': method,
            'max_new_tokens': args.max_new_tokens,
            'model_name': model_name,
            'data_name': data_name,
            'num_samples': args.num_samples,
            'batch_size': args.batch_size,
            'prune_signal': args.prune_signal,
            'prune_step': args.prune_step,
            'prune_kvlength': args.prune_kvlength,
            'kept_window': args.kept_window,
            'head_type': args.head_type,
            'seq_prune': args.seq_prune,
            'sink_type': args.sink_type,
            'sink_size': args.sink_size,
            'ratio': args.ratio,
            f'pass@{args.num_samples}': (sum(pass_k) / len(pass_k))*100,
            f'major@{args.num_samples}': (sum(major_k) / len(major_k))*100,
            f'avg@{args.num_samples}': (avg_k)*100,
            'throughput_avg': sum(throughput_avg) / len(throughput_avg), # tokens/s
            'throughput_last': sum(throughput_last) / len(throughput_last),
            'latency': sum(latency), 
            'memory': max(memory) , # each batch 
            'num_tokens': sum(num_tokens) / len(num_tokens), # avg tokens per sample
            'peak_kv': peak_kv
        }
    elif method == "streaming":
        metrics = {
            'model_name': model_name,
            'data_name': data_name,
            'num_samples': args.num_samples,
            'batch_size': args.batch_size,
            'prune_signal': args.prune_signal,
            'prune_step': args.prune_step,
            'prune_kvlength': args.prune_kvlength,
            'kept_window': args.kept_window,
            'ratio': args.ratio,
            f'pass@{args.num_samples}': (sum(pass_k) / len(pass_k))*100,
            f'major@{args.num_samples}': (sum(major_k) / len(major_k))*100,
            f'avg@{args.num_samples}': (avg_k)*100,
            'throughput_avg': sum(throughput_avg) / len(throughput_avg), # tokens/s
            'throughput_last': sum(throughput_last) / len(throughput_last),
            'latency': sum(latency), 
            'memory': max(memory) , # each batch 
            'num_tokens': sum(num_tokens) / len(num_tokens), # avg tokens per sample
            'peak_kv': peak_kv
        }
    elif method == "h2o":
        metrics = {
            'model_name': model_name,
            'data_name': data_name,
            'ratio': args.ratio,
            'num_samples': args.num_samples,
            'batch_size': args.batch_size,
            'prune_signal': args.prune_signal,
            'prune_step': args.prune_step,
            'prune_kvlength': args.prune_kvlength,
            'kept_window': args.kept_window,
            'ratio': args.ratio,
            f'pass@{args.num_samples}': (sum(pass_k) / len(pass_k))*100,
            f'major@{args.num_samples}': (sum(major_k) / len(major_k))*100,
            f'avg@{args.num_samples}': (avg_k)*100,
            'throughput_avg': sum(throughput_avg) / len(throughput_avg), # tokens/s
            'throughput_last': sum(throughput_last) / len(throughput_last),
            'latency': sum(latency), 
            'memory': max(memory) , # each batch 
            'num_tokens': sum(num_tokens) / len(num_tokens), # avg tokens per sample
            'peak_kv': peak_kv
        }
    else:
        raise ValueError("method must be one of ['transformers', 'dynts', 'h2o', 'window', 'snapkv', 'rkv', 'streaming']")

    return metrics

def infer_on_device(args):

    # unpack args
    device = args['device']
    dataset_slice = args['dataset_slice']
    model_name_or_path = args['model_name_or_path']
    method = args['method']
    batch_size = args['batch_size']
    head_type = args['head_type']
    print(f"Device: {device} | Data size: {len(dataset_slice)}")

    # load model and tokenizer
    model, tokenizer, model_name = load_model_and_tokenizer(model_name_or_path, method, head_type)
    model.to(device)
    model.eval()

    # get inferencer
    split_token_ids, think_start_token_id, think_end_token_id = get_key_token_ids(tokenizer) 
    inferencer = get_inferencer(method, think_end_token_id, model_name, args)

    # results 
    file_data = []
    statistics_data = []
    for i in tqdm(range(0, len(dataset_slice), batch_size), desc=f"[{device}]"):
        set_seed(args['seed'])

        batched_datas = dataset_slice[i:i+batch_size]
        batched_prompts = [data['prompt'] for data in batched_datas]
        print(f"[{device}] Infer batch [{i}:{i+len(batched_datas)}], batch size: {len(batched_datas)}")
        results = inferencer.infer(model, tokenizer, batched_prompts=batched_prompts)

        # print(f"Device {args['device_id']} - i: {i}")
    
        assert len(results[0]) == len(batched_datas)
        info_id = []
        for idx, (data, res) in enumerate(zip(batched_datas, results[0])):
            # print(idx)
            # if idx == 7:
                # if device == "cuda:1":
                #     import rpdb; rpdb.set_trace()
                #     torch.distributed.barrier()

            decode_text = tokenizer.decode(res, skip_special_tokens=True)
            if args['model_name_or_path'] in ['LGAI-EXAONE/EXAONE-Deep-7.8B']:
                pred_answer = extract_answer(decode_text.split("</thought>")[-1], data_name='math', use_last_number=False)
            else:
                pred_answer = extract_answer(decode_text.split("</think>")[-1], data_name=args['data_name'])
            
            # if len(pred_answer) >= 50:
            #     print("Long answer detected: pred_answer", pred_answer)
            #     pred_answer = ""
            # score = math_equal(pred_answer, data['ground_truth'])

            # 设置闹钟，10秒后触发
            # 注册信号
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(10)  # 设置 10 秒闹钟
            try:
                score = math_equal(pred_answer, data['ground_truth'])
                signal.alarm(0)  # 如果执行成功，取消闹钟
            except TimeoutError:
                print("math_equal timed out!")
                score = False
            except Exception as e:
                signal.alarm(0) # 确保出错也关闭闹钟
                score = False

            # score = math_equal(pred_answer, data['ground_truth'])
            file_data.append({
                "question_id": data['question_id'],
                "answer_id": data['answer_id'],
                "prompt": data['prompt'],
                "decode_text": decode_text,
                "ground_truth": data['ground_truth'],
                "pred_answer": pred_answer,
                "score": score,
                "num_decode_tokens": results[1]['num_decode_tokens'][idx]
            })
            info_id.append((data['question_id'], data['answer_id']))

        statistics_data.append({
            'info_id': info_id,
            'peak_kvcache': results[1]['peak_kvcache'],
            'time_per_decode_step': [i for i in results[1]['time_per_decode_step']],
            'kvmem_per_decode_step': [i for i in results[1]['kvmem_per_decode_step']],
            'num_decode_tokens': results[1]['num_decode_tokens']
            })

        torch.cuda.empty_cache()
    print("Finish inference on device:", device)

    return (file_data, statistics_data)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="", type=str)
    parser.add_argument("--data_dir", default="datas/dataset", type=str)
    parser.add_argument("--save_dir", default="", type=str)
    parser.add_argument("--end", default=-1, type=int)
    
    parser.add_argument("--temperature", default=0.6, type=float)
    parser.add_argument("--max_new_tokens", default=16384, type=int) # 32768
    parser.add_argument("--top_p", default=0.95, type=float)
    parser.add_argument("--top_k", default=20, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num_samples", default=1, type=int)

    parser.add_argument("--device_parallel_size", default=2, type=int)

    parser.add_argument("--method", default="transformers", type=str, choices=["transformers", "dynts", "dynts-random", "dynts-bottom", "dynts-noque", "h2o", "window", "streaming", "sep", "snapkv", "rkv"])
    parser.add_argument("--finsh_type", default="step", type=str, choices=["step", "time", "kvlength"])
    parser.add_argument("--max_time", default=None, type=int)
    parser.add_argument("--max_kvlength", default=None, type=int)

    # dynts
    parser.add_argument("--prune_signal", default='step', type=str, choices=['step', 'kvlength'])
    parser.add_argument("--prune_step", default=None, type=int)
    parser.add_argument("--prune_kvlength", default=None, type=int)
    parser.add_argument("--kept_window", default=None, type=int)
    parser.add_argument("--evaluation", action="store_true")
    parser.add_argument("--head_type", default="classification", choices=["classification", "regression"])
    parser.add_argument("--seq_prune", action="store_true")
    parser.add_argument("--pruning_size", default=None, type=int)

    # window
    parser.add_argument("--sink_type", default="none", type=str, choices=["none", "native", "question"])
    parser.add_argument("--sink_size", default=20, type=int)

    # h2o
    parser.add_argument("--ratio", default=None, type=float)

    # sanpkv
    parser.add_argument("--ob_window", default=32, type=int)

    # rkv
    parser.add_argument("--sim_threshold", default=0.5, type=float)
    parser.add_argument("--retain_ratio", default=0.2, type=float)
    parser.add_argument("--mix_lambda", default=0.1, type=float)
    parser.add_argument("--max_pool_kernel_size", default=7, type=int)
    parser.add_argument("--b_buffer", default=128, type=int)
    

    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)

    set_seed(args.seed)

    # setting multiprocessing method
    set_start_method("spawn", force=True)

    # load tokenizer and data
    model_name = get_model_name(args.model_name_or_path)
    data_name = args.data_dir.split("/")[-1]
    datas = load_jsonl(f"{args.data_dir}/test.jsonl")
    if args.end > 0:
        datas = datas[:args.end]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # if 'gpqa_d' in data_name:
    #     data_name = 'gpqa_d'
    datas = prepare_data(datas, tokenizer, data_name, model_name, args)

    print(f"Model: {model_name} | Data: {data_name} | Method: {args.method} | Data size: {len(datas)}")

    # get save path
    file_save_path = f"./{args.save_dir}/files/{model_name}_{data_name}_N{args.num_samples}-R{args.ratio}-KW{args.kept_window}-PKVL{args.prune_kvlength}.jsonl"
    statistics_save_path = f"./{args.save_dir}/statistics/{model_name}_{data_name}_N{args.num_samples}-R{args.ratio}-KW{args.kept_window}-PKVL{args.prune_kvlength}.jsonl"
    matrics_save_path = f"./{args.save_dir}/metrics.jsonl"
    print(file_save_path)
    print(statistics_save_path)


    signal.signal(signal.SIGTERM, handler_kill_signal)
    signal.signal(signal.SIGINT, handler_kill_signal)

    if args.evaluation == True and os.path.exists(file_save_path) == True and os.path.exists(statistics_save_path) == True:
        print("Load existing inference results for evaluation.")
        file_data = load_jsonl(file_save_path)
        # if args.statistics == True:
        statistics_data = load_jsonl(statistics_save_path)
    else:
        datas_for_devices = split_batches_to_devices(datas, batch_size=args.batch_size, ndev=args.device_parallel_size)

        # init per device args
        args_for_devices = []
        for idx in range(args.device_parallel_size):
            dataset_slice = datas_for_devices[idx]
            args_for_per = copy.deepcopy(vars(args))
            args_for_per['device'] = f"cuda:{idx}" if torch.cuda.is_available() else "cpu"
            args_for_per['dataset_slice'] = dataset_slice
            args_for_per['device_id'] = idx
            args_for_per['data_name'] = data_name
            args_for_devices.append(args_for_per)
        
        # multiprocessing
        outputs = []
        with Pool(processes=args.device_parallel_size, maxtasksperchild=1) as pool:
            outputs = pool.map(infer_on_device, args_for_devices)
        # outputs = infer_on_device(args_for_devices[0])
        print("Finish inference on all devices.")
        file_data, statistics_data = [], []
        for f, s in outputs:
            file_data.extend(f)
            statistics_data.extend(s)
        # sort results
        file_data = sorted(file_data, key=lambda x: (x['question_id'], x['answer_id']))

        # save results
        os.makedirs(os.path.dirname(file_save_path), exist_ok=True)
        save_jsonl(file_data, file_save_path)
        os.makedirs(os.path.dirname(statistics_save_path), exist_ok=True)
        save_jsonl(statistics_data, statistics_save_path)

    # evaluation
    # get question score list
    avg_k = sum([data['score'] for data in file_data]) / len(file_data) 
    q_score = get_question_score(file_data)
    pass_k, major_k = [], []
    for qid, scores in q_score.items():
        pass_k.append(get_passk(scores, k=args.num_samples))
        major_k.append(max(set(scores), key=scores.count))

    metrics = get_metrics(args.method, model_name, data_name, statistics_data, pass_k, major_k, avg_k, args)
    print(metrics)

    # save metrics
    os.makedirs(os.path.dirname(matrics_save_path), exist_ok=True)
    save_jsonl_append(metrics, matrics_save_path)