import json
import math
import torch
import random
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS

from models.qwen import myQwen2ForCausalLM, myQwen3ForCausalLM
from models.exaone import myExaoneForCausalLM, ExaoneForCausalLM
from qw_evaluation.parser import extract_answer, parse_ground_truth, parse_question
from utils.attention import h2o_eager_attention_forward, my_eager_attention_forward


ALL_ATTENTION_FUNCTIONS['h2o_eager'] = h2o_eager_attention_forward
ALL_MASK_ATTENTION_FUNCTIONS._global_mapping['h2o_eager'] = ALL_MASK_ATTENTION_FUNCTIONS['eager']

ALL_ATTENTION_FUNCTIONS['my_eager'] = my_eager_attention_forward
ALL_MASK_ATTENTION_FUNCTIONS._global_mapping['my_eager'] = ALL_MASK_ATTENTION_FUNCTIONS['eager']

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def save_jsonl(datas, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in datas:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + "\n")

def save_jsonl_append(data, path):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seq_token_info(text, tokenizer):
    '''
        sub_step: [(sub_start, sub_end)]; 
            split by '\n\n'
        type_sub_step: [0,1,1,1,2]
            0 is question substep, 1 is reasoing and 2 is answer
        think_index_start: the index of <think> in token list
        think_index_end: the index of </think> in token list 
    '''

    sub_step, type_sub_step = [],[]
    think_index_start, think_index_end = 0,0
    flag_type_sub_step = 0
    start, end = 3,0

    inputs = tokenizer.encode(text, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(inputs)

    for i, token in enumerate(tokens):
        if i < start: # start=3; from '<|im_start|>', 'user', 'Ċ', and skip 'Ċ' in '<think>', 'Ċ', 
            continue

        if '<think>' in token: # this is question part
            think_index_start = i
            end = i - 5
            sub_step.append((start,end))
            type_sub_step.append(flag_type_sub_step)
            start = i + 2
            flag_type_sub_step = 1 # after this token, into reasoning part
        
        if '</think>' in token:
            think_index_end = i
            end = i
            sub_step.append((start,end))
            type_sub_step.append(flag_type_sub_step)
            start = i + 2 # '\n\n'
            flag_type_sub_step = 2 # after this token, into answer part

        if 'ĊĊ' in token: # 'ĊĊ' token is '\n\n'
            end = i
            sub_step.append((start,end))
            type_sub_step.append(flag_type_sub_step)
            start = i + 1
    # import rpdb; rpdb.set_trace()
    # torch.distributed.barrier()
    # add final ans
    end = len(tokens)
    sub_step.append((start,end))
    type_sub_step.append(flag_type_sub_step)

    return sub_step, type_sub_step, think_index_start, think_index_end


def seq_token_info_v2(text, tokenizer):
    '''
        sub_step: [(sub_start, sub_end)]; 
            split by '\n\n'
        type_sub_step: [0,1,1,1,2]
            0 is question substep, 1 is reasoing and 2 is answer
        think_index_start: the index of <think> in token list
        think_index_end: the index of </think> in token list 
    '''

    sub_step, type_sub_step = [],[]
    think_index_start, think_index_end = 0,0
    flag_type_sub_step = 0
    start, end = 3,0

    inputs = tokenizer.encode(text, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(inputs)

    for i, token in enumerate(tokens):
        if i < start: # start=3; from '<|im_start|>', 'user', 'Ċ', and skip 'Ċ' in '<think>', 'Ċ', 
            continue

        if '<thought>' in token: # this is question part
            think_index_start = i
            end = i - 5
            sub_step.append((start,end))
            type_sub_step.append(flag_type_sub_step)
            start = i + 2
            flag_type_sub_step = 1 # after this token, into reasoning part
        
        if '</thought>' in token:
            think_index_end = i
            end = i
            sub_step.append((start,end))
            type_sub_step.append(flag_type_sub_step)
            start = i + 2 # '\n\n'
            flag_type_sub_step = 2 # after this token, into answer part

        if 'ĊĊ' in token: # 'ĊĊ' token is '\n\n'
            end = i
            sub_step.append((start,end))
            type_sub_step.append(flag_type_sub_step)
            start = i + 1
    # import rpdb; rpdb.set_trace()
    # torch.distributed.barrier()
    # add final ans
    end = len(tokens)
    sub_step.append((start,end))
    type_sub_step.append(flag_type_sub_step)

    return sub_step, type_sub_step, think_index_start, think_index_end

def get_think_token_ids(tokenizer):
    think_start_token_id = tokenizer.convert_tokens_to_ids("<think>")
    think_end_token_id = tokenizer.convert_tokens_to_ids("</think>")
    return think_start_token_id, think_end_token_id

def get_pad_token_id(tokenizer):
    pad_token_id = tokenizer.pad_token_id
    return pad_token_id

def load_model_and_tokenizer(model_name_or_path, method=None, head_type=None):


    if 'outputs/' in model_name_or_path:
        if 'DeepSeek-R1-Distill-Qwen-7B' in model_name_or_path:
            if head_type == 'regression':
                model = myQwen2ForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, head_type="regression", attn_implementation="my_eager")
            else:
                model = myQwen2ForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, head_type="classification", attn_implementation="my_eager")
        elif 'Qwen3-8B' in model_name_or_path:
            if head_type == 'regression':
                model = myQwen3ForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, head_type="regression", attn_implementation="my_eager")
            else:   
                model = myQwen3ForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, head_type="classification", attn_implementation="my_eager")
        elif 'DeepSeek-R1-Distill-Llama-8B' in model_name_or_path or 'Nanbeige' in model_name_or_path:
            if head_type == 'regression':
                from models.llama import myLlamaForCausalLM
                model = myLlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, head_type="regression", attn_implementation="my_eager")
            else:
                from models.llama import myLlamaForCausalLM
                model = myLlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, head_type="classification", attn_implementation="my_eager")
        elif 'EXAONE-Deep-7.8B' in model_name_or_path:
            if head_type == 'regression':
                model = myExaoneForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, head_type="regression", attn_implementation="my_eager", trust_remote_code=True)
            else:
                model = myExaoneForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, head_type="classification", attn_implementation="my_eager", trust_remote_code=True)
        model_name = model_name_or_path.split("/")[-3].split("_")[0]
    else:
        if model_name_or_path in ["LGAI-EXAONE/EXAONE-Deep-7.8B"]:
            if method == 'h2o':
                print('h2o')
                model = ExaoneForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, attn_implementation="h2o_eager", trust_remote_code=True)
            else:
                print(model_name_or_path, 'transformer')
                model = ExaoneForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, attn_implementation="my_eager", trust_remote_code=True)
                # model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, trust_remote_code=True)
            model_name = "EXAONE-Deep-7.8B"

        else:
            if method == 'h2o':
                print('h2o')
                model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, attn_implementation="h2o_eager")
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, attn_implementation="my_eager")
            model_name = model_name_or_path.split("/")[-1]
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    if "Nanbeige4" in model_name_or_path:
        tokenizer.add_special_tokens({"additional_special_tokens": ["<think>", "</think>"]})

    return model, tokenizer, model_name 

def get_model_name(model_name_or_path):
    if 'outputs/' in model_name_or_path:
        model_name = model_name_or_path.split("/")[-3].split("_")[0]
    else:
        model_name = model_name_or_path.split("/")[-1]
    return model_name

def prepare_data(datas, tokenizer, data_name, model_name, args):

    n_samples = args.num_samples
    et = True if model_name == "Qwen3-8B" else False

    new_datas = []
    for q_idx, data in enumerate(datas):
        question = parse_question(data, data_name)
        if et == True:
            # print('et true')
            texts = tokenizer.apply_chat_template(
                    [{"role": "user", "content": question.strip()}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True
                )
        else:
            texts = tokenizer.apply_chat_template(
                    [{"role": "user", "content": question.strip()}],
                    tokenize=False,
                    add_generation_prompt=True
                )
        gt = parse_ground_truth(data, data_name)[1]
        for a_idx in range(n_samples):
            new_datas.append({
                "question_id": q_idx,
                "answer_id": a_idx,
                "prompt": texts,
                "ground_truth": gt,
            })
    return new_datas

def get_question_score(datas):
    q_score = {}
    for data in datas:
        qid = data['question_id']
        if qid not in q_score:
            q_score[qid] = []
        q_score[qid].append(data['score'])
    return q_score

def get_passk(scores, k=1):
    # print('passk')
    n = len(scores)
    c = scores.count(1)
    passk = 1 - math.comb(n - c, k) / math.comb(n, k)
    return passk

def get_key_token_ids(tokenizer):
    split_token_ids = []
    think_start_token_id = -1
    think_end_token_id = -1
    for token, token_id in tokenizer.get_vocab().items():
        if "ĊĊ" in token:
            split_token_ids.append(token_id)
        if "<think>" in token:
            think_start_token_id = token_id
        if "</think>" in token:
            think_end_token_id = token_id
    return split_token_ids, think_start_token_id, think_end_token_id


def convert_is_to_seqis(importance_scores, sub_step):
    seq_importance_scores = []
    for (start, end) in sub_step:
        step_is = torch.mean(importance_scores[start:end], dim=0)
        seq_importance_scores.append(step_is)
    seq_importance_scores = torch.stack(seq_importance_scores, dim=0)
    return seq_importance_scores


def get_think_index(text, tokenizer, model_name_or_path):

    if model_name_or_path in ['LGAI-EXAONE/EXAONE-Deep-7.8B']:
        s1, s2, s3 = 389, 52040, 391  # <thought> token ids
        e1, e2, e3 = 2240, 52040, 391  # </thought> token ids
    elif model_name_or_path in ['Nanbeige/Nanbeige4-3B-Thinking-2511']:
        s1, s2, s3 = 152434, 20993, 152426  # <think> token ids
        e1, e2, e3 = 897, 20993, 152426  # </think> token ids

    inputs = tokenizer.encode(text, add_special_tokens=False)

    max_len = len(inputs)

    think_start = 0
    think_end = 0
    i = 0
    while i < max_len:
        if inputs[i] == s1 and inputs[i+1] == s2 and inputs[i+2] == s3: # <thought>
            think_start = i+2
        if inputs[i] == e1 and inputs[i+1] == e2 and inputs[i+2] == e3: # </thought>
            think_end = i
            break
        i += 1
    return think_start, think_end

def get_separator(tokenizer):
    vocab = tokenizer.get_vocab()
    separators=['.', ',', '?', '!', ';', ':', 'Ċ', 'Ġ', '<｜begin▁of▁sentence｜>']
    separator_token_ids = []
    for token, token_id in vocab.items():
        if token in separators:
            separator_token_ids.append(token_id)
    return separator_token_ids