import os
import time
import json
from sympy import im
import torch
import hydra
import random
import logging


from tqdm import tqdm
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional, Union
from tensordict import TensorDict
from contextlib import nullcontext

from utils.utils import seq_token_info, get_think_index, seq_token_info_v2

from torch import nn, optim
from torch.utils.data import Dataset

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, PreTrainedModel
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.generation import GenerationMixin
from transformers.utils import ModelOutput
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, Qwen2PreTrainedModel, Qwen2ForCausalLM
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.configuration_utils import GenerationConfig, CompileConfig
from transformers.generation.utils import GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput

from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl.trainer.fsdp_sft_trainer import FSDPSFTTrainer
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager
from verl.utils.profiler import log_gpu_memory_usage
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import get_cosine_schedule_with_warmup, get_wsd_schedule_with_warmup
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.tracking import Tracking
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_clip_grad_norm_,
    fsdp2_load_full_state_dict,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
)
from verl.utils.ulysses import (
    gather_outpus_and_unpad,
    get_ulysses_sequence_parallel_world_size,
    ulysses_pad_and_slice_inputs,
)

from utils.utils import set_seed
from models.qwen import myQwen2ForCausalLM, myQwen3ForCausalLM
from models.llama import myLlamaForCausalLM
from models.phi import myPhi3ForCausalLM
from models.exaone import myExaoneForCausalLM
from models.utils import freeze_model

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input

# Typing shortcuts
GenerateNonBeamOutput = Union[GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))



def load_model_and_tokenizer(model_name_or_path: str):
    """
    Load the model and tokenizer from the specified path.
    """

    if "Qwen3" in model_name_or_path:
        print('Use Qwen3 model')
        model = myQwen3ForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="cuda",
            return_dict_in_generate=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False
        )
    
    elif "Distill-Qwen" in model_name_or_path:
        print('Use Qwen2 model')
        model = myQwen2ForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="cuda",
            return_dict_in_generate=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False
        )
    elif "Distill-Llama" in model_name_or_path:
        print('Use Llama model')
        model = myLlamaForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="cuda",
            return_dict_in_generate=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False
        )
    elif "Phi" in model_name_or_path:
        print('Use Phi model')
        model = myPhi3ForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="cuda",
            return_dict_in_generate=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False
        )
    else:
        model = myQwen2ForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="cuda",
            return_dict_in_generate=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return model, tokenizer

class SFTDataset(Dataset):
    def __init__(self, data_path):
        self.data = self.load_jsonl(data_path)
        self.ml = self.max_length()
        # print(self.ml)
    def load_jsonl(self, path):
        datas = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                datas.append(item)
        return datas
    
    def max_length(self):
        """
        Return the maximum length of the text in the dataset.
        """
        return max(len(item["importance_score"]) for item in self.data)
        # return max(len(item["importance_score"][0]) for item in self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        texts = self.data[idx]["texts"]
        # weight = self.data[idx]["weight"]
        # importance_score = self.data[idx]["importance_score"][0]
        # rank = torch.distributed.get_rank()
        # if rank == 0:
        #     import rpdb
        #     rpdb.set_trace()
        # torch.distributed.barrier() 

        # importance_score = torch.nanmean(torch.tensor(self.data[idx]["importance_score"]), dim=0).tolist() # for two attn weights
        importance_score = self.data[idx]["importance_score"]

        # padding the importance score to the maximum length
        if len(importance_score) < self.ml:
            importance_score += [0] * (self.ml - len(importance_score))
        importance_score = torch.tensor(importance_score, dtype=torch.bfloat16)

        # if len(weight) < self.ml:
        #     weight += [1] * (self.ml - len(weight))
        # weight = torch.tensor(weight, dtype=torch.bfloat16)

        # Return a dictionary with the necessary tensors
        return {
            "texts": texts,
            "importance_score": importance_score,
            # "weight": weight,
        }

class Trainer(FSDPSFTTrainer):
    def __init__(
        self,
        config,
        device_mesh: DeviceMesh,
        ulysses_device_mesh: DeviceMesh,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ):
        self.config = config
        self.device_mesh = device_mesh
        self.ulysses_device_mesh = ulysses_device_mesh
        self.sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # normalize dp size
        self._normalize_config_bsz()

        # Set sequence parallel size
        self.config.ulysses_sequence_parallel_size = getattr(self.config, "ulysses_sequence_parallel_size", 1)
        self.use_remove_padding = getattr(self.config, "use_remove_padding", False)
        if self.device_mesh.get_rank() == 0:
            print(f"Using sequence parallel size: {self.config.ulysses_sequence_parallel_size}")
            print(f"Using remove padding: {self.use_remove_padding}")

        # load datasets
        self._build_dataloader(train_dataset, val_dataset)

        # load model and tokenizer
        self._load_model_and_tokenizer()

        # load optimizer
        self._load_optimizer()

        if self.device_mesh.get_rank() == 0:
            print("Configs:", self.config)
        self.device_name = get_device_name()


    def _load_model_and_tokenizer(self):
        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib

            importlib.import_module(self.config.model.external_lib)

        log_gpu_memory_usage("Before model allocation", logger=logger)

        trust_remote_code = self.config.model.trust_remote_code
        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # load config first
        config = AutoConfig.from_pretrained(self.config.model.model_name_or_path, trust_remote_code=trust_remote_code)
        self.model_config = config
        
        if self.config.ulysses_sequence_parallel_size > 1:
            assert self.use_remove_padding, "Sequence parallel is only supported when remove_padding is enabled"

        init_context = get_init_weight_context_manager(
            use_meta_tensor=not config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context():

            if "Qwen3" in self.config.model.model_name_or_path:
                print('Use Qwen3 model')
                self.model: PreTrainedModel = myQwen3ForCausalLM.from_pretrained(
                    self.config.model.model_name_or_path,
                    config=config,
                    torch_dtype=torch_dtype,
                    attn_implementation="flash_attention_2",
                    trust_remote_code=trust_remote_code,
                    head_type=self.config.trainer.head_type,

                )
            elif "Distill-Qwen" in self.config.model.model_name_or_path:
                print('Use Qwen2 model')
                self.model: PreTrainedModel = myQwen2ForCausalLM.from_pretrained(
                    self.config.model.model_name_or_path,
                    config=config,
                    torch_dtype=torch_dtype,
                    attn_implementation="flash_attention_2",
                    trust_remote_code=trust_remote_code,
                    head_type=self.config.trainer.head_type,
                )
            elif "Distill-Llama" in self.config.model.model_name_or_path or "Seed" in self.config.model.model_name_or_path or "Nanbeige" in self.config.model.model_name_or_path:
                print('Use Llama model')
                self.model: PreTrainedModel = myLlamaForCausalLM.from_pretrained(
                    self.config.model.model_name_or_path,
                    config=config,
                    torch_dtype=torch_dtype,
                    attn_implementation="flash_attention_2",
                    trust_remote_code=trust_remote_code,
                    head_type=self.config.trainer.head_type,
                )
            elif "Phi" in self.config.model.model_name_or_path:
                print('Use Phi model')
                self.model: PreTrainedModel = myPhi3ForCausalLM.from_pretrained(
                    self.config.model.model_name_or_path,
                    config=config,
                    torch_dtype=torch_dtype,
                    attn_implementation="flash_attention_2",
                    trust_remote_code=trust_remote_code,
                    head_type=self.config.trainer.head_type,
                )
            elif "EXAONE" in self.config.model.model_name_or_path:
                print('Use EXAONE-Deep model')
                self.model: PreTrainedModel = myExaoneForCausalLM.from_pretrained(
                    self.config.model.model_name_or_path,
                    config=config,
                    torch_dtype=torch_dtype,
                    attn_implementation="flash_attention_2",
                    trust_remote_code=trust_remote_code,
                    head_type=self.config.trainer.head_type,
                )
            else:
                print('ERROR')

            # self.model: PreTrainedModel = myQwen2ForCausalLM.from_pretrained(
            #     self.config.model.model_name_or_path,
            #     config=config,
            #     torch_dtype=torch_dtype,
            #     attn_implementation="flash_attention_2",
            #     trust_remote_code=trust_remote_code,
            # )
            if self.use_remove_padding or self.config.ulysses_sequence_parallel_size > 1:
                from verl.models.transformers.monkey_patch import apply_monkey_patch
                apply_monkey_patch(model=self.model, ulysses_sp_size=self.config.ulysses_sequence_parallel_size)

            # TODO: we can add LoRA here

        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        log_gpu_memory_usage("After model allocation", logger=logger)

        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32
        )

        # freeze the model parameters except for the is_head layer
        self.model = freeze_model(self.model, 'is_head')

        # apply FSDP2 wrap the model
        auto_wrap_policy = get_fsdp_wrap_policy(
            self.model,
            config=self.config.model.fsdp_config.wrap_policy,
            is_lora=self.config.model.get("lora_rank", 0) > 0,
        )
        if self.device_mesh.get_rank() == 0:
            print("FSDP Auto Wrap Policy:", auto_wrap_policy)

        if not self.config.model.fsdp_config.cpu_offload:
            cpu_offload = None
        else:
            cpu_offload = CPUOffload(offload_params=self.config.model.fsdp_config.offload_params)

        fsdp_strategy = self.config.model.strategy

        if fsdp_strategy == "fsdp":
            self.fsdp_model = FSDP(
                self.model,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=False,
            )
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True
            )

            fsdp_kwargs = {
                "mesh": self.device_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": True,
            }
            full_state = self.model.state_dict()
            apply_fsdp2(self.model, fsdp_kwargs, self.config.model.fsdp_config)
            fsdp2_load_full_state_dict(self.model, full_state, self.device_mesh, cpu_offload)
            self.fsdp_model = self.model
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")
    
        log_gpu_memory_usage("After FSDP wrapping", logger=logger)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.model_name_or_path)

        if self.config.model.model_name_or_path in ["Nanbeige/Nanbeige4-3B-Thinking-2511"]:
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["<think>", "</think>"]})
    
    def _load_optimizer(self):
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.fsdp_model.parameters()),
            lr=self.config.optim.lr,
            betas=self.config.optim.betas,
            weight_decay=self.config.optim.weight_decay,
        )

        log_gpu_memory_usage("After initialize optimizer", logger=logger)

        # setting step
        self.steps_per_epoch = len(self.train_dataloader)
        self.total_steps = self.steps_per_epoch * self.config.trainer.total_epochs
        if self.device_mesh.get_rank() == 0:
            print(
                f"Number of steps/epoch {self.steps_per_epoch}, number of epochs "
                f"{self.config.trainer.total_epochs}, total number of steps {self.total_steps}"
            )
        
        # setting lr warmup
        num_warmup_steps = int(self.total_steps * self.config.optim.warmup_steps_ratio)
        if not hasattr(self.config.optim, "lr_scheduler") or self.config.optim.lr_scheduler == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps
            )
        elif self.config.optim.lr_scheduler == "wsd":
            self.lr_scheduler = get_wsd_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps
            )
        else:
            raise ValueError(f"Unknown lr scheduler: {self.config.optim.lr_scheduler}")

    def train(self):
        """
        Train the model.
        """
        rank = self.device_mesh.get_rank()
        if rank == 0:
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
            )

        global_step = 0
        last_valid_metric = None
        # compute the total training steps.
        # the total training steps in SFT is mainly for early exit
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        # Loop over epochs
        for epoch in range(self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)
            for data in tqdm(
                self.train_dataloader,
                total=self.steps_per_epoch,
                desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
                disable=rank != 0,
            ):
                global_step += 1

                # Process the batch and create a TensorDict for training
                batch = self.process_batch(data)
                data = TensorDict(batch, batch_size=self.config.data.train_batch_size).to(self.device_name)
                

                # Forward pass
                metric = self.training_step(data)

                if rank == 0:
                    tracking.log(data=metric, step=global_step)

                is_last_step = global_step >= self.total_training_steps
                is_valid_step = global_step % self.config.trainer.test_freq == 0
                is_save_step = global_step % self.config.trainer.save_freq == 0

                if self.config.trainer.test_freq == -1:
                    is_valid_step = False

                if rank == 0:
                    print(is_last_step, is_valid_step, is_save_step, global_step)

                # early exit or validation step
                if is_last_step or (self.config.trainer.test_freq > 0 and is_valid_step):
                    # Perform validation
                    val_losses = []
                    for val_data in self.val_dataloader:
                        val_batch = self.process_batch(val_data)
                        val_data = TensorDict(val_batch, batch_size=self.config.data.micro_batch_size_per_gpu).to(
                            self.device_name
                        )
                        val_loss = self.validation_step(val_data)
                        val_losses.append(val_loss)
                    if rank == 0:
                        val_loss = torch.mean(torch.stack(val_losses))
                        metric = {"val/loss": val_loss.detach().item()}
                        tracking.log(data=metric, step=global_step)
                        last_valid_metric = metric
                    torch.distributed.barrier()
                    print("eval yes")
                if is_last_step or (self.config.trainer.save_freq > 0 and is_save_step):
                    print("save")
                    self.save_checkpoint(step=global_step)

                if is_last_step:
                    if rank == 0:
                        print(f"Final validation metrics: {last_valid_metric}")
                    return

                # if rank == 0:
                #     import rpdb
                #     rpdb.set_trace()
                # torch.distributed.barrier()  # 调试完通知其他继续


    def save_is_head(self, step):
        pass


    def kendall_tau_loss(
        self,
        y_true: torch.Tensor,     # (B, N)
        y_pred: torch.Tensor,     # (B, N)
        tau: float = 1.0,         # 温度
        mode: str = "logistic",   # 'logistic'|'sigmoid'|'hinge'
        margin: float = 0.0,      # 仅 hinge 用
        mask: torch.Tensor | None = None  # (B, N) 1=有效, 0=padding
    ):
        assert y_true.shape == y_pred.shape
        B, N = y_true.shape

        if mask is None:
            mask = torch.ones_like(y_true, dtype=torch.bool)
        else:
            mask = mask.bool()

        # 仅比较有效位置两两组合：构造 (B,N,N) 的有效对矩阵
        m1 = mask.unsqueeze(2) & mask.unsqueeze(1)   # (B,N,N)
        # 真实差与预测差
        dy = y_true.unsqueeze(2) - y_true.unsqueeze(1)  # (B,N,N)
        dp = y_pred .unsqueeze(2) - y_pred .unsqueeze(1)

        # 上三角去重 & 去对角
        tri = torch.triu(torch.ones(N, N, dtype=torch.bool, device=y_true.device), diagonal=1)
        valid = (m1 & tri)  # (B,N,N)

        # ties：若 y_i==y_j，通常不计入；也可给小权重
        not_tie_y = (dy != 0)
        valid = valid & not_tie_y

        sgn = torch.sign(dy).float()                 # (-1,0,1)，这里 0 已被过滤
        z = (sgn * dp) / (tau + 1e-12)               # 一致性分数

        # 取有效对
        z = z[valid]

        if mode == "logistic":
            loss_pairs = nn.functional.softplus(-z)              # log(1+e^{-z})
        elif mode == "sigmoid":
            loss_pairs = 1.0 - torch.sigmoid(z)
        elif mode == "hinge":
            loss_pairs = torch.clamp(margin - z, min=0)
        else:
            raise ValueError("mode must be logistic|sigmoid|hinge")

        # 按 batch 聚合
        # 为了按 batch 规约，需要每个样本的对数。构建每样本 pair 数量:
        pair_counts = valid.view(B, -1).sum(dim=1).clamp_min(1)
        # 为了聚合，先填回 (B, N, N) 再汇总
        Lmat = torch.zeros((B, N, N), device=y_true.device, dtype=z.dtype)
        Lmat[valid] = loss_pairs
        loss_per_sample = (Lmat.sum(dim=(1,2)) / pair_counts)  # (B,)
        return loss_per_sample.mean()
    
    def kendall_tau_loss_v2(
        self,
        scores: torch.Tensor,      # (B, N) 预测打分 s
        labels: torch.Tensor,      # (B, N) 真实分值 y（连续也可）
        mask: torch.Tensor | None = None,  # (B, N) 1=有效, 0=padding
        tau: float = 1.0,          # 温度（越小越尖锐）
    ) -> torch.Tensor:
        """
        Kendall Tau 排序损失：对所有有效位置两两组合 (i, j)，做 pairwise logistic loss:
            z_ij = (s_i - s_j) / tau
            loss_ij = softplus(-sgn(y_i - y_j) * z_ij) = log(1 + exp(-sgn(y_i - y_j) * (s_i - s_j)/tau))
        这样 s_i > s_j 越多，loss 越小；
        """
        assert scores.shape == labels.shape
        B, N = scores.shape
        device = scores.device

        if mask is None:
            mask = torch.ones_like(scores, dtype=torch.bool)
        else:
            mask = mask.bool()

        pair_losses = []
        for b in range(B):
            valid = mask[b]                    # (N,)
            n_valid = int(valid.sum().item())

            if n_valid <= 1:
                continue

            # 只取有效位置上的 label / score
            y = labels[b][valid]               # (n_valid,)
            s = scores[b][valid]               # (n_valid,)

            # 构建所有成对组合
            dy = y.unsqueeze(1) - y.unsqueeze(0)  # (n_valid, n_valid)
            dp = s.unsqueeze(1) - s.unsqueeze(0)  # (n_valid, n_valid)

            # 构建量化权重
            order = torch.argsort(y, descending=True)  # (n_valid,)
            rank = torch.empty_like(order)
            rank[order] = torch.arange(len(order), device=device)  # (n_valid,)，从 0 开始
            tri = torch.triu(torch.ones(n_valid, n_valid, dtype=torch.bool, device=device), diagonal=1)
            rank_diff = torch.abs(rank.unsqueeze(1) - rank.unsqueeze(0))  # (n_valid, n_valid)
            rank_diff = rank_diff.masked_select(tri)
            rank_w = rank_diff / rank_diff.max()
            rank_w = torch.softmax(rank_w, dim=-1)

            # y_w = y.unsqueeze(1).repeat(1, n_valid)
            # y_w = y_w.masked_select(tri)
            # y_w = y_w / y_w.max()
            # y_w = torch.softmax(y_w, dim=-1)
            # dp = torch.tanh(10*dp)  # 放大差异

            sgn = torch.sign(dy).float()          # (-1,0,1)
            z = (sgn * dp) / (tau + 1e-16)       # 一致性分数

            loss_ij = nn.functional.softplus(-z)  # log(1+e^{-sgn(y_i-y_j)*z_ij})

            # 仅聚合上三角去重 & 去对角
            # tri = torch.triu(torch.ones(n_valid, n_valid, dtype=torch.bool, device=device), diagonal=1)
            loss_ij = loss_ij.masked_select(tri) 
            loss_ij = rank_w * loss_ij

            pair_loss = loss_ij.sum()
            pair_losses.append(pair_loss)

        pair_loss = torch.stack(pair_losses).mean()

        return pair_loss

    def kendall_tau_loss_v3(
        self,
        scores: torch.Tensor,      # (B, N) 预测打分 s
        labels: torch.Tensor,      # (B, N) 真实分值 y（连续也可）
        mask: torch.Tensor | None = None,  # (B, N) 1=有效, 0=padding
        tau: float = 1.0,          # 温度（越小越尖锐）
    ) -> torch.Tensor:
        """
        Kendall Tau 排序损失：对所有有效位置两两组合 (i, j)，做 pairwise logistic loss:
            z_ij = (s_i - s_j) / tau
            loss_ij = softplus(-sgn(y_i - y_j) * z_ij) = log(1 + exp(-sgn(y_i - y_j) * (s_i - s_j)/tau))
        这样 s_i > s_j 越多，loss 越小；
        """
        assert scores.shape == labels.shape
        B, N = scores.shape
        device = scores.device

        if mask is None:
            mask = torch.ones_like(scores, dtype=torch.bool)
        else:
            mask = mask.bool()

        pair_losses = []
        for b in range(B):
            valid = mask[b]                    # (N,)
            n_valid = int(valid.sum().item())

            if n_valid <= 1:
                continue

            # 只取有效位置上的 label / score
            y = labels[b][valid]               # (n_valid,)
            s = scores[b][valid]               # (n_valid,)

            # 构建所有成对组合
            dy = y.unsqueeze(1) - y.unsqueeze(0)  # (n_valid, n_valid)
            dp = s.unsqueeze(1) - s.unsqueeze(0)  # (n_valid, n_valid)

            # 构建量化权重
            order = torch.argsort(y, descending=True)  # (n_valid,)
            rank = torch.empty_like(order)
            rank[order] = torch.arange(len(order), device=device)  # (n_valid,)，从 0 开始
            tri = torch.triu(torch.ones(n_valid, n_valid, dtype=torch.bool, device=device), diagonal=1)
            
            rank_diff = torch.abs(rank.unsqueeze(1) - rank.unsqueeze(0))  # (n_valid, n_valid)
            rank_diff = rank_diff.masked_select(tri).float()
            rank_w = rank_diff / rank_diff.mean()

            sgn = torch.sign(dy).float()          # (-1,0,1)
            z = (sgn * dp) / (tau + 1e-16)       # 一致性分数

            mask_z = (z < 0).bool()

            loss_ij = nn.functional.softplus(-z)  # log(1+e^{-sgn(y_i-y_j)*z_ij})
            loss_ij = torch.where(mask_z, loss_ij, torch.zeros_like(loss_ij))
            

            # 仅聚合上三角去重 & 去对角
            # tri = torch.triu(torch.ones(n_valid, n_valid, dtype=torch.bool, device=device), diagonal=1)
            loss_ij = loss_ij.masked_select(tri) 
            loss_ij = rank_w * loss_ij

            pair_loss = torch.mean(loss_ij)
            pair_losses.append(pair_loss)
            # import rpdb; rpdb.set_trace()
            # torch.distributed.barrier()

        pair_loss = torch.stack(pair_losses).mean()

        return pair_loss


    def pairwise_rank_loss(
        self,
        scores: torch.Tensor,      # (B, N) 预测打分 s
        labels: torch.Tensor,      # (B, N) 真实分值 y（连续也可）
        mask: torch.Tensor | None = None,  # (B, N) 1=有效, 0=padding
        tau: float = 1.0,          # 温度（越小越尖锐）
        ratio: float = 0.3,        # 取前多少比例作为 head
    ) -> torch.Tensor:
        """
        Pairwise 排序损失：重点约束「标签前 10%」在预测上要大于「后 90%」。
        - 对每个 batch 样本，先根据 labels 找到有效位置中的前 10% 作为 head，其余作为 tail；
        - 对 head vs tail 所有成对组合 (i in head, j in tail)，做 pairwise logistic loss:
            z_ij = (s_i - s_j) / tau
            loss_ij = softplus(-z_ij) = log(1 + exp(-(s_i - s_j)/tau))
        这样 s_i > s_j 越多，loss 越小；
        """
        assert scores.shape == labels.shape
        B, N = scores.shape
        device = scores.device

        if mask is None:
            mask = torch.ones_like(scores, dtype=torch.bool)
        else:
            mask = mask.bool()

        head_tail_losses = []
        for b in range(B):
            valid = mask[b]                    # (N,)
            n_valid = int(valid.sum().item())

            if n_valid <= 1:
                continue

            # 只取有效位置上的 label / score
            y = labels[b][valid]               # (n_valid,)
            s = scores[b][valid]               # (n_valid,)

            # ===== 1. 找到标签前 10% 的 head 集合 同时构建 rank weight=====
            k = max(1, int(len(y) * ratio))  # 至少 1 个
            top_vals, top_idx = torch.topk(y, k=k, largest=True, sorted=False)  # (k,)
            order = torch.argsort(y, descending=True)  # (n_valid,)
            rank = torch.empty_like(order)
            rank[order] = torch.arange(len(order), device=device)  # (n_valid,)，从 0 开始


            # head / tail 的本地 mask（在压缩后的 valid 索引内）
            local_head_mask = torch.zeros_like(y, dtype=torch.bool)
            local_head_mask[top_idx] = True
            local_tail_mask = ~local_head_mask  # 其余有效位置都视为 tail

            if local_tail_mask.sum() == 0:
                # 极端情况：全都被当成 head，就没法做 head vs tail 对比
                continue

            s_head = s.masked_select(local_head_mask)     # (H,)
            s_tail = s.masked_select(local_tail_mask)     # (T,)
            rank_head = rank.masked_select(local_head_mask)  # (H,)
            rank_tail = rank.masked_select(local_tail_mask)  # (T,)


            # ===== 2. Head vs Tail 的 pairwise loss =====
            z_ht =  s_head.unsqueeze(1) - s_tail.unsqueeze(0)  # (H, T)
            mask_z_ht = (z_ht < 0).bool()   
            loss_ij = nn.functional.softplus(-z_ht)
            loss_ij = torch.where(mask_z_ht, loss_ij, torch.zeros_like(loss_ij))

            z_rank = torch.abs(rank_head.unsqueeze(1) - rank_tail.unsqueeze(0)).float()
            z_rank = z_rank / z_rank.mean()

            loss_ij = torch.mean(z_rank * loss_ij)
            # import rpdb; rpdb.set_trace()
            # torch.distributed.barrier()
            head_tail_losses.append(loss_ij)

        head_tail_loss = torch.stack(head_tail_losses).mean()

        return head_tail_loss

    def topk_error_rate(
        self,
        scores: torch.Tensor,      # (B, N) 预测打分 s
        labels: torch.Tensor,      # (B, N) 真实分值 y（连续也可）
        mask: torch.Tensor | None = None,  # (B, N) 1=有效, 0=padding
        ratio: float = 0.3,        # 取前多少比例作为 head
    ) -> torch.Tensor:
        """
        Pairwise 排序损失：重点约束「标签前 10%」在预测上要大于「后 90%」。
        - 对每个 batch 样本，先根据 labels 找到有效位置中的前 10% 作为 head，其余作为 tail；
        - 对 head vs tail 所有成对组合 (i in head, j in tail)，做 pairwise logistic loss:
            z_ij = (s_i - s_j) / tau
            loss_ij = softplus(-z_ij) = log(1 + exp(-(s_i - s_j)/tau))
        这样 s_i > s_j 越多，loss 越小；
        """
        assert scores.shape == labels.shape
        B, N = scores.shape
        device = scores.device

        if mask is None:
            mask = torch.ones_like(scores, dtype=torch.bool)
        else:
            mask = mask.bool()

        errors = []
        for b in range(B):
            valid = mask[b]                    # (N,)
            n_valid = int(valid.sum().item())

            if n_valid <= 1:
                continue

            # 只取有效位置上的 label / score
            y = labels[b][valid]               # (n_valid,)
            s = scores[b][valid]               # (n_valid,)

            # ===== 1. 找到标签前 10% 的 head 集合 同时构建 rank weight=====
            k = max(1, int(len(y) * ratio))  # 至少 1 个
            top_vals, top_idx = torch.topk(y, k=k, largest=True, sorted=False)  # (k,)
  
            # head / tail 的本地 mask（在压缩后的 valid 索引内）
            local_head_mask = torch.zeros_like(y, dtype=torch.bool)
            local_head_mask[top_idx] = True
            local_tail_mask = ~local_head_mask  # 其余有效位置都视为 tail

            if local_tail_mask.sum() == 0:
                # 极端情况：全都被当成 head，就没法做 head vs tail 对比
                continue

            s_head = s.masked_select(local_head_mask)     # (H,)
            s_tail = s.masked_select(local_tail_mask)     # (T,)

            # ===== 2. Head vs Tail 的 pairwise loss =====
            # z_{ij} = (s_head_i - s_tail_j), s_head_i - s_tail_j< 0; 0 , s_head_i - s_tail_j > 0
            z_ht = s_head.unsqueeze(1) - s_tail.unsqueeze(0)  # (H, T)
            z_ht = torch.where(z_ht < 0, torch.ones_like(z_ht), torch.zeros_like(z_ht))  # 只惩罚错误排序的对

            error_rate = torch.sum(z_ht) / (z_ht.shape[0] * z_ht.shape[1])
            errors.append(error_rate)

        error_rate = torch.stack(errors).mean()

        return error_rate
    

    def isin(
        self,
        scores: torch.Tensor,      # (B, N) 预测打分 s
        labels: torch.Tensor,      # (B, N) 真实分值 y（连续也可）
        mask: torch.Tensor | None = None,  # (B, N) 1=有效, 0=padding
        ratio: float = 0.1,        # 取前多少比例作为 head
    ) -> torch.Tensor:
        """
        Pairwise 排序损失：重点约束「标签前 10%」在预测上要大于「后 90%」。
        - 对每个 batch 样本，先根据 labels 找到有效位置中的前 10% 作为 head，其余作为 tail；
        - 对 head vs tail 所有成对组合 (i in head, j in tail)，做 pairwise logistic loss:
            z_ij = (s_i - s_j) / tau
            loss_ij = softplus(-z_ij) = log(1 + exp(-(s_i - s_j)/tau))
        这样 s_i > s_j 越多，loss 越小；
        """
        assert scores.shape == labels.shape
        B, N = scores.shape
        device = scores.device

        if mask is None:
            mask = torch.ones_like(scores, dtype=torch.bool)
        else:
            mask = mask.bool()

        pred_rate = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        isin_dict = {pr: [] for pr in pred_rate}
        for b in range(B):
            valid = mask[b]                    # (N,)
            n_valid = int(valid.sum().item())

            if n_valid <= 1:
                continue

            # 只取有效位置上的 label / score
            y = labels[b][valid]               # (n_valid,)
            s = scores[b][valid]               # (n_valid,)

            k = max(1, int(len(y) * ratio))
            argsort_y = torch.argsort(y, descending=True)
            argsort_s = torch.argsort(s, descending=True)

            top_argsort_y = argsort_y[:k]

            for sp in pred_rate:
                sk = max(1, int(len(s) * sp))
                top_argsort_s = argsort_s[:sk]
                mask_isin = torch.isin(top_argsort_s, top_argsort_y)
                isin_dict[sp].append((sum(mask_isin) / len(top_argsort_y)).item())

        isin = {key: round(sum(value)/len(value), 5) for key, value in isin_dict.items()}
        # import rpdb; rpdb.set_trace()
        # torch.distributed.barrier()

        return isin
    
    def listmle_weighted_loss(
        self,
        scores: torch.Tensor,      # (B, N) 预测打分 s
        labels: torch.Tensor,      # (B, N) 真实分值 y（连续也可）
        mask: torch.Tensor | None = None,  # (B, N) 1=有效, 0=padding
        tau: float = 1.0,          # 温度（越小越尖锐）
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Weighted ListMLE（按真实分值加权），支持 batch 与 mask，O(B*N) 稳定实现。
        L = sum_j w_{pi_j} * (logsumexp_{k>=j}(s_{pi_k}/tau) - s_{pi_j}/tau) / sum_j w_{pi_j}
        最后在 batch 上取均值。
        """
        assert scores.shape == labels.shape
        B, N = scores.shape
        device = scores.device

        if mask is None:
            mask = torch.ones_like(scores, dtype=torch.bool)
        else:
            mask = mask.bool()

        # 1) 基于真实分值 labels 的降序得到真序 π（无效位设为 -inf 防止进入前缀）
        y_masked = labels.clone()
        y_masked[~mask] = float("-inf")
        order = torch.argsort(y_masked, dim=-1, descending=True, stable=True)  # (B,N)

        # 2) 重排 scores/mask 到真序，并除以 tau
        s_perm = torch.gather(scores, dim=1, index=order) / tau
        m_perm = torch.gather(mask,   dim=1, index=order)

        # 3) 由真实分值构造权重 w，并按真序重排
        # w_raw  = _weights_from_labels(labels, mask, mode=w_mode, power=w_power, eps=eps)  # (B,N)
        w_raw = torch.log1p(labels.clone()) # 避免负权重
        w_perm = torch.gather(w_raw, dim=1, index=order) * m_perm  # (B,N)

        # 4) 计算从尾到头的 logsumexp：tail_lse[j] = logsumexp(s_perm[j:])
        tail_lse = torch.logcumsumexp(torch.flip(s_perm, dims=[-1]), dim=-1)
        tail_lse = torch.flip(tail_lse, dims=[-1])  # (B,N)

        # 5) 逐位置损失项与加权
        per_pos = (tail_lse - s_perm) * m_perm.to(s_perm.dtype)  # (B,N)
        num = (per_pos * w_perm).sum(dim=1)                      # (B,)
        den = w_perm.sum(dim=1).clamp_min(eps)                   # (B,)
        loss_per_sample = num / den

        # 6) 在 batch 维度上求均值
        return loss_per_sample.mean()

    def listnet_loss(
        self,
        scores: torch.Tensor,      # (B, N) 预测打分 s
        labels: torch.Tensor,      # (B, N) 真实分值 y（连续也可）
        mask: torch.Tensor | None = None,  # (B, N) 1=有效, 0=padding
        tau: float = 1.0,          # 温度（越小越尖锐）
        eps: float = 1e-16,
    ) -> torch.Tensor:
        """
        ListNet 损失，支持 batch 与 mask，O(B*N) 稳定实现。
        L = - sum_i P_{y}(i) * log P_{s}(i)
        其中 P_{y}(i) = exp(y_i / tau) / sum_j exp(y_j / tau)，P_{s}(i) 类似定义。
        最后在 batch 上取均值。
        """
        assert scores.shape == labels.shape
        B, N = scores.shape
        device = scores.device

        if mask is None:
            mask = torch.ones_like(scores, dtype=torch.bool)
        else:
            mask = mask.bool()
        
        losses = []
        for b in range(B):
            valid = mask[b]                    # (N,)
            n_valid = int(valid.sum().item())

            if n_valid <= 1:
                continue

            # 只取有效位置上的 label / score
            y = labels[b][valid]               # (n_valid,)
            s = scores[b][valid]               # (n_valid,)

            # 计算 P_y 与 P_s
            p_y = torch.softmax(y / tau, dim=0)  # (n_valid,)
            p_s = torch.softmax(s / tau, dim=0)  # (n_valid,)

            # 计算交叉熵损失
            loss = - (p_y * torch.log(p_s + eps)).sum()
            losses.append(loss)
        
        listnet_loss = torch.stack(losses).mean()
        return listnet_loss
        
    def weight_bin(self, scores, mask, bins = 5, eps = 1e-16, alpha = 1.0) -> torch.Tensor:
        """
        将 scores 分成若干个 bin，并返回每个 bin 的权重（分数和）。
        """
        B, N = scores.shape

        scores = scores.view(-1)
        mask = mask.view(-1)

        mask_bool = mask.bool()
        v_scores = scores[mask_bool]
    
        min_score = v_scores.min()
        max_score = v_scores.max()
        bin_edges = torch.linspace(min_score, max_score, bins + 1, device=scores.device)
        bin_ids = torch.bucketize(v_scores, bin_edges, right=False) - 1
        bin_ids = bin_ids.clamp(0, bins - 1)
        counts = torch.bincount(bin_ids, minlength=bins)
        inv_freq = ((1.0 / (counts + eps)) ** alpha)
        w_valid = inv_freq[bin_ids] 
        # w_valid = w_valid * (w_valid.numel() / (w_valid.sum() + eps))
        w_valid = w_valid / w_valid.mean()

        weights = torch.zeros_like(scores, dtype=scores.dtype)
        weights[mask_bool] = w_valid.to(dtype=weights.dtype)

        weights = weights.view(B, N)

        return weights

    def _compute_loss_and_backward(self, batch, do_backward=True):
        """Compute loss with optional sequence parallelism and remove padding features"""
        use_sp = self.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1

        # Move inputs to GPU and prepare loss mask
        input_ids = batch["input_ids"].to(self.device_name)
        attention_mask = batch["attention_mask"].to(self.device_name)
        position_ids = batch["position_ids"].to(self.device_name)
        importance_scores = batch["importance_scores"].to(self.device_name)
        # weights = batch['weights'].to(self.device_name)
        is_mask = batch["is_mask"].to(self.device_name)

        if self.config.trainer.head_type == "classification":
            loss_fct = nn.CrossEntropyLoss(reduction="none")
        elif self.config.trainer.head_type == "regression":
            loss_fct = nn.MSELoss(reduction="none")
        
        # Context manager for sequence parallel if needed
        context = self.sharding_manager if use_sp else nullcontext()
        with context, torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            if not use_sp:
                # Standard forward pass without sequence parallel
                labels = importance_scores
                output = self.fsdp_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                )
                
                # 分类
                if self.config.trainer.head_type == "classification":
                    pred_is = output.importance_scores.view(-1,2)
                    labels = importance_scores.view(-1)
                    is_mask = is_mask.view(-1)
                    if pred_is.shape[0] != labels.shape[0]:
                        # import rpdb; rpdb.set_trace()
                        assert f"shape mismatch: pred {pred_is.shape}, label {labels.shape}"
                    loss = loss_fct(pred_is, labels)
                    loss = loss * is_mask.to(loss.device)
                elif self.config.trainer.head_type == "regression":
                    # Remove the last dimension for importance scores
                    
                    pred_is = output.importance_scores.squeeze(-1)
                    is_mask = is_mask.to(pred_is.device)
                    # weights = weights.to(pred_is.device)
                    w_is_mask1 = labels < 2.5 # 2 5
                    w_is_mask2 = (labels >= 4) & (labels < 5) # 3 4
                    # w_is_mask3 = (labels >= 5) # 5 7
                    w_is = torch.where(w_is_mask1, torch.tensor(1, device=is_mask.device), torch.tensor(1, device=is_mask.device))
                    w_is = torch.where(w_is_mask2, torch.tensor(5, device=is_mask.device), w_is)
                    # w_is = torch.where(w_is_mask3, torch.tensor(7, device=is_mask.device), w_is)
                    loss1 = w_is * loss_fct(pred_is, labels) * is_mask
                    # loss1 = loss_fct(pred_is, labels) * is_mask
                    # import rpdb; rpdb.set_trace()   
                    # torch.distributed.barrier()
                    # loss1 = loss_fct(pred_is, labels) * is_mask
                    # import rpdb; rpdb.set_trace()
                    # torch.distributed.barrier()
                    # torch.masked_select

                    # is_mask = is_mask.to(loss1.device)
                    # loss1 = loss1 * is_mask
                    # loss1 = torch.log(torch.cosh(pred_is - labels)) * is_mask

                    # topk loss
                    # ratio = 0.05
                    # masked_labels = labels * is_mask
                    # k = (torch.min(torch.sum(is_mask,dim=-1)) * ratio).int().item()
                    # topk_masked_labels, index = torch.topk(masked_labels,k,dim=-1)
                    # topk_pred_is = torch.gather(pred_is, dim=-1, index=index)
                    # loss_topk = torch.mean(loss_fct(topk_pred_is, topk_masked_labels))

                    
                    # loss_listnet = self.listnet_loss(
                    #     scores=pred_is,
                    #     labels=labels,
                    #     mask=is_mask,
                    #     tau=1.0
                    # )

                    # loss_pair = self.pairwise_rank_loss(
                    #     scores=pred_is,
                    #     labels=labels,
                    #     mask=is_mask,
                    #     tau=1.0,
                    #     ratio=0.1
                    # )

                    # error = self.topk_error_rate(
                    #     scores=pred_is,
                    #     labels=labels,
                    #     mask=is_mask,
                    #     ratio=0.05
                    # )

                    isin = self.isin(
                        scores=pred_is,
                        labels=labels,
                        mask=is_mask,
                        ratio=0.2
                    )

                    if self.config.trainer.ktl == True:
                        loss_kendall = self.kendall_tau_loss(
                            y_pred=pred_is,
                            y_true=labels,
                            mask=is_mask,
                            tau=0.5
                        )


                    # loss_kendall = self.kendall_tau_loss_v2(
                    #     scores=pred_is,
                    #     labels=labels,
                    #     mask=is_mask,
                    #     tau=0.5
                    # )
                    # import rpdb; rpdb.set_trace()
                    # loss2 = self.listmle_weighted_loss(
                    #     scores=pred_is,
                    #     labels=labels,
                    #     mask=is_mask,
                    #     tau=0.5
                    # )
    
                # rank = torch.distributed.get_rank()
                # if rank == 0:
                #     import rpdb
                #     rpdb.set_trace()
                # torch.distributed.barrier() 

            else:
                # IMPORTANT: We have a big assumption here, so we can shard the SAME sequence across SP ranks
                # i.e., each GPU has <1 sequence, and each SP group has 1 sequence
                # 1. All SP ranks will receive the *SAME* batch
                # 2. Different SP groups will receive *DIFFERENT* batches
                # This is implemented by the DistributedSampler

                batch_size, seqlen = input_ids.shape

                # all batch > 1, and removed padding. 相当于将整个没有padding的batch数据拼接在一起，
                input_ids_rmpad, indices, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # Unpad position_ids to align rotary. 相当于将位置编码拼接在一起
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

                # Pad and slice inputs for sequence parallelism. 相当于划分数据到不同的SP组
                input_ids_rmpad_sliced, position_ids_rmpad_padded, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=get_ulysses_sequence_parallel_world_size()
                )

                # For computing loss. 这里需要将importance_scores也进行相同的处理用于计算loss
                # importance_scores_rmpad, is_indices, *_ = unpad_input(
                #     importance_scores.unsqueeze(-1), attention_mask
                # )
                importance_scores_rmpad = index_first_axis(
                    rearrange(importance_scores.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)
                importance_scores_rmpad_sliced, _, _ = ulysses_pad_and_slice_inputs(
                    importance_scores_rmpad, None, get_ulysses_sequence_parallel_world_size()
                )


                # input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)
                # input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                #     input_ids_rmpad_rolled, None, get_ulysses_sequence_parallel_world_size()
                # )
                # input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # Forward pass
                output = self.fsdp_model(
                    input_ids=input_ids_rmpad_sliced,
                    attention_mask=None,  # Not needed with flash attention varlen
                    position_ids=position_ids_rmpad_padded,
                    use_cache=False,
                )
                is_pred_rmpad_sliced = output.importance_scores.squeeze(-1)  # Remove the last dimension for importance scores

                # Compute loss locally then aggregate
                loss = loss_fct(is_pred_rmpad_sliced, importance_scores_rmpad_sliced)
                # Gather and unpad for sequence parallelism
                loss = loss.squeeze(0)  # Remove the first dimension for importance scores
                loss = gather_outpus_and_unpad(loss, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                
                # This is the loss collected from all ulysses ranks
                full_loss = pad_input(hidden_states=loss.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen).squeeze(-1)
                
                loss = full_loss * is_mask.to(full_loss.device)


            valid_token_this_rank = torch.sum(is_mask)

            if self.config.data.balance_dp_token:
                torch.distributed.all_reduce(valid_token_this_rank)
                dp_size = self.ulysses_device_mesh.size("dp") if use_sp else torch.distributed.get_world_size()
            else:
                dp_size = 1

            if self.config.trainer.head_type == "classification":
                loss = torch.sum(loss) / (valid_token_this_rank + 1e-8) * dp_size 
            elif self.config.trainer.head_type == "regression":
                loss1 = torch.sum(loss1) / (valid_token_this_rank + 1e-8) * dp_size
                # loss = self.config.trainer.alpha * loss1 + (1-self.config.trainer.alpha) * loss_kendall + 2 * loss_topk
                # loss = self.config.trainer.alpha * loss1 + (1 - self.config.trainer.alpha) * loss_pair
                # loss = self.config.trainer.alpha * loss1 + (1 - self.config.trainer.alpha) * loss_listnet
                # loss = self.config.trainer.alpha * loss1
                if self.config.trainer.ktl == True:
                    loss = self.config.trainer.alpha * loss1 + (1-self.config.trainer.alpha) * (loss_kendall )
                else:
                    loss = loss1
            else:
                raise ValueError("head_type must be classification|regression")

            if do_backward:
                loss.backward()

            if self.device_mesh.get_rank() == 0:
                if self.config.trainer.head_type == "regression":
                    # print(f"Loss: {loss.item():.4f}, Loss1: {loss1.item():.4f}, Loss listnet: {loss_listnet.item():.4f}, Loss_topk: {loss_topk.item():.4f}, Loss kt: {loss_kendall.item():.4f}")
                    # print(f"Loss: {loss.item():.4f}, Loss1: {loss1.item():.4f}, Loss_pair: {loss_pair.item():.4f}, Error: {error.item():.4f}")
                    # print(f"Loss: {loss.item():.4f}, Loss1: {loss1.item():.4f}, Loss_listnet: {loss_listnet.item():.4f}, Error: {error.item():.4f}")
                    # print(f"Loss: {loss.item():.4f}, Loss1: {loss1.item():.4f}, Congdie: {isin.item():.4f}")
                    if self.config.trainer.ktl == True:
                        print(f"Loss: {loss.item():.4f}, Loss1: {loss1.item():.4f}, Loss_kt: {loss_kendall.item():.4f}, Isin: {isin}")
                        # print(f"Loss: {loss.item():.4f}, Loss1: {loss1.item():.4f}, Loss_kt: {loss_kendall.item():.4f}")
                    else:
                        print(f"Loss: {loss.item():.4f}, Loss1: {loss1.item():.4f}, Isin: {isin}")
                        # print(f"Loss: {loss.item():.4f}, Loss1: {loss1.item():.4f}")
            return loss

    def soft_clip_log(self, x: torch.Tensor, q: float = 0.95, s: float = 1.0, eps: float = 1e-12):
        """
        y = t + s * log(1 + (x - t)/s)
        s 控制压缩强度（越小压缩越强）。严格单调，保序。
        """
        t = torch.quantile(x.float(), q)
        below = x <= t
        y = torch.empty_like(x)
        y[below] = x[below]
        z = (x[~below] - t) / max(s, eps)
        y[~below] = t + s * torch.log1p(z.clamp_min(0))
        return y

    def soft_clip_log_v2(self, x: torch.Tensor, t: float, s: float = 1.0, eps: float = 1e-12):
        """
        y = t + s * log(1 + (x - t)/s)
        s 控制压缩强度（越小压缩越强）。严格单调，保序。
        """
        below = x <= t
        y = torch.empty_like(x)
        y[below] = x[below]
        z = (x[~below] - t) / max(s, eps)
        y[~below] = t + s * torch.log1p(z.clamp_min(0))
        return y   

    def normalize_is(self, x: torch.Tensor, lo=0, hi=1e-3):
        assert hi > lo
        x_dtype = x.dtype
        x = x.clone()
        mask = x > lo
        if mask.any():
            vals = x[mask].float()  # 计算用高精度
            vmin, vmax = vals.min(), vals.max()
            if (vmax - vmin) < 1e-30:
                # 全都一样时，放到区间中点
                new_vals = torch.full_like(vals, (lo + hi) / 2)
            else:
                new_vals = lo + (vals - vmin) / (vmax - vmin) * (hi - lo)
            x[mask] = new_vals.to(x_dtype)
        return x

    def standardize_shift_to_positive(self, X, target_mean=1.0, target_std=0.5, eps=1e-18):

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

    def power(self, x, lam=0.5, n=10.0):
        y = (((x) ** lam) / lam) * n
        return y

    def process_batch(self, batch):
        """
        Process a batch of data.
        """
        texts = batch["texts"]
        importance_scores = batch["importance_score"]
        # weights = batch["weight"]

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding_side="right",
            padding=True,
            add_special_tokens=False,
        )

        # if importance_scores.shape[1] != inputs.input_ids.shape[1]:
        #     import rpdb; rpdb.set_trace()

        batch_len = len(inputs.input_ids[0])

        processed_iss = []
        # processed_weights = []  
        is_mask = []
        starts, ends = [], []
        # for text, importance_score, weight in zip(texts, importance_scores, weights):
        for text, importance_score in zip(texts, importance_scores):
            if self.config.model.model_name_or_path in ["LGAI-EXAONE/EXAONE-Deep-7.8B"]:
                think_index_start, think_index_end = get_think_index(text, self.tokenizer, self.config.model.model_name_or_path)
            else:
                sub_step, type_sub_step, think_index_start, think_index_end = seq_token_info(text, self.tokenizer)
            # print(f"think_index_start: {think_index_start}, think_index_end: {think_index_end}")
            # import rpdb; rpdb.set_trace()
            # torch.distributed.barrier()

            starts.append(torch.tensor(think_index_start+1))
            ends.append(torch.tensor(think_index_end))

            # TODO: 标准化
            a = importance_score[:think_index_start+1]
            b = importance_score[think_index_start+1:think_index_end]
            c = importance_score[think_index_end:]
            
            if self.config.trainer.head_type == "classification":
                '''分类'''
                a = torch.zeros_like(a).to(torch.int64)
                thr = torch.quantile(b.float(), self.config.trainer.pruning_ratio)
                b = (b > thr).to(torch.int64)
                c = torch.zeros_like(c).to(torch.int64)
            elif self.config.trainer.head_type == "regression":
                '''回归'''

                if self.config.model.model_name_or_path in ["deepseek-ai/DeepSeek-R1-Distill-Llama-8B"]:
                    b = b * 10000
                    # b = b
                    # b = self.soft_clip_log_v2(b, t=150, s=1.0)  # 压缩
                elif self.config.model.model_name_or_path in ["Nanbeige/Nanbeige4-3B-Thinking-2511"]:
                    b = self.power(b, lam=0.3, n=1000.0)
                    # b = b
                elif self.config.model.model_name_or_path in ["LGAI-EXAONE/EXAONE-Deep-7.8B"]:
                    b = self.power(b, lam=0.3, n=10.0)
                elif self.config.model.model_name_or_path in ["deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"]:
                    b = self.power(b, lam=0.3, n=10)
                    # b = b
                else:
                    raise ValueError("Model name not recognized for regression scaling")

            processed_iss.append(torch.cat([a, b, c], dim=0)[:batch_len])

            # create mask for importance scores
            a_mask = torch.zeros_like(a)
            b_mask = torch.ones_like(b)
            c_mask = torch.zeros_like(c)
            is_mask.append(torch.cat([a_mask, b_mask, c_mask], dim=0)[:batch_len])

            # create weights for importance scores
            # processed_weights.append(weight[:batch_len])

        processed_iss = torch.stack(processed_iss, dim=0)
        # processed_weights = torch.stack(processed_weights, dim=0)
        is_mask = torch.stack(is_mask, dim=0)
        starts = torch.stack(starts, dim=0)
        ends = torch.stack(ends, dim=0)

        inputs_idxs = inputs.input_ids
        inputs_attention_mask = inputs.attention_mask
        # get position ids
        position_ids = torch.clip(torch.cumsum(inputs_attention_mask, dim=-1) - 1, min=0, max=None)

        # rank = self.device_mesh.get_rank()
        # if rank == 0:
        #     import rpdb
        #     rpdb.set_trace()
        #torch.distributed.barrier()  # 调试完通知其他继续
        if inputs_idxs.shape[1] != processed_iss.shape[1]:
            import rpdb; rpdb.set_trace()
        return {
            "input_ids": inputs_idxs,
            "attention_mask": inputs_attention_mask,
            "position_ids": position_ids,
            "importance_scores": processed_iss,
            # "weights": processed_weights,
            "is_mask": is_mask,
            "think_index_start": starts,
            "think_index_end": ends,
        }

@hydra.main(config_path="./recipes/configs", config_name="sft_trainer", version_base="1.2")
def main(config):

    print(config)

    set_seed(config.trainer.seed)

    device_name = get_device_name()
    local_rank, rank, world_size = initialize_global_process_group()

    device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(
        device_type=device_name,
        mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),
        mesh_dim_names=("dp", "sp"),
    )

    train_dataset = SFTDataset(config.data.train_files)
    val_dataset = SFTDataset(config.data.val_files)
    trainer = Trainer(
        config=config,
        device_mesh=device_mesh,
        ulysses_device_mesh=ulysses_device_mesh,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()

    destroy_global_process_group()


if __name__ == "__main__":
    main()
