
import torch

from torch import nn, optim

from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from transformers.models.phi3.modeling_phi3 import Phi3ForCausalLM, Phi3Model, Phi3PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "ByteDance-Seed/Seed-Coder-8B-Reasoning"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

import pdb; pdb.set_trace()