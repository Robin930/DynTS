import torch

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from transformers.cache_utils import Cache
from transformers.utils import ModelOutput
from transformers.generation.utils import GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput

@dataclass
class myCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    importance_scores: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None

@dataclass
class myGenerateDecoderOnlyOutput(GenerateDecoderOnlyOutput):
    importance_scores: Optional[torch.FloatTensor] = None
    purning_mask: Optional[torch.BoolTensor] = None
    decode_time: Optional[list] = None


def freeze_model(model, al):
    """
    Freeze the model parameters except for the specified layers.
    """
    for name, param in model.named_parameters():
        if al not in name:
            param.requires_grad = False
        else:
            print(f"Layer {name} is trainable.")
    return model