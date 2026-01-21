
import torch

from torch import nn, optim

from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, Qwen2PreTrainedModel, Qwen2ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3Model
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from models.utils import myCausalLMOutputWithPast
from models.is_head import ISMLP

@auto_docstring
class myQwen2ForCausalLM(Qwen2ForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}


    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if kwargs.get("head_type") == "classification":
            self.is_head = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size * 2, bias=False),
                                    nn.Softplus(),
                                    nn.Linear(config.hidden_size * 2, config.hidden_size // 2, bias=False),
                                    nn.Softplus(),
                                    nn.Linear(config.hidden_size // 2, 2, bias=False))
        elif kwargs.get("head_type") == "regression":
            self.is_head = nn.Sequential(nn.Linear(config.hidden_size, 4096 * 2, bias=False),
                                    nn.Softplus(),
                                    nn.Linear(4096 * 2, 4096//2, bias=False),
                                    nn.Softplus(),
                                    nn.Linear(4096 // 2, 1, bias=False),
                                    nn.Softplus())
            # self.is_head = nn.Sequential(nn.Linear(config.hidden_size, 4096 * 2, bias=False),
            #                         nn.SiLU(),
            #                         nn.Linear(4096 * 2, 4096 // 2, bias=False),
            #                         nn.SiLU(),
            #                         nn.Linear(4096 // 2, 1, bias=False),
            #                         nn.Softplus()
            #                         )

        else:
            raise ValueError("head_type must be 'classification' or 'regression'")

        self.post_init()


    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

        >>> model = Qwen2ForCausalLM.from_pretrained("meta-qwen2/Qwen2-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-qwen2/Qwen2-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        
        importance_scores = self.is_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return myCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            importance_scores=importance_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class myQwen3ForCausalLM(Qwen3ForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)


        if kwargs.get("head_type") == "classification":
            self.is_head = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size * 2, bias=False),
                                    nn.Softplus(),
                                    nn.Linear(config.hidden_size * 2, config.hidden_size // 2, bias=False),
                                    nn.Softplus(),
                                    nn.Linear(config.hidden_size // 2, 2, bias=False))
        elif kwargs.get("head_type") == "regression":
            self.is_head = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size * 2, bias=False),
                                    nn.Softplus(),
                                    nn.Linear(config.hidden_size * 2, config.hidden_size // 2, bias=False),
                                    nn.Softplus(),
                                    nn.Linear(config.hidden_size // 2, 1, bias=False),
                                    nn.Softplus())
        else:
            raise ValueError("head_type must be 'classification' or 'regression'")

        # Initialize weights and apply final processing
        self.post_init()

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

        >>> model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        importance_scores = self.is_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return myCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            importance_scores=importance_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


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