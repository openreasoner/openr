from collections import OrderedDict
from dataclasses import dataclass
import gc
from typing import List, Optional, Tuple, Union
import torch
from torch import nn
import transformers
from transformers.modeling_outputs import ModelOutput
from model.modeling_base import PreTrainedModelWrapper
from model.utils import findattr
from peft import PeftConfig


def make_head(n_embd: int, out: int, dtype: type = torch.float32) -> nn.Sequential:
    """Returns a generic sequential MLP head."""
    return nn.Sequential(
        nn.Linear(n_embd, n_embd * 2, dtype=dtype),
        nn.ReLU(),
        nn.Linear(n_embd * 2, out, dtype=dtype),
    )


def hf_get_hidden_size(config: transformers.PretrainedConfig) -> int:
    """Returns the hidden layer dimensionality of the model architecture specified
    by the HuggingFace transformers config.
    NOTE: Different model configurations have different hidden size attribute names.
        - hidden_size: (OPTConfig, BloomConfig)
        - n_embd: (GPT2Config, GPTJConfig)
        - d_model: (PegasusConfig, XLNetConfig)
    """
    hidden_size_attrs = ("hidden_size", "n_embd", "d_model")
    return findattr(config, hidden_size_attrs)


@dataclass
class CausalLMOutputWithValue(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    value: Optional[torch.FloatTensor] = None


class AutoModelForCausalLMWithValueHead(PreTrainedModelWrapper):
    """An `AutoModel` class wrapper for `transformers` causal models that have a
    language modeling head and a value head
    """

    _auto_model_parent_class = transformers.AutoModelForCausalLM
    _supported_modules = ["v_head", "peft_config"]

    def __init__(
        self,
        base_model: transformers.PreTrainedModel,
        peft_config: Optional[PeftConfig] = None,
    ):
        super().__init__(base_model, peft_config=peft_config)
        self.v_head = make_head(hf_get_hidden_size(self.base_model.config), 1)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        position_ids: Optional[List[torch.FloatTensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithValue]:
        # # set use cache = False to use flash-attn
        if use_cache is None:
            use_cache = False

        forward_kwargs = self.get_compatible_forward_kwargs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        forward_kwargs["output_hidden_states"] = True
        forward_kwargs["return_dict"] = True

        # # set use cache = False to use flash-attn
        # assert not use_cache, ("To use flast-attn, you should now use cache", use_cache)
        # forward_kwargs["use_cache"] = False

        # outputs = self.base_model(**forward_kwargs)
        # value = self.v_head(outputs.hidden_states[-1]).squeeze(-1)
        # if not return_dict:
        #     outputs = (outputs.logits,) + outputs[1:] + (value,)
        #     return outputs

        # return CausalLMOutputWithValue(**outputs, value=value)

        forward_kwargs.pop("labels")
        transformer_outputs = self.base_model.transformer(**forward_kwargs)
        # hidden_states = transformer_outputs[0]
        value = self.v_head(transformer_outputs.hidden_states[-1]).squeeze(-1)
        return CausalLMOutputWithValue(value=value)

    def generate(self, *args, **kwargs) -> Union[ModelOutput, torch.LongTensor]:
        return self.base_model.generate(*args, **kwargs)

    def state_dict(self, *args, heads_only=False, **kwargs):
        """
        Returns the state dictionary of the model. We add the state dictionary of the value head
        to the state dictionary of the wrapped model by prepending the key with `v_head.`.
        """
        v_head_state_dict = self.v_head.state_dict(*args, **kwargs)
        if heads_only:
            model_state_dict = OrderedDict()
        else:
            model_state_dict = self.base_model.state_dict(*args, **kwargs)

        for k, v in v_head_state_dict.items():
            model_state_dict[f"v_head.{k}"] = v

        return model_state_dict

    def post_init(self, state_dict):
        """
        Adds the state dictionary of the value head to the state dictionary of the wrapped model
        by prepending the key with `v_head.`. This function removes the `v_head.` prefix from the
        keys of the value head state dictionary.
        """
        super().post_init()

        for k in list(state_dict.keys()):
            if "v_head." in k:
                state_dict[k.replace("v_head.", "")] = state_dict.pop(k)
        self.v_head.load_state_dict(state_dict, strict=False)
        del state_dict
        gc.collect()  # noqa: E702

    def gradient_checkpointing_enable(self):
        self.base_model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.base_model.gradient_checkpointing_disable()
