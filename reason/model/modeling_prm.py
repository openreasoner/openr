from collections import OrderedDict
from dataclasses import dataclass
import gc
from typing import Optional, Tuple
import torch
import torch.nn as nn
from transformers import AutoModel
import transformers
from transformers.modeling_outputs import ModelOutput
from tsllm.model.modeling_base import PreTrainedModelWrapper


@dataclass
class CategoricalHeadLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None


class CategoricalHeadedLLM(PreTrainedModelWrapper):
    _auto_model_parent_class = AutoModel
    _supported_modules = ["cat_head"]
    _supported_args = ["n_out", "loss_fn"]

    def __init__(
        self,
        base_model: AutoModel,
        n_out: int = 3,
        loss_fn: callable = nn.CrossEntropyLoss(),
        **kwargs,
    ):
        super().__init__(base_model, **kwargs)
        self.base_model = base_model
        self.n_out = n_out
        hidden_state = self.base_model.config.hidden_size

        self.cat_head = nn.Linear(hidden_state, n_out)

        self.loss_fn = loss_fn

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        labels: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        hidden_states = outputs[0]
        logits = self.cat_head(hidden_states)

        loss = None
        if labels is not None:
            labels = labels.view(-1).to(logits.device)
            # you should keep logits with shape (-1, n_class) and labels with shape [-1]
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels)

        return CategoricalHeadLMOutputWithPast(loss=loss, logits=logits)

    def gradient_checkpointing_enable(self):
        self.base_model.gradient_checkpointing_enable()

    def state_dict(self, *args, heads_only=False, **kwargs):
        """
        Returns the state dictionary of the model. We add the state dictionary of the value head
        to the state dictionary of the wrapped model by prepending the key with `cat_head.`.
        """
        cat_head_state_dict = self.cat_head.state_dict(*args, **kwargs)
        if heads_only:
            model_state_dict = OrderedDict()
        else:
            model_state_dict = self.base_model.state_dict(*args, **kwargs)

        for k, v in cat_head_state_dict.items():
            model_state_dict[f"cat_head.{k}"] = v

        return model_state_dict

    def post_init(self, state_dict):
        """
        Adds the state dictionary of the value head to the state dictionary of the wrapped model
        by prepending the key with `cat_head.`. This function removes the `cat_head.` prefix from the
        keys of the value head state dictionary.
        """
        super().post_init()

        for k in list(state_dict.keys()):
            if "cat_head." in k:
                state_dict[k.replace("cat_head.", "")] = state_dict.pop(k)
        self.cat_head.load_state_dict(state_dict, strict=False)
        del state_dict
        gc.collect()

    @classmethod
    def from_config(cls, config: transformers.PretrainedConfig, **kwargs):
        """Instantiate the pretrained pytorch model from a configuration.

        Args:
            config (transformers.PretrainedConfig): The configuration to use to
                instantiate the base model.

        NOTE: Loading a model from its configuration file does **not** load the
        model weights. It only affects the model's configuration. Use
        `~transformers.AutoModel.from_pretrained` to load the model weights.
        """
        if kwargs is not None:
            wrapped_model_kwargs, from_config_kwargs = cls._split_kwargs(kwargs)
        else:
            from_config_kwargs = {}
            wrapped_model_kwargs = {}
        base_model = cls._auto_model_parent_class.from_config(
            config, **from_config_kwargs
        )

        model = cls(base_model, **wrapped_model_kwargs)
        return model


@dataclass
class ValueHeadLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    value: torch.FloatTensor = None


class ValueHeadedLLM(PreTrainedModelWrapper):
    _auto_model_parent_class = AutoModel
    _supported_modules = ["cat_head"]
    _supported_args = ["loss_fn"]

    def __init__(
        self, base_model: AutoModel, loss_fn: callable = nn.CrossEntropyLoss(), **kwargs
    ):
        super().__init__(base_model, **kwargs)
        self.base_model = base_model
        hidden_state = self.base_model.config.hidden_size

        self.cat_head = nn.Linear(hidden_state, 1)

        self.loss_fn = loss_fn

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        labels: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        outputs = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        )

        hidden_states = outputs[0]
        value = self.cat_head(hidden_states).squeeze(dim=-1)

        loss = None

        return ValueHeadLMOutputWithPast(loss=loss, value=value)

    def gradient_checkpointing_enable(self):
        self.base_model.gradient_checkpointing_enable()

    def state_dict(self, *args, heads_only=False, **kwargs):
        """
        Returns the state dictionary of the model. We add the state dictionary of the value head
        to the state dictionary of the wrapped model by prepending the key with `cat_head.`.
        """
        cat_head_state_dict = self.cat_head.state_dict(*args, **kwargs)
        if heads_only:
            model_state_dict = OrderedDict()
        else:
            model_state_dict = self.base_model.state_dict(*args, **kwargs)

        for k, v in cat_head_state_dict.items():
            model_state_dict[f"cat_head.{k}"] = v

        return model_state_dict

    def post_init(self, state_dict):
        """
        Adds the state dictionary of the value head to the state dictionary of the wrapped model
        by prepending the key with `cat_head.`. This function removes the `cat_head.` prefix from the
        keys of the value head state dictionary.
        """
        super().post_init()

        for k in list(state_dict.keys()):
            if "cat_head." in k:
                state_dict[k.replace("cat_head.", "")] = state_dict.pop(k)
        self.cat_head.load_state_dict(state_dict, strict=False)
        del state_dict
        gc.collect()

    @classmethod
    def from_config(cls, config: transformers.PretrainedConfig, **kwargs):
        """Instantiate the pretrained pytorch model from a configuration.

        Args:
            config (transformers.PretrainedConfig): The configuration to use to
                instantiate the base model.

        NOTE: Loading a model from its configuration file does **not** load the
        model weights. It only affects the model's configuration. Use
        `~transformers.AutoModel.from_pretrained` to load the model weights.
        """
        if kwargs is not None:
            wrapped_model_kwargs, from_config_kwargs = cls._split_kwargs(kwargs)
        else:
            from_config_kwargs = {}
            wrapped_model_kwargs = {}
        base_model = cls._auto_model_parent_class.from_config(
            config, **from_config_kwargs
        )

        model = cls(base_model, **wrapped_model_kwargs)
        return model
