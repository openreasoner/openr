import ctranslate2
from ctranslate2.converters import TransformersConverter
from typing import Optional, List
from transformers import PreTrainedModel
import os
import sentencepiece as spm


def load_ct2_model(ct2_model_path, **generator_kwargs):
    ct2_generator = ctranslate2.Generator(ct2_model_path, **generator_kwargs)
    ct2_sp = None
    # spm.SentencePieceProcessor(
    #     os.path.join(ct2_model_path, "tokenizer.model"))
    return ct2_generator, ct2_sp


class OnlineHfConverter(TransformersConverter):
    """Initializes the converter.

    Arguments:
      model_name_or_path: Name of the pretrained model to download, or path to the
        directory containing the pretrained model.
      activation_scales: Path to the pre-computed activation scales. Models may
        use them to rescale some weights to smooth the intermediate activations
        and improve the quantization accuracy. See
        https://github.com/mit-han-lab/smoothquant.
      copy_files: List of filenames to copy from the Hugging Face model to the
        converted model directory.
      load_as_float16: Load the model weights as float16. More precisely, the model
        will be loaded with ``from_pretrained(..., torch_dtype=torch.float16)``.
      revision: Revision of the model to download from the Hugging Face Hub.
      low_cpu_mem_usage: Enable the flag ``low_cpu_mem_usage`` when loading the model
        with ``from_pretrained``.
      trust_remote_code: Allow converting models using custom code.
    """

    def __init__(
        self,
        model: Optional[PreTrainedModel],
        model_name_or_path: str,
        activation_scales: Optional[str] = None,
        copy_files: Optional[List[str]] = None,
        load_as_float16: bool = False,
        revision: Optional[str] = None,
        low_cpu_mem_usage: bool = False,
        trust_remote_code: bool = False,
    ):
        super().__init__(
            model_name_or_path,
            activation_scales,
            copy_files,
            load_as_float16,
            revision,
            low_cpu_mem_usage,
            trust_remote_code,
        )
        self.model = model

    def load_model(self, model_class, model_name_or_path, **kwargs):
        if self.model is None:
            return model_class.from_pretrained(model_name_or_path, **kwargs)
        else:
            return self.model
