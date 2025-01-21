"""
A model worker that executes the model.
"""

import argparse
import base64
import gc
import json
import os
from binascii import Error
from typing import List, Optional
import uuid

import numpy as np
import torch
import torch.nn.functional as F
from click import prompt
from sympy.physics.units import temperature
from transformers import set_seed
import uvicorn

from fastchat.constants import ErrorCode, SERVER_ERROR_MSG
from fastchat.model.model_adapter import (
    load_model,
    add_model_args,
    get_generate_stream_function,
)
from fastchat.modules.awq import AWQConfig
from fastchat.modules.exllama import ExllamaConfig
from fastchat.modules.xfastertransformer import XftConfig
from fastchat.modules.gptq import GptqConfig

from prm.code.evaluate import output
from reason.llm_service.workers.base_model_worker import BaseModelWorker, app
from fastchat.utils import (
    build_logger,
    get_context_length,
    str_to_torch_dtype,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList, StoppingCriteria

worker_id = str(uuid.uuid4())[:8]
logger = build_logger("model_worker", f"model_worker_{worker_id}.log")


class HuggingfaceWorker(BaseModelWorker):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        model_names: List[str],
        limit_worker_concurrency: int,
        no_register: bool,
        device: str,
        num_gpus: int,
        max_gpu_memory: str,
        revision: str = None,
        dtype: Optional[torch.dtype] = None,
        load_8bit: bool = False,
        cpu_offloading: bool = False,
        gptq_config: Optional[GptqConfig] = None,
        awq_config: Optional[AWQConfig] = None,
        exllama_config: Optional[ExllamaConfig] = None,
        xft_config: Optional[XftConfig] = None,
        stream_interval: int = 2,
        conv_template: Optional[str] = None,
        embed_in_truncate: bool = False,
        seed: Optional[int] = None,
        debug: bool = False,
        **kwargs,
    ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            model_path,
            model_names,
            limit_worker_concurrency,
            conv_template=conv_template,
        )

        logger.info(f"Loading the model {self.model_names} on worker {worker_id} ...")
        self.model = AutoModelForCausalLM.from_pretrained(model_path,
                                                          device_map=device,
                                                          dtype=dtype,
                                                          trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.device = device
        if self.tokenizer.pad_token == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.context_len = get_context_length(self.model.config)
        # self.generate_stream_func = get_generate_stream_function(self.model, model_path)
        print("Use `generate_stream` only for now")
        from reason.llm_service.workers.inference import generate_stream

        self.generate_stream_func = generate_stream
        self.stream_interval = stream_interval
        self.embed_in_truncate = embed_in_truncate
        self.seed = seed

        if not no_register:
            self.init_heart_beat()

    def generate_stream_gate(self, params):
        """
        Vllm Caller Api
        """
        self.call_ct += 1

        params["logprobs"] = 1

        try:
            if self.seed is not None:
                set_seed(self.seed)

            promt = params["prompt"]
            n = params.get("n", 1)
            temperature = params.get('temperature', 0.)
            top_p = params.get('top_p', 1.)
            top_k = int(params.get('top_k', -1))
            max_new_tokens = int(params.get("max_new_tokens", 256))
            stop_str = params.get("stop", [])
            output_score = params.get("output_scores", False)

            if self.tokenizer.eos_token not in stop_str:
                stop_str += [self.tokenizer.eos_token]

            model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            if output_score:
                gen_cfg = {
                    "return_dict_in_generate": True,
                    "output_scores": True
                }
            else:
                gen_cfg = {}

            outputs = self.model.generate(**model_inputs,
                                          **gen_cfg,
                                          do_sample=(temperature>1e-5),
                                          temperature=temperature,
                                          top_p=top_p,
                                          top_k=None if top_k==-1 else top_k,
                                          max_new_tokens=max_new_tokens,
                                          num_return_sequences=n,
                                          stop_strings=stop_str,
                                          tokenizer=self.tokenizer)

            if output_score:
                generated_ids = [outputs.sequences[i][len(model_inputs.input_ids[0]):] for i in range(n)]
                finish_reason = ["length" if len(i) >= max_new_tokens else "stop" for i in generated_ids]
                scores = outputs.scores
                log_probs = []
                for i, logits in enumerate(scores):
                    token_id = generated_ids[0][i]
                    log_prob = F.log_softmax(logits[0], dim=-1)[token_id].item()
                    if not np.isnan(log_prob):
                        log_probs.append(log_prob)
            else:
                generated_ids = [outputs[i][len(model_inputs.input_ids[0]):] for i in range(n)]
                finish_reason = ['length' if len(i) >= max_new_tokens else 'stop' for i in generated_ids]
                log_probs = [1.0] * len(generated_ids)

            generated_text = self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )

            ret = {
                "text": generated_text,
                "error_code": 0,
                "finish_reason": finish_reason,
                "usage": {"prompt_tokens": [len(i) for i in model_inputs.input_ids],
                          "completion_tokens": [len(i) for i in generated_ids]},
                "logprobs": {"token_logprobs": log_probs},
                "output_token_len": [len(i) for i in generated_ids],
                "cumulative_logprob": [sum(log_probs)]
            }

            return json.dumps(ret).encode() + b"\0"

        except torch.cuda.OutOfMemoryError as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
            yield json.dumps(ret).encode() + b"\0"
        except (ValueError, RuntimeError) as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
            yield json.dumps(ret).encode() + b"\0"
        except Exception as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR
            }

    def generate_gate(self, params):
        x = self.generate_stream_gate(params)

        return json.loads(x[:-1].decode())

    def __process_embed_chunk(self, input_ids, attention_mask, **model_type_dict):
        if model_type_dict.get("is_bert"):
            model_output = self.model(input_ids)
            if model_type_dict.get("is_robert"):
                data = model_output.last_hidden_state
            else:
                data = model_output[0]
        elif model_type_dict.get("is_t5"):
            model_output = self.model(input_ids, decoder_input_ids=input_ids)
            data = model_output.encoder_last_hidden_state
        else:
            model_output = self.model(input_ids, output_hidden_states=True)
            if model_type_dict.get("is_chatglm"):
                data = model_output.hidden_states[-1].transpose(0, 1)
            else:
                data = model_output.hidden_states[-1]

        if hasattr(self.model, "use_cls_pooling") and self.model.use_cls_pooling:
            sum_embeddings = data[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1).expand(data.size()).float()
            masked_embeddings = data * mask
            sum_embeddings = torch.sum(masked_embeddings, dim=1)
        token_num = torch.sum(attention_mask).item()

        return sum_embeddings, token_num

    def __encode_base64(self, embeddings: torch.Tensor) -> List[str]:
        embeddings = embeddings.cpu()
        return [
            base64.b64encode(e.numpy().tobytes()).decode("utf-8") for e in embeddings
        ]

    @torch.inference_mode()
    def get_embeddings(self, params):
        self.call_ct += 1

        try:
            tokenizer = self.tokenizer
            ret = {"embedding": [], "token_num": 0}

            model_type_dict = {
                "is_llama": "llama" in str(type(self.model)),
                "is_t5": "t5" in str(type(self.model)),
                "is_chatglm": "chatglm" in str(type(self.model)),
                "is_bert": "bert" in str(type(self.model)),
                "is_robert": "robert" in str(type(self.model)),
            }

            if self.embed_in_truncate:
                encoding = tokenizer.batch_encode_plus(
                    params["input"],
                    padding=True,
                    truncation="longest_first",
                    return_tensors="pt",
                    max_length=self.context_len,
                )
            else:
                encoding = tokenizer.batch_encode_plus(
                    params["input"], padding=True, return_tensors="pt"
                )
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = input_ids != tokenizer.pad_token_id

            base64_encode = params.get("encoding_format", None)

            if self.embed_in_truncate:
                embedding, token_num = self.__process_embed_chunk(
                    input_ids, attention_mask, **model_type_dict
                )
                if (
                    not hasattr(self.model, "use_cls_pooling")
                    or not self.model.use_cls_pooling
                ):
                    embedding = embedding / token_num
                normalized_embeddings = F.normalize(embedding, p=2, dim=1)
                ret["token_num"] = token_num
            else:
                all_embeddings = []
                all_token_num = 0
                for i in range(0, input_ids.size(1), self.context_len):
                    chunk_input_ids = input_ids[:, i : i + self.context_len]
                    chunk_attention_mask = attention_mask[:, i : i + self.context_len]

                    # add cls token and mask to get cls embedding
                    if (
                        hasattr(self.model, "use_cls_pooling")
                        and self.model.use_cls_pooling
                    ):
                        cls_tokens = (
                            torch.zeros(
                                (chunk_input_ids.size(0), 1),
                                dtype=chunk_input_ids.dtype,
                                device=chunk_input_ids.device,
                            )
                            + tokenizer.cls_token_id
                        )
                        chunk_input_ids = torch.cat(
                            [cls_tokens, chunk_input_ids], dim=-1
                        )
                        mask = torch.ones(
                            (chunk_attention_mask.size(0), 1),
                            dtype=chunk_attention_mask.dtype,
                            device=chunk_attention_mask.device,
                        )
                        chunk_attention_mask = torch.cat(
                            [mask, chunk_attention_mask], dim=-1
                        )

                    chunk_embeddings, token_num = self.__process_embed_chunk(
                        chunk_input_ids, chunk_attention_mask, **model_type_dict
                    )
                    if (
                        hasattr(self.model, "use_cls_pooling")
                        and self.model.use_cls_pooling
                    ):
                        all_embeddings.append(chunk_embeddings * token_num)
                    else:
                        all_embeddings.append(chunk_embeddings)
                    all_token_num += token_num

                all_embeddings_tensor = torch.stack(all_embeddings)
                embedding = torch.sum(all_embeddings_tensor, dim=0) / all_token_num
                normalized_embeddings = F.normalize(embedding, p=2, dim=1)

                ret["token_num"] = all_token_num

            if base64_encode == "base64":
                out_embeddings = self.__encode_base64(normalized_embeddings)
            else:
                out_embeddings = normalized_embeddings.tolist()
            ret["embedding"] = out_embeddings

            gc.collect()
            torch.cuda.empty_cache()
            if self.device == "xpu":
                torch.xpu.empty_cache()
            if self.device == "npu":
                torch.npu.empty_cache()
        except torch.cuda.OutOfMemoryError as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
        except (ValueError, RuntimeError) as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
        return ret


def create_model_worker():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    add_model_args(parser)
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument("--embed-in-truncate", action="store_true")
    parser.add_argument(
        "--limit-worker-concurrency",
        type=int,
        default=5,
        help="Limit the model concurrency to prevent OOM.",
    )
    parser.add_argument("--stream-interval", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Overwrite the random seed for each generation.",
    )
    parser.add_argument(
        "--debug", type=bool, default=False, help="Print debugging messages"
    )
    parser.add_argument(
        "--ssl",
        action="store_true",
        required=False,
        default=False,
        help="Enable SSL. Requires OS Environment variables 'SSL_KEYFILE' and 'SSL_CERTFILE'.",
    )
    args = parser.parse_args()
    logger.info(f"args: {args}")

    if args.gpus:
        if len(args.gpus.split(",")) < args.num_gpus:
            raise ValueError(
                f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    gptq_config = GptqConfig(
        ckpt=args.gptq_ckpt or args.model_path,
        wbits=args.gptq_wbits,
        groupsize=args.gptq_groupsize,
        act_order=args.gptq_act_order,
    )
    awq_config = AWQConfig(
        ckpt=args.awq_ckpt or args.model_path,
        wbits=args.awq_wbits,
        groupsize=args.awq_groupsize,
    )
    if args.enable_exllama:
        exllama_config = ExllamaConfig(
            max_seq_len=args.exllama_max_seq_len,
            gpu_split=args.exllama_gpu_split,
            cache_8bit=args.exllama_cache_8bit,
        )
    else:
        exllama_config = None
    if args.enable_xft:
        xft_config = XftConfig(
            max_seq_len=args.xft_max_seq_len,
            data_type=args.xft_dtype,
        )
        if args.device != "cpu":
            print("xFasterTransformer now is only support CPUs. Reset device to CPU")
            args.device = "cpu"
    else:
        xft_config = None

    worker = HuggingfaceWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.model_path,
        args.model_names,
        args.limit_worker_concurrency,
        revision=args.revision,
        no_register=args.no_register,
        device=args.device,
        num_gpus=args.num_gpus,
        max_gpu_memory=args.max_gpu_memory,
        dtype=str_to_torch_dtype(args.dtype),
        load_8bit=args.load_8bit,
        cpu_offloading=args.cpu_offloading,
        gptq_config=gptq_config,
        awq_config=awq_config,
        exllama_config=exllama_config,
        xft_config=xft_config,
        stream_interval=args.stream_interval,
        conv_template=args.conv_template,
        embed_in_truncate=args.embed_in_truncate,
        seed=args.seed,
        debug=args.debug,
    )
    return args, worker


if __name__ == "__main__":
    args, worker = create_model_worker()
    if args.ssl:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info",
            ssl_keyfile=os.environ["SSL_KEYFILE"],
            ssl_certfile=os.environ["SSL_CERTFILE"],
        )
    else:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
