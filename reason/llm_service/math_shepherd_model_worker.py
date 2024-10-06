"""
A model worker that executes the model.
"""
import argparse
import base64
import gc
import json
import os
from typing import List, Optional
import uuid

import torch
import torch.nn.functional as F
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
from reason.llm_service.base_model_worker import BaseModelWorker, app
from fastchat.utils import (
    build_logger,
    get_context_length,
    str_to_torch_dtype,
)


worker_id = str(uuid.uuid4())[:8]
logger = build_logger("math_shepherd_model_worker", f"math_shepherd_model_worker_{worker_id}.log")

# Math-Shepherd Model Defined
GOOD_TOKEN = '+'
BAD_TOKEN = '-'
STEP_TAG = 'ки'

class ModelWorker(BaseModelWorker):
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
        self.model, self.tokenizer = load_model(
            model_path,
            device=device,
            num_gpus=num_gpus,
            max_gpu_memory=max_gpu_memory,
            dtype=dtype,
            load_8bit=load_8bit,
            cpu_offloading=cpu_offloading,
            gptq_config=gptq_config,
            awq_config=awq_config,
            exllama_config=exllama_config,
            xft_config=xft_config,
            debug=debug,
        )
        self.device = device
        if self.tokenizer.pad_token == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.context_len = get_context_length(self.model.config)
        self.generate_stream_func = get_generate_stream_function(self.model, model_path)
        self.stream_interval = stream_interval
        self.embed_in_truncate = embed_in_truncate
        self.seed = seed

        if not no_register:
            self.init_heart_beat()

    @torch.inference_mode()
    def value_inference_gate(self, params):
        candidate_tokens = self.tokenizer.encode(f"{GOOD_TOKEN} {BAD_TOKEN}")[1:] # [648, 387]
        step_tag_id = self.tokenizer.encode(f"{STEP_TAG}")[-1] # 12902

        def _math_shepherd_infer_fn(input_str: str):
            input_id = torch.tensor(
                [self.tokenizer.encode(input_str)], device=self.device)
            logits = self.model(input_id).logits[:,:,candidate_tokens]
            scores = logits.softmax(dim=-1)[:,:,0] 
            step_scores = scores[input_id == step_tag_id]
            return step_scores
        def _batch_math_shepherd_infer_fn(input_strs: List[str], batch_size: int):
            ret_scores = []
            for i in range(0, len(input_strs), batch_size):
                inputs = self.tokenizer(
                    input_strs[
                        i*batch_size:
                        min((i+1)*batch_size, len(input_strs))
                    ], 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True).to(self.device)
                logits = self.model(**inputs).logits[:,:,candidate_tokens]
                scores = logits.softmax(dim=-1)[:,:,0]

                for i, s in enumerate(inputs["input_ids"]):
                    step_scores = scores[i, s == step_tag_id]
                    ret_scores.append(step_scores.tolist())
            return ret_scores

        input_str = params["input_str"]
        try:
            if isinstance(input_str, list):
                # value_2 = _batch_math_shepherd_infer_fn(
                #     input_str, 
                #     batch_size=16
                # )
                # value = value_2

                value = [
                    _math_shepherd_infer_fn(s).tolist() for s in input_str
                ]
                # # verify two values
                # for v1, v2 in zip(value, value_2):
                #     assert torch.allclose(
                #         torch.tensor(v1), torch.tensor(v2), 1e-6), [v1, v2]
            else:
                value = _math_shepherd_infer_fn(input_str).tolist()
            ret = {"input": input_str, "value": value} 
            gc.collect()
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
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

    worker = ModelWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.model_path,
        args.model_names,
        args.limit_worker_concurrency,
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
