from pathlib import Path
from typing import Callable
import ctranslate2
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import sentencepiece as spm
import os
from tqdm import tqdm
import argparse
import jsonlines
from tsllm.distributed.utils import print_rank_0
from tsllm.llm.text_generation import llm_gen_ct2
from tsllm.llm.ct2_utils import OnlineHfConverter
from tsllm.envs import get_env_datasets, get_default_query_str_builder, get_env_answer_checker
from tsllm.argparse_utils import list_of_ints, str2bool
from importlib import import_module


def _cot_gen(
    ct2_generator,
    tokenizer,
    query_str_build_fn: Callable,
    problem,
    n=100,
    stop=2,
    max_new_tokens=256,
    **kwargs,
):
    # prompt = "Question: " + problem["question"] + "\nAnswer: Let's think step by step\n"
    # if use_prefix:
    #     prompt = prefix + "\n" + prompt
    prompt = query_str_build_fn(problem["question"])
    texts, logps = llm_gen_ct2(
        ct2_generator,
        tokenizer,
        static_prompt=None,
        prompt=prompt,
        num_sequence=n,
        stop=stop,
        max_new_tokens=max_new_tokens,
        **kwargs,
    )

    return texts


"""convert hf model to ct2 model"""
# base_llm_dir = "huggyllama/llama-7b"
# cache_dir =
# ct2_dir =


# base_llm = AutoModelForCausalLM.from_pretrained(base_llm_dir,
#                                                 cache_dir=cache_dir)
# tokenizer = AutoTokenizer.from_pretrained(base_llm_dir, cache_dir=cache_dir)
# cvt = OnlineHfConverter(model=base_llm,
#                         model_name_or_path=base_llm_dir,
#                         copy_files=["tokenizer.model"])
# cvt.convert(ct2_dir, force=True, quantization="bfloat16")


def check_answers(check_fn, problem_inst, texts):
    # groundtruth = extract_groundtruth(problem_inst["answer"])
    write_obj = {
        "question": problem_inst["question"],
        "groundtruth": problem_inst["answer"],
    }
    ans_list = []

    cnt = 0
    for txt in texts:
        correct = check_fn(problem_inst["question"], problem_inst["answer"], txt)
        if correct:
            cnt += 1
        ans_list.append({"text": txt, "correct": correct})

    write_obj["answer"] = ans_list

    return write_obj, cnt, len(texts)


def main(args):

    train_ds, test_ds = get_env_datasets(args.env_name)
    if args.test:
        ds = test_ds
    else:
        ds = train_ds

    args.output_path = Path(args.output_path)
    if not args.output_path.parent.exists():
        args.output_path.parent.mkdir(parents=True)

    ct2_generator = ctranslate2.Generator(
        args.ct2_dir, device="cuda", device_index=args.gpu_ids, compute_type="bfloat16"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    print(
        "LOADING CT2 MODEL AT: {}. DEVICES={}.\nOUTPUT_DIR: {}".format(
            args.ct2_dir, args.gpu_ids, args.output_path
        )
    )

    query_str_build_fn = partial(
        get_default_query_str_builder(args.env_name), is_few_shot=args.is_few_shot
    )
    cot_gen = partial(
        _cot_gen,
        ct2_generator,
        tokenizer,
        query_str_build_fn,
        n=args.k,
        stop=tokenizer.eos_token_id,  # llama tokenizer: 2 is eos_token_id
        max_new_tokens=256,
        temperature=args.t,
        top_p=1,
        top_k=100,
        max_batch_size=args.max_batch_size,
    )

    checker_fn = get_env_answer_checker(args.env_name)
    correct_num, total_num = 0, 0
    with ThreadPoolExecutor(args.num_workers) as pool:
        results = pool.map(cot_gen, ds)
        with jsonlines.open(args.output_path, "w") as writer:
            for i, txts in enumerate(pbar := tqdm(results, total=len(ds))):
                write_obj, cnt, len_list = check_answers(checker_fn, ds[i], txts)
                writer.write(write_obj)
                correct_num += cnt
                total_num += len_list
                pbar.set_description(
                    "{}-corrent: {:.3%}[{}/{}]".format(
                        i + 1, correct_num / total_num, correct_num, total_num
                    )
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # env/task config
    parser.add_argument("--is_few_shot", type=str2bool, default=False)
    parser.add_argument("--test", type=str2bool, default=False)
    parser.add_argument("--env_name", type=str, required=True)

    parser.add_argument("-k", type=int, default=1)
    parser.add_argument("-t", type=float, default=0.7)

    parser.add_argument("--ct2_dir", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    parser.add_argument(
        "--gpu_ids", type=list_of_ints, default=[0, 1, 2, 3, 4, 5, 6, 7]
    )
    parser.add_argument("--max_batch_size", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()
    main(args)
