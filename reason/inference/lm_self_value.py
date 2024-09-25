import torch
from typing import Union, List
from tsllm.model import ValueHeadedLLM
from transformers import AutoTokenizer
import re
import numpy as np


@torch.inference_mode()
def tot_value_fn(
    critic: ValueHeadedLLM,
    tokenizer: AutoTokenizer,
    env_name: str,
    input_str: Union[List[str], str],
):
    if env_name == "game24":
        from envs.game24.prompt import VALUE_PROMPT, VALUE_LAST_STEP_PROMPT
    else:
        print("tot_value_fn does not support env {}.".format(env_name))
        raise NotImplementedError

    token_batch = []
    for text in input_str:
        last_line = text.strip().split("\n")[-1]
        if "left" in last_line:
            current_numbers = last_line.split("left: ")[-1].split(")")[0]
            prompt = VALUE_PROMPT.format(input=current_numbers)
        else:
            inp = text.strip().split("\n")[1].replace("Input: ", "")
            ans = last_line.lower().replace("The answer is: ", "")
            prompt = VALUE_LAST_STEP_PROMPT.format(input=inp, answer=ans)
        prompt_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(prompt))
        token_batch.append(prompt_tokens)

    step_results = critic.generate_batch(
        token_batch,
        sampling_temperature=1.0,
        sampling_topp=1.0,
        sampling_topk=100,
        max_length=128,
        return_scores=True,
        include_prompt_in_result=False,
        end_token=[tokenizer.eos_token_id],
        static_prompt=None,
        max_batch_size=0,
        num_hypotheses=3,  # it is the n_evaluate_sample in tot
    )

    values = []
    value_map = {"impossible": 0.001, "likely": 1, "sure": 20}
    for res in list(step_results):
        v_res = []
        for seq in res.sequences_ids:
            text = tokenizer.decode(seq)
            matchs = re.findall(r"\bimpossible|sure|likely\b", text)
            if len(matchs) > 0:
                # it will generate too much imagination, thus we chose only the first one.
                v_seq = value_map[matchs[0]]
            else:
                # default to likely
                v_seq = value_map["likely"]
            v_res.append(v_seq)
        values.append(np.mean(v_res))

    return np.array(values)
