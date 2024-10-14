"""This should be not published, it's only a test helping function"""
import tree
import numpy as np
import sys
from reason.evaluation.evaluator import judge_ans
from reason.inference.rm_call import RMRemoteCaller, LocalCaller
# from prm.infer_fns import LocalCaller
from reason.evaluation.evaluator import Task
from reason.reranking.vote_utils import AGG_FN_MAP
from tqdm import tqdm
import jsonlines
from pathlib import Path
import copy


def main(record_path, prm_and_tags):
    with jsonlines.open(record_path, 'r') as reader:
        my_outputs = list(reader)

    task = Task('MATH')

    # ms_rm = RMRemoteCaller("math-shepherd-mistral-7b-prm", step_tag='ки\n')
    # qw_rm = RMRemoteCaller("Qwen2.5-Math-7B-PRM", step_tag='\n\n\n\n\n')

    prm_new_outputs_dict = {}
    prm_result_collect = {}
    for prm_name, step_tag in prm_and_tags:
        rm_call = LocalCaller(prm_name, step_tag=step_tag)
        res_list = []
        orig_res_list = []
        new_output_list = []
        for output in tqdm(my_outputs):
            new_output = copy.deepcopy(output)
            question = output['question']
            groundtruth = output['groundtruth']

            preds = []
            original_v = []

            format_str = "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}"

            for x in output['output']:
                preds.append(x['text'])
                original_v.append(x['value'])
            
            rm_inputs = [format_str.format(question=question, answer=a) for a in preds]
            # print(rm_inputs[0])
            # break
            rm_v = rm_call(rm_inputs)
            for n_out, new_v in zip(new_output['output'], rm_v):
                n_out['value'] = new_v
            assert len(rm_v) == len(original_v)
            res = {
                k: 
                judge_ans(
                    question,
                    task.extract_groundtruth(groundtruth),
                    preds,
                    rm_v,
                    k,
                    task.extract_answer,
                    task.judge_correct
                )
                for k in AGG_FN_MAP.keys() if 'orm' not in k
            }
            orig_res = {
                k: 
                judge_ans(
                    question,
                    task.extract_groundtruth(groundtruth),
                    preds,
                    original_v,
                    k,
                    task.extract_answer,
                    task.judge_correct
                )
                for k in AGG_FN_MAP.keys() if 'orm' not in k
            }
            res_list.append(res)
            orig_res_list.append(orig_res)
            if 'other_prm' not in output:
                output['other_prm'] = {}
            output['other_prm'][prm_name] = {
                'value': rm_v,
                'result': res
            }
            new_output_list.append(new_output)
        prm_result_collect[prm_name] = tree.map_structure(lambda *xs: np.mean(xs), *res_list)
        orig_res = tree.map_structure(lambda *xs: np.mean(xs), *orig_res_list)
        # print(orig_res)
        print("PRM {}: {}\nORIG: {}".format(prm_name, prm_result_collect[prm_name], orig_res))
        prm_new_outputs_dict[prm_name] = new_output_list


    print(prm_result_collect)

    for prm_name, new_objs in prm_new_outputs_dict.items():
        with jsonlines.open(f"new_record_{prm_name.split('/')[-1]}.jsonl", 'w') as f:
            f.write_all(new_objs)

if __name__ == "__main__":
    # HOW TO USE
    # First deploy your PRMs and then set `step_tags` here
    # You can also modify the code to run the prm locally.
    # prm_and_tags = [
    #     # ("math-shepherd-mistral-7b-prm", 'ки\n'),
    #     ("/mnt/nfs1/workspace/o1-dev/prm/prm_results_qwen_new.31/bs_256_lr_0.0001_datasets_aps_shepherd/checkpoint-2331", '\n\n\n\n\n'),
    # ]
    prm_and_tags = [
        # ("math-shepherd-mistral-7b-prm", 'ки\n'),
        ("/mnt/nfs1/workspace/o1-dev/prm/prm_results_qwen_new.31/bs_256_lr_0.0001_datasets_all/checkpoint-2127", '\n\n\n\n\n'),
    ]
    # prm_and_tags = [
    #     # ("math-shepherd-mistral-7b-prm", 'ки\n'),
    #     ("/mnt/nfs1/workspace/o1-dev/prm/prm_results_qwen_new.1/bs_256_lr_0.0001_datasets_math_aps/checkpoint-595", '\n\n\n\n\n'),
    # ]

    main(Path(__file__).parent / 'record.jsonl', prm_and_tags)