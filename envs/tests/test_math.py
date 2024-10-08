from envs.MATH import (
    Env,
    get_train_test_dataset,
    extract_groundtruth,
    judge_correct,
    extract_answer,
)

if __name__ == "__main__":
    from transformers import AutoTokenizer
    from reason.evaluation.evaluator import MathEvaluator, Task
    from reason.inference.lm_call import VLLMRemoteCaller
    from reason.inference.rm_call import DummyRewardModelCaller
    from reason.evaluation.methods import cot
    import jsonlines
    import tree
    import numpy as np
    from tqdm import tqdm
    _, ds = get_train_test_dataset()
    print(len(ds))
    print(ds[0])
    problem = ds[0]
    env = Env(
        config={
            "max_actions": 10,
            "max_length": 10,
            "stop_str": "The answer is ",
            "generation_config": {
                "max_new_tokens": 512,
                "temperature": 0.3,
                "top_p": 1.0,
                "top_k": 50,
            },
        },
        math_problems=[
            {
                "question": problem["question"],
                "answer": extract_groundtruth(problem["answer"]),
            }
        ],
        llm_gen_fn=VLLMRemoteCaller("Qwen2.5-Math-7B-Instruct", "http://0.0.0.0:28777"),
        reset=False,
    )

    env.reset(update_legal_action=True)
    print(env.legal_actions)
    env.step(action=env.legal_actions[0]["action"])
    print(env.get_state())
    print(env.question)
    print(env.answer)

    task = Task("MATH")
    evaluator = MathEvaluator("MATH",
                              VLLMRemoteCaller("Qwen2.5-Math-7B-Instruct", 'http://0.0.0.0:28777'),
                              DummyRewardModelCaller())
    
    # res = env.step(f"The answer is: {problem['answer']} ки", update_legal_action=False)
    # print(res)
    # print(env.get_state())

    # a = "Step 1: To convert from rectangular coordinates to polar coordinates, we can use the formulas:\n$r = \\sqrt{x^2 + y^2}$ and $\\theta = \\arctan\\left(\\frac{y}{x}\\right)$. ки\nStep 2: In this case, $x = 0$ and $y = 3$. ки\nStep 3: Plugging these values into the formulas, we get:\n$r = \\sqrt{0^2 + 3^2} = 3$ and $\\theta = \\arctan\\left(\\frac{3}{0}\\right)$. ки\nStep 4: Since $\\frac{3}{0}$ is undefined, we need to find an alternative way to find $\\theta$. ки\nStep 5: We know that the point $(0,3)$ lies on the positive y-axis, so $\\theta = \\frac{\\pi}{2}$. ки\nStep 6: Therefore, the polar coordinates of the point $(0,3)$ are $\\boxed{(3,\\frac{\\pi}{2})}$. The answer is: (3,\\frac{\\pi}{2}) ки"
    # g = "\\left( 3, \\frac{\\pi}{2} \\right)"
    # print(judge_correct(None, g, extract_answer(a)))
    test_ds = task.test_ds
    with jsonlines.open('/hpc2ssd/JH_DATA/spooler/qxiao183/workspace/ziyu/open_reasoner/Qwen2.5-Math/evaluation/outputs/math_eval/math/test500_qwen25-math-cot_-1_seed0_t0.0_s0_e-1.jsonl', 'r') as f:
        res = []
        for outputs in tqdm(f):
            
            problem_inst = {"question": outputs['question'], "answer": outputs['solution']}
            res.append(evaluator.analyze_output(problem_inst, outputs['code'])[0])
            if res[-1]['majority_vote'] != outputs['score'][0]:
                print(res[-1], outputs['score'][0])
            # print(res[-1], outputs['score'][0])
    
    print(tree.map_structure(lambda *xs: np.mean(xs), *res))
