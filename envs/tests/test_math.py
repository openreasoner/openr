from envs.MATH import (
    Env,
    get_train_test_dataset,
    extract_groundtruth,
    judge_correct,
    extract_answer,
)

if __name__ == "__main__":
    from transformers import AutoTokenizer

    # tokenizer = AutoTokenizer.from_pretrained(
    #     "/hpc2ssd/JH_DATA/spooler/qxiao183/workspace/hf_models/peiyi9979/mistral-7b-sft"
    # )
    # _, ds = get_train_test_dataset()
    # print(len(ds))
    # print(ds[0])
    # problem = ds[0]
    # env = Env(
    #     config={
    #         "max_actions": 10,
    #         "max_length": 10,
    #         "stop_str": "The answer is ",
    #         "generation_config": {
    #             "max_new_tokens": 64,
    #             "do_sample": True,
    #             "temperature": 1,
    #             "top_p": 1.0,
    #             "top_k": 100,
    #             "return_dict_in_generate": True,
    #             "output_scores": True,
    #             "use_cache": True,
    #         },
    #     },
    #     math_problems=[
    #         {
    #             "question": problem["question"],
    #             "answer": extract_groundtruth(problem["answer"]),
    #         }
    #     ],
    #     llm_gen_fn=None,
    #     tokenizer=tokenizer,
    #     reset=False,
    # )

    # env.reset(update_legal_action=False)
    # print(env.get_state())
    # res = env.step(f"The answer is: {problem['answer']} ки", update_legal_action=False)
    # print(res)
    # print(env.get_state())

    a = "Step 1: To convert from rectangular coordinates to polar coordinates, we can use the formulas:\n$r = \\sqrt{x^2 + y^2}$ and $\\theta = \\arctan\\left(\\frac{y}{x}\\right)$. ки\nStep 2: In this case, $x = 0$ and $y = 3$. ки\nStep 3: Plugging these values into the formulas, we get:\n$r = \\sqrt{0^2 + 3^2} = 3$ and $\\theta = \\arctan\\left(\\frac{3}{0}\\right)$. ки\nStep 4: Since $\\frac{3}{0}$ is undefined, we need to find an alternative way to find $\\theta$. ки\nStep 5: We know that the point $(0,3)$ lies on the positive y-axis, so $\\theta = \\frac{\\pi}{2}$. ки\nStep 6: Therefore, the polar coordinates of the point $(0,3)$ are $\\boxed{(3,\\frac{\\pi}{2})}$. The answer is: (3,\\frac{\\pi}{2}) ки"
    g = "\\left( 3, \\frac{\\pi}{2} \\right)"
    print(judge_correct(None, g, extract_answer(a)))
