from envs.prontoqa.env import (
    PrOntoQAEnv,
    COT_EXAMPLES,
    COT_TASK_DESC,
    PROBLEM_FORMAT_STR,
    SEP,
)

if __name__ == "__main__":
    problem_input = 'Butterflies are lepidopterans. Every arthropod is small. Whales are not small. Invertebrates are animals. Every insect is an arthropod. Lepidopterans are insects. Every insect is six-legged. Every arthropod is an invertebrate. Animals are multicellular. Polly is a lepidopteran. Is the statement "Polly is not small" true or false?'
    env = PrOntoQAEnv(
        config={},
        math_problems=[
            {
                "question": 'Butterflies are lepidopterans. Every arthropod is small. Whales are not small. Invertebrates are animals. Every insect is an arthropod. Lepidopterans are insects. Every insect is six-legged. Every arthropod is an invertebrate. Animals are multicellular. Polly is a lepidopteran. Is the statement "Polly is not small" true or false?',
                "answer": False,
            }
        ],
        tokenizer=None,
        llm_gen_fn=None,
        reset=False,
    )

    env.reset(False)
    print(env.get_state())
    print("correct: ", env._is_correct("The answer is false."))
    build_query_str = PrOntoQAEnv.build_query_str
    print("\n\n====== ZERO SHOT COT ============")
    print(
        build_query_str(
            COT_TASK_DESC, COT_EXAMPLES, PROBLEM_FORMAT_STR, problem_input, SEP, False
        )
    )
    print("\n\n====== FEW SHOT COT ============")
    print(
        build_query_str(
            COT_TASK_DESC, COT_EXAMPLES, PROBLEM_FORMAT_STR, problem_input, SEP, True
        )
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    print("\n\n====== default sft dataset ============")
    from envs import get_default_sft_data_builder, get_env_datasets

    train_ds, _ = get_env_datasets("prontoqa")
    q2idx_dict = {}
    for idx, problem_inst in enumerate(train_ds):
        question = problem_inst["question"]
        q2idx_dict[question] = idx
    sft_data = get_default_sft_data_builder("prontoqa")(
        "tsllm/envs/prontoqa/train_data/train.jsonl",
        q2idx_dict,
        tokenizer=tokenizer,
        add_eos_token=True,
        is_few_shot=False,
    )

    print("Len train_ds: {}\ntrian_ds[0]:\n{}".format(len(train_ds), train_ds[0]))
    print("Len sft_data: {}\nsft_data[0]:\n{}".format(len(sft_data), sft_data[0]))

    print("\n\n====== default critic dataset ============")
    from envs import get_default_critic_data_builder

    critic_data = get_default_critic_data_builder("prontoqa")(
        "tsllm/envs/prontoqa/train_data/train.jsonl",
        q2idx_dict,
        tokenizer=tokenizer,
        is_few_shot=False,
    )
    print(
        "Len critic_data: {}\ncritic_data[0]:\n{}".format(
            len(critic_data), critic_data[0]
        )
    )
    print(len(tokenizer.encode(critic_data[0]["query_str"] + critic_data[0]["answer"])))
    print(len(tokenizer.encode(critic_data[0]["query_str"])))
