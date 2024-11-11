import logging
import os
import json
import argparse
from omegaprm import OmegaPRM, LanguageModel
from tqdm import tqdm
from typing import Dict

DS_NAME = "math-aps-v2"


# Set up logging based on provided log file prefix
def setup_logging(log_file_prefix: str):
    log_filename = f"{log_file_prefix}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# Load questions from JSON
def load_questions(filepath: str):
    with open(filepath, 'r') as f:
        return json.load(f)


# Filter a single question based on 32 rollouts
def should_process_question(question: Dict[str, str], llm: LanguageModel) -> bool:
    prompt = question["problem"]
    correct_answer = question["final_answer"]

    has_correct = False
    has_incorrect = False

    initial_batch_answers = llm.generate_rollout(prompt, 32)

    for answer in initial_batch_answers:

        if llm.evaluate_correctness(answer, correct_answer):
            has_correct = True
        else:
            has_incorrect = True

        if has_correct and has_incorrect:
            logger.info(f"Question passed filter: {question['problem']}")
            return True


    return False


# Run OmegaPRM on a question if it passes the filter
def process_question(omega_prm: OmegaPRM, question: Dict[str, str]):
    logger.info(f"Processing question with OmegaPRM: {question['problem']}")
    reasoning_steps = omega_prm.run(question["problem"], question["final_answer"])
    collected_data = {
        "question": question["problem"],
        "final_answer": question["final_answer"],
        "reasoning_steps": reasoning_steps
    }
    return collected_data


# Save collected data for each question
def save_question_data(collected_data: Dict, index: int, output_path: str):
    with open(output_path, 'w') as f:
        json.dump(collected_data, f, indent=4)
    logger.info(f"Saved processed data to {output_path}")


def main(args):
    global logger
    logger = setup_logging(args.log_file_prefix)

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{DS_NAME}.jsonl")

    logger.info("Starting OmegaPRM processing")
    logger.info(f"Using model: {args.model_name} on device: {args.device}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Question file: {args.question_file}")

    questions = load_questions(args.question_file)

    llm = LanguageModel(
        model_name=args.model_name,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        model_type=args.model_type
    )

    omega_prm = OmegaPRM(
        LM=llm,
        c_puct=args.c_puct,
        alpha=args.alpha,
        beta=args.beta,
        L=args.length_scale,
        k=args.num_rollouts,
        N=args.max_search_count,
        rollout_budget=args.rollout_budget,
        save_data_tree=args.save_data_tree,
    )

    processed_count = 0  # Counter for processed questions

    for idx, question in enumerate(questions):
        if should_process_question(question, llm):
            collected_data = process_question(omega_prm, question)
            save_question_data(collected_data, idx, output_file)
            processed_count += 1
        else:
            logger.info(f"Skipping question: {question['problem']}")

    # Log summary
    logger.info(f"Total questions processed by OmegaPRM: {processed_count}/{len(questions)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OmegaPRM on filtered questions")

    parser.add_argument("--question_file", type=str, required=True, help="Path to the questions JSON file")
    parser.add_argument("--output_dir", type=str, default="output", help=f"Directory to save the output file {DS_NAME}.jsonl")
    parser.add_argument("--log_file_prefix", type=str, default="omega_prm", help="Prefix for the log files")
    parser.add_argument("--model_name", type=str, default="/root/.cache/modelscope/hub/Qwen/Qwen2___5-Math-7B-Instruct",
                        help="Model name or path for the language model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (e.g., 'cuda', 'cpu')")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Max tokens for LLM generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for LLM generation")
    parser.add_argument("--top_k", type=int, default=30, help="Top-K sampling for LLM generation")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P sampling for LLM generation")
    parser.add_argument("--model_type", type=str, default="hf",
                        help="Model backend to use ('hf' for Hugging Face or 'vllm')")

    # OmegaPRM parameters with provided defaults
    parser.add_argument("--c_puct", type=float, default=0.125, help="Exploration constant for OmegaPRM")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for MC(s) in OmegaPRM")
    parser.add_argument("--beta", type=float, default=0.9, help="Length penalty for OmegaPRM")
    parser.add_argument("--length_scale", type=int, default=500, help="length scale in OmegaPRM")
    parser.add_argument("--num_rollouts", type=int, default=16,
                        help="Number of rollouts for Monte Carlo estimation in OmegaPRM")
    parser.add_argument("--max_search_count", type=int, default=20, help="Max search count in OmegaPRM")
    parser.add_argument("--rollout_budget", type=int, default=200, help="Rollout budget for OmegaPRM")
    parser.add_argument("--save_data_tree", type=bool, default=True, help="Save data in tree structure for OmegaPRM")


    args = parser.parse_args()
    main(args)
