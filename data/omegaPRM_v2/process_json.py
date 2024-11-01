import json
import os
import random
from typing import List
import math

def sample_questions(input_filepath: str, output_filepath: str, num_samples: int):
    """
    Samples a specified number of questions from the input JSON file and saves them to an output JSON file.

    Parameters:
    - input_filepath (str): Path to the original JSON file containing all questions.
    - output_filepath (str): Path to save the sampled questions JSON file.
    - num_samples (int): Number of questions to sample.
    """
    with open(input_filepath, 'r') as f:
        questions = json.load(f)

    # Sample questions
    sampled_questions = random.sample(questions, min(num_samples, len(questions)))

    # Save sampled questions to the output file
    with open(output_filepath, 'w') as f:
        json.dump(sampled_questions, f, indent=4)

    print(f"Saved {len(sampled_questions)} sampled questions to {output_filepath}")


def split_questions(input_filepath: str, output_dir: str, questions_per_file: int):
    """
    Splits the input JSON file into multiple smaller JSON files, each containing a specified number of questions.

    Parameters:
    - input_filepath (str): Path to the original JSON file containing all questions.
    - output_dir (str): Directory to save the split JSON files.
    - questions_per_file (int): Number of questions per split file.
    """
    with open(input_filepath, 'r') as f:
        questions = json.load(f)

    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Split questions into chunks and save each chunk as a separate file
    for i in range(0, len(questions), questions_per_file):
        chunk = questions[i:i + questions_per_file]
        output_filepath = os.path.join(output_dir, f"questions_part_{i // questions_per_file + 1}.json")
        with open(output_filepath, 'w') as f:
            json.dump(chunk, f, indent=4)

        print(f"Saved {len(chunk)} questions to {output_filepath}")


def split_questions_uniformly(input_filepath: str, output_directory: str, num_files: int):
    """
    Split a JSON file containing questions into a specified number of files with approximately equal questions.

    Parameters:
    - input_filepath (str): Path to the JSON file containing the list of questions.
    - output_directory (str): Directory to save the split JSON files.
    - num_files (int): Number of files to split the questions into.

    Each output file will contain approximately len(questions) / num_files questions.
    """
    # Load questions from the input file
    with open(input_filepath, 'r') as f:
        questions = json.load(f)

    # Calculate the number of questions per file
    total_questions = len(questions)
    questions_per_file = math.ceil(total_questions / num_files)

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Split the questions and write to output files
    for i in range(num_files):
        start_idx = i * questions_per_file
        end_idx = min(start_idx + questions_per_file, total_questions)
        questions_subset = questions[start_idx:end_idx]

        output_filepath = os.path.join(output_directory, f"questions_part_{i + 1}.json")
        with open(output_filepath, 'w') as f_out:
            json.dump(questions_subset, f_out, indent=4)

        print(f"Saved {len(questions_subset)} questions to {output_filepath}")


# Example usage
if __name__ == "__main__":
    split_questions_uniformly("extracted_problems_and_answers.json", "output_directory", 8)
#
# # Sampling a subset of questions from the original file
#   sample_questions("extracted_problems_and_answers.json", "sampled_questions.json", 10)
#
# # Splitting the original file into multiple files with each containing 5 questions
#   split_questions("extracted_problems_and_answers.json", "output_directory", 5)
