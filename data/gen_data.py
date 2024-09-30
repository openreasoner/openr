import json
import logging
from datetime import datetime
from core import State
from utils import getrollouts, process_annotation, cal_mc_bs, cal_mc

def load_json_file(file_path):
    """
    Load data from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file.
        
    Returns:
        list: A list of dictionaries containing the problem and final answer.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def setup_logging(log_file):
    """
    Set up logging configuration.
    
    Args:
        log_file (str): Path to the log file.
    """
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def main():
    # Path to the JSON file and log file
    json_file_path = 'extracted_problems_and_answers.json'
    log_file_path = 'processing_log.log'
    
    # Set up logging
    setup_logging(log_file_path)
    
    # Start the process and log it
    logging.info("Started processing the JSON file.")
    
    # Load the JSON data
    data = load_json_file(json_file_path)
    
    # Process each problem and its final answer
    for i, item in enumerate(data):
        problem = item.get('problem', 'No problem found')
        final_answer = item.get('final_answer', 'No answer found')
        
        # Print to console
        print(f"Problem {i + 1}: {problem}")
        print(f"Final Answer: {final_answer}")
        
        # Log each problem and answer
        logging.info(f"Processed Problem {i + 1}: {problem}")
        logging.info(f"Final Answer: {final_answer}")
        
        # Call getrollout and handle the result
        states = []
        root = State(problem, "", final_answer)
        max_roll_num = 20
        rollouts, corrs = getrollouts(root, max_roll_num)
        mcst = cal_mc_bs(root)
        root.mc = mcst

        states.append(root)

        if sum(corrs) > 0 and sum(corrs) < max_roll_num: 
            print("Process annotation ...\n")
            filename = str(i+1) +'_states_list.json'
            process_annotation(problem, final_answer, states, filename)
    
    # Log completion
    logging.info("Finished processing the JSON file.")

if __name__ == "__main__":
    main()