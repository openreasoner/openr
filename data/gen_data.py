import json
import logging
from datetime import datetime
from module import Node, perform_rollouts, process_annotations, calculate_mc_score

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
        
        # Initialize the root node and perform rollouts
        nodes = []
        root_node = Node(problem, "", final_answer)
        max_rollouts = 20
        rollouts, correctness_flags = perform_rollouts(root_node, max_rollouts)
        mc_score = calculate_mc_score(root_node)
        root_node.mc_score = mc_score

        nodes.append(root_node)

        # Check if further processing is needed
        if 0 < sum(correctness_flags) < max_rollouts:
            print("Processing annotations ...\n")
            filename = f"{i+1}_nodes_data.json"
            process_annotations(problem, final_answer, nodes, filename)
        
    # Log completion
    logging.info("Finished processing the JSON file.")

if __name__ == "__main__":
    main()
