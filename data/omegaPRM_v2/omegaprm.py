import heapq
import math
import random
import re
import json
from typing import List, Tuple, Dict, Any, Optional
import itertools
from llm_utils import LLMService

# Helper function to separate reasoning steps
def separate_steps(steps: List[str], mode: str = 'join') -> Any:
    delimiter = "\n\n"
    if mode == 'join':
        if not isinstance(steps, list):
            raise TypeError("For 'join' mode, 'steps' must be a list of strings.")
        return delimiter.join(steps)
    elif mode == 'split':
        if not isinstance(steps, str):
            raise TypeError("For 'split' mode, 'steps' must be a string.")
        return steps.split(delimiter)
    else:
        raise ValueError("Mode should be either 'join' or 'split'.")

# Helper function to check correctness of a generated response
def check_correctness(generated_response: str, expected_answer: str) -> bool:
    sentences = re.split(
        r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', generated_response.strip()
    )
    last_sentence = sentences[-1] if sentences else ''
    return expected_answer.strip() in last_sentence.strip()


class LanguageModel:
    def __init__(self, model_name="/root/.cache/modelscope/hub/Qwen/Qwen2___5-Math-7B-Instruct",
                 device="cuda", max_new_tokens=512, temperature=0.7, top_k=30, top_p=0.9, model_type="vllm"):
        """
        Initialize the LanguageModel with parameters for the LLM service.

        Parameters:
        - model_name (str): Path or model name for the LLM.
        - device (str): Device for computation (e.g., 'cuda', 'cpu').
        - max_new_tokens (int): Max tokens for response generation.
        - temperature (float): Sampling temperature for diversity.
        - top_k (int): Top-K sampling for diversity.
        - top_p (float): Top-P sampling for response diversity.
        """
        self.llm_service = LLMService(
            model_name=model_name,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            model_type=model_type
        )
        self.default_prompt = (
            "Please complete the answer for the question based on the given steps without generating existing steps again, "
            "and separate your following steps using \\n\\n.\n\n"
        )
        self.llm_service.start_service()

    def generate_rollout(self, state_prefix: str, num_copies) -> List[str]:
        """
        Combine the default prompt with the state prefix and generate a response.

        Parameters:
        - state_prefix (str): The current solution prefix.

        Returns:
        - str: Generated response from LLM.
        """
        prompt = self.default_prompt + state_prefix
        batch_response = self.llm_service.generate_response(prompt, num_copies)
        return batch_response # Assuming the response format has ['role'] entries and 'assistant' response

    def update_prompt(self, new_prompt: str):
        """
        Update the default prompt if necessary.

        Parameters:
        - new_prompt (str): The new prompt template.
        """
        self.default_prompt = new_prompt

    def evaluate_correctness(self, response: str, expected_answer: str) -> bool:
        """
        Check if the generated solution matches the expected answer.

        Parameters:
        - solution (str): The complete generated response.
        - expected_answer (str): The expected answer to compare with.

        Returns:
        - bool: True if the expected answer is in the final part of the solution.
        """
        return check_correctness(response, expected_answer)


# Define the State class
class State:
    def __init__(self, solution_prefix: str, parent: Optional['State'] = None):
        self.solution_prefix = solution_prefix  # Solution prefix as a single string
        self.parent = parent  # Reference to the parent state
        self.N = 0  # Visit count (number of times selected)
        self.total_rollouts = 0  # Total number of rollouts generated from this state
        self.correct_rollouts = 0  # Number of correct rollouts
        self.MC: Optional[float] = None  # Monte Carlo estimation (c/k)
        self.Q: Dict[str, float] = {}  # Q(s, r): estimated value for each rollout
        self.R: List[str] = []  # Set of all rollouts from this state
        self.incorrect_rollouts: List[str] = []  # List of incorrect rollouts
        self.children: List['State'] = []  # List of child states

    def add_rollout(self, rollout: str):
        self.R.append(rollout)

    def add_incorrect_rollout(self, rollout: str):
        if rollout not in self.incorrect_rollouts:
            self.incorrect_rollouts.append(rollout)

    def get_full_solution(self) -> str:
        # Return the complete solution from the root to this state
        if self.parent:
            return self.parent.get_full_solution() + '\n\n' + self.solution_prefix
        else:
            return self.solution_prefix

    def get_new_text(self) -> str:
        """
        Return the new text added at this node compared to the parent.
        """
        if self.parent:
            parent_text = self.parent.solution_prefix
            new_text = self.solution_prefix[len(parent_text):].strip()
            return new_text
        else:
            # Root node (the question)
            return self.solution_prefix.strip()

    def get_text_with_labels(self) -> Dict[str, Any]:
        """
        Return a nested dictionary where each node contains:
        - 'text': The new text at this node.
        - 'mc_value': The MC value at this node.
        - 'children': A list of child nodes with the same structure.
        """
        data = {
            'text': self.get_new_text(),
            'mc_value': self.MC,
            'children': [child.get_text_with_labels() for child in self.children]
        }
        return data


# Define the Search Tree class
class SearchTree:
    def __init__(self):
        self.root: Optional[State] = None
        self.nodes: List[State] = []  # List of all states

    def add_state(self, state: State):
        self.nodes.append(state)

# Define the Candidate Pool as a priority queue with update capability
class CandidatePool:
    def __init__(self):
        self.heap: List[Tuple[float, int]] = []  # Heap of (-priority, unique_id)
        self.entry_finder: Dict[int, Tuple[float, int]] = {}  # Maps unique_id to (-priority, unique_id)
        self.counter = itertools.count()  # Unique sequence count
        self.id_to_rollout: Dict[int, Tuple[State, str]] = {}  # Maps unique_id to (state, rollout)
        self.latest_id_per_rollout: Dict[Tuple[int, str], int] = {}  # Maps (state_id, rollout) to unique_id

    def add_or_update(self, state: State, rollout: str, priority: float):
        """
        Add a new rollout or update the priority of an existing rollout.

        Parameters:
        - state (State): The state associated with the rollout.
        - rollout (str): The rollout string.
        - priority (float): The new priority score.
        """
        state_id = id(state)  # Unique identifier for the state object
        rollout_key = (state_id, rollout)

        # Check if the rollout already exists in the pool
        if rollout_key in self.latest_id_per_rollout:
            # Previous unique_id exists; it is now outdated
            old_unique_id = self.latest_id_per_rollout[rollout_key]
            # Mark the old entry as invalid by removing it from entry_finder
            if old_unique_id in self.entry_finder:
                del self.entry_finder[old_unique_id]
                del self.id_to_rollout[old_unique_id]

        # Assign a new unique_id for the updated rollout
        unique_id = next(self.counter)
        self.latest_id_per_rollout[rollout_key] = unique_id

        # Add the new entry to the heap and mappings
        heapq.heappush(self.heap, (-priority, unique_id))  # Max-heap using negative priority
        self.entry_finder[unique_id] = (-priority, unique_id)
        self.id_to_rollout[unique_id] = (state, rollout)

    def pop(self) -> Tuple[Optional[State], Optional[str]]:
        """
        Pop the rollout with the highest priority.

        Returns:
        - Tuple[Optional[State], Optional[str]]: The state and rollout string, or (None, None) if empty.
        """
        while self.heap:
            neg_priority, unique_id = heapq.heappop(self.heap)
            # Check if this unique_id is still valid
            if unique_id in self.entry_finder:
                # Valid entry
                state, rollout = self.id_to_rollout.pop(unique_id)
                del self.entry_finder[unique_id]
                # Remove from latest_id_per_rollout
                state_id = id(state)
                rollout_key = (state_id, rollout)
                if self.latest_id_per_rollout.get(rollout_key) == unique_id:
                    del self.latest_id_per_rollout[rollout_key]
                return state, rollout
            # Else, outdated entry; skip
        return None, None

    def is_empty(self) -> bool:
        return not self.entry_finder

# Define the OmegaPRM algorithm
class OmegaPRM:
    def __init__(self, LM: LanguageModel,  c_puct: float, alpha: float, beta: float, L: int, k: int, N: int,
                 rollout_budget: int, save_data_tree: bool):
        """
        Initialize the OmegaPRM algorithm.

        Parameters:
        - LM (LanguageModel): The language model instance.
        - expected_answer (str): The expected answer for correctness checking.
        - c_puct (float): Exploration constant.
        - alpha (float): Weight for MC(s).
        - beta (float): Length penalty.
        - L (int): Maximum solution length.
        - k (int): Number of rollouts for Monte Carlo estimation.
        - N (int): Maximum search count.
        """
        self.LM = LM  # Language Model
        self.expected_answer = None
        self.c_puct = c_puct
        self.alpha = alpha
        self.beta = beta
        self.L = L
        self.k = k
        self.N = N
        self.rollout_budget = rollout_budget
        self.save_data_tree = save_data_tree

        self.T = SearchTree()
        self.C = CandidatePool()

        self.n = 0
        self.total_rollouts = 0




    def reset(self):
        """Reset internal state variables to prepare for a fresh run."""
        self.expected_answer = None
        self.T = SearchTree()  # Reset search tree
        self.C = CandidatePool()  # Reset candidate pool
        self.n = 0
        self.total_rollouts = 0
        self.collected_data = []  # Clear collected data

    def run(self, question: str, answer: str) -> List:
        """
        Execute the OmegaPRM algorithm.

        Parameters:
        - question (str): The question to generate solutions for.

        Returns:
        - Collected data: List of dictionaries.
        """
        self.reset()

        print(f"Running OmegaPRM for question: '{question}'\n")
        # Initialization
        initial_state = State(solution_prefix=question, parent=None)
        self.expected_answer = answer
        self.T.root = initial_state
        self.T.add_state(initial_state)
        self.n = 0

        # Monte Carlo Estimation for initial_state
        self.monte_carlo_estimation(initial_state)

        # Main loop
        while self.n < self.N and self.total_rollouts < self.rollout_budget and not self.C.is_empty():
            # Selection Phase
            selected_state, selected_rollout = self.selection_phase()
            if selected_state is None or selected_rollout is None:
                # print("No more candidates to explore. Terminating search.\n")
                break

            self.expansion_phase_binary_search(selected_state, selected_rollout)

            # Maintenance Phase
            self.maintenance_phase(selected_state)

            # Increment search count
            self.n += 1

        if self.save_data_tree:
            data = self.collect_tree_structure()
        else:
            data = self.collect_solution_prefixes()
        return data

    def monte_carlo_estimation(self, state: State):
        """
        Perform Monte Carlo estimation for state by generating k rollouts
        and computing MC(s) = c / k, where c is the number of correct rollouts.
        """
        c = 0  # Correct rollouts count
        incorrect_rollouts = []
        correct_rollouts = []
        batct_rollouts = self.LM.generate_rollout(state.solution_prefix, self.k)

        # Increment visit count of selected state
        state.N += 1

        for i, rollout in enumerate(batct_rollouts):
            # Increment number of total rollouts
            self.total_rollouts += 1

            # Generate rollout r_i

            state.add_rollout(rollout)

            # Evaluate correctness of final answer in rollout
            full_solution = (state.solution_prefix + '\n\n' + rollout).strip() if state.solution_prefix else rollout
            is_correct = self.LM.evaluate_correctness(full_solution, self.expected_answer)

            # print(f"Rollout {i + 1} Correctness: {'Correct' if is_correct else 'Incorrect'}\n")

            if is_correct:
                c += 1
                correct_rollouts.append(rollout)
            else:
                incorrect_rollouts.append(rollout)
                state.add_incorrect_rollout(rollout)  # Track incorrect rollouts

        # Update total rollouts and correct rollouts
        state.total_rollouts += self.k
        state.correct_rollouts += c
        state.MC = state.correct_rollouts / state.total_rollouts if state.total_rollouts > 0 else 0

        # print(f"Monte Carlo Estimation for State ID {self.T.nodes.index(state)}: MC = {state.MC:.2f}, Total Rollouts = {state.total_rollouts}, Correct Rollouts = {state.correct_rollouts}\n")

        if state.MC == 1.0:
            # Add all correct rollouts to the tree as new states
            for rollout in correct_rollouts:
                self.add_correct_rollout_to_tree(state, rollout)
        elif state.MC == 0.0:
            # State is incorrect; no further action
            # print(f"State ID {self.T.nodes.index(state)} has MC == 0. No further rollouts will be added.\n")
            return
        else:
            # 0 < MC(s) < 1.0
            # Add correct rollouts to the tree
            for rollout in correct_rollouts:
                self.add_correct_rollout_to_tree(state, rollout)
            # Add incorrect rollouts to candidate pool with updated priorities
            for rollout in incorrect_rollouts:

                priority = self.compute_selection_score(state, rollout)
                self.C.add_or_update(state, rollout, priority)


    def compute_Q(self, state: State, rollout: str) -> float:
        """
        Compute Q(s, r) = alpha^{1 - MC(s)} * beta^{len(r)/L}, where len(r) is based on word count.
        """
        # Count words in the rollout
        word_count = len(rollout.split())
        length_penalty = word_count / self.L
        Q_value = (self.alpha ** (1 - state.MC)) * (self.beta ** length_penalty)
        return Q_value

    def compute_U(self, state: State) -> float:
        """
        Compute U(s) = c_puct * sqrt(sum_{s'} N(s')) / (1 + N(s))
        """
        N_total = sum(s.N for s in self.T.nodes)
        if N_total == 0:
            N_total = 1  # Prevent division by zero
        U_s = self.c_puct * (math.sqrt(N_total)) / (1 + state.N)
        return U_s

    def compute_selection_score(self, state: State, rollout: str) -> float:
        """
        Compute selection score: Score(s, r) = Q(s, r) + U(s)
        """
        Q_s_r = self.compute_Q(state, rollout)
        U_s = self.compute_U(state)
        score = Q_s_r + U_s
        return score

    def selection_phase(self) -> Tuple[Optional[State], Optional[str]]:
        """
        Select (state, rollout) with the highest score from candidate pool C.
        """
        selected_state, selected_rollout = self.C.pop()
        return selected_state, selected_rollout

    def add_correct_rollout_to_tree(self, parent_state: State, rollout: str):
        """
        Add the correct rollout to the tree as a child of parent_state.
        """
        new_solution_prefix = (parent_state.solution_prefix + '\n\n' + rollout).strip() if parent_state.solution_prefix else rollout
        new_state = State(solution_prefix=new_solution_prefix, parent=parent_state)
        new_state.MC = 1.0  # Since the rollout is correct
        new_state.total_rollouts = 0
        new_state.correct_rollouts = 0
        self.T.add_state(new_state)
        parent_state.children.append(new_state)  # Add to parent's children


    def expansion_phase_binary_search(self, parent_state: State, rollout: str):
        """
        Expansion phase that adds the rollout as a new state and performs Monte Carlo estimation
        using Binary Search to efficiently find the correct rollout.

        Parameters:
        - parent_state (State): The state from which the rollout was selected.
        - rollout (str): The rollout string that was selected and is incorrect.
        """
        # Separate the rollout into individual steps
        steps = separate_steps(rollout, mode='split')

        # Perform binary search to find incorrect steps

        self.binary_search_incorrect_step(parent_state, steps, 0, len(steps) - 1)

    def binary_search_incorrect_step(self, s_ast: State, steps: List[str], left: int, right: int):
        """
        Recursively perform binary search to find all incorrect steps in the rollout.

        Parameters:
        - s_ast (State): The selected parent state.
        - steps (List[str]): The rollout steps as a list.
        - left (int): Left index of the current search interval.
        - right (int): Right index of the current search interval.
        """
        if left > right:
            return

        mid = (left + right) // 2
        new_steps = steps[left:mid + 1]
        if new_steps:
            prefix_solution = s_ast.solution_prefix + '\n\n' + separate_steps(new_steps, mode='join')
        else:
            prefix_solution = s_ast.solution_prefix
        # Create new state s_new
        s_new = State(solution_prefix=prefix_solution.strip(), parent=s_ast)
        self.T.add_state(s_new)
        s_ast.children.append(s_new)

        # Perform Monte Carlo estimation for s_new
        self.monte_carlo_estimation(s_new)


        if s_new.MC == 0:

            # Found incorrect step; continue searching in the left half to find earlier incorrect steps

            self.binary_search_incorrect_step(s_ast, steps, left, mid - 1)
        else:
            # Steps up to mid are correct; continue searching in the right half

            self.binary_search_incorrect_step(s_new, steps, mid + 1, right)

    def maintenance_phase(self, state: State):
        """
        Update statistics and candidate pool for all incorrect rollouts associated with the state.

        Parameters:
        - state (State): The state whose incorrect rollouts need to be updated.
        """

        # Iterate through all incorrect rollouts of the state
        for rollout in state.incorrect_rollouts:
            # Since we've already determined these rollouts are incorrect, no need to re-evaluate correctness

            priority = self.compute_selection_score(state, rollout)
            # Update the candidate pool with the new priority
            self.C.add_or_update(state, rollout, priority)
            # print(f"Updated Incorrect Rollout: '{rollout}' with new priority: {priority:.4f}")

        # print("Maintenance Phase Completed.\n")

    def collect_solution_prefixes(self) -> List[Dict[str, Any]]:
        """
        Collect all solution prefixes and their corresponding MC values from the search tree.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing solution prefixes and their MC values.
        """
        collected_data = []
        for node in self.T.nodes:
            solution_prefix = node.solution_prefix
            mc_value = node.MC
            collected_data.append({
                "solution_prefix": solution_prefix,
                "mc_value": mc_value
            })
        return collected_data

    def collect_tree_structure(self) -> Dict[str, Any]:
        """
        Collect the tree structure starting from the root.

        Returns:
            Dict[str, Any]: A nested dictionary representing the tree structure.
        """
        if self.T.root:
            tree_data = self.T.root.get_text_with_labels()
            return tree_data
        return {}






# Example usage
if __name__ == "__main__":
    # Initialize the Language Model
    LM = LanguageModel(
        device="cuda",
        max_new_tokens=2048,
        model_type="vllm"
    )

    # Define the question and expected answer
    question = "Melinda will roll two standard six-sided dice and make a two-digit number with the two numbers she rolls. For example, if she rolls a 6 and a 3, she can either form 36 or 63. What is the probability that she will be able to make an integer between 10 and 20, inclusive? Express your answer as a common fraction."
    expected_answer =  "\\frac{11}{36}"

    # Initialize OmegaPRM with parameters
    omega_prm = OmegaPRM(
        LM=LM,
        c_puct=0.125,
        alpha=0.5,
        beta=0.9,
        L=500,
        k=16,
        N=10,
        rollout_budget=100,
        save_data_tree=True,
    )

    # Run the OmegaPRM algorithm
    collected_data = omega_prm.run(question, expected_answer)

    # Save the collected solutions to a JSON file
    with open("collected_solutions2.json", "w") as f:
        json.dump(collected_data, f, indent=4)


