from typing import List, Dict, Tuple
import json

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Optional
import math, random
from copy import deepcopy
from enum import Enum, unique

from envs.MATH.parse_utils_qwen import extract_answer as extract_fn, parse_ground_truth
from envs.MATH.grader import math_equal


def override(f):
    return f


@unique
class Node_Type(Enum):
    USER_QUESTION = "USER_QUESTION"
    REPHRASED_USER_QUESTION = "REPHRASED_USER_QUESTION"
    DIRECT_ANSWER = "DIRECT_ANSWER"
    SUBQUESTION = "SUBQUESTION"
    RE_SUBANSWER = "RE_SUBANSWER"
    OST_STEP = "OST_STEP"


class GeneratorError(Exception):
    def __init__(self, source, io_input, io_output_list) -> None:
        super().__init__()

        self.source = source
        self.io_input = io_input
        self.io_output_list = io_output_list


def read_txt(file_path):
    assert str(file_path).endswith(".txt")
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
    return data


def read_json(file_path):
    assert str(file_path).endswith(".json")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_json(js_obj, file_path):
    assert str(file_path).endswith(".json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(js_obj, f, indent=4)


def extract_answer(answer_str: str) -> str:
    return extract_fn(answer_str, data_name="math")


def extract_groundtruth(groundtruth_str: str) -> str:
    return parse_ground_truth(groundtruth_str, data_name="math")


def judge_correct(
    problem_str: str, extracted_groundtruth: Optional[str], answer: str
) -> bool:
    # return grade_answer(given_answer=answer, ground_truth=extracted_groundtruth)
    result = math_equal(answer, extracted_groundtruth)
    return result


def concat_ost_steps(solution_trace: Dict[int, Dict[str, str]]) -> Tuple[str, int]:
    """Return: concatenated one-step thought steps, next one-step thought step id"""
    last_tuple = list(solution_trace.items())[-1]
    last_tuple_id, last_tuple_recording = last_tuple[0], last_tuple[1]
    assert "ost_step" in last_tuple_recording.keys()
    if len(last_tuple_recording["ost_step"]) > 0:
        solution_trace_str = ""
        for step_id, step_text in last_tuple_recording["ost_step"].items():
            solution_trace_str += f"Step {step_id}: " + step_text + "\n"
        return solution_trace_str, step_id + 1
    else:
        # no one-step thought step yet
        return "", 1


def concat_subqs_and_subas(
    solution_trace: Dict[int, Dict[str, str]], question_index: int
) -> Tuple[str, int]:
    """Return: concatenated subqs and suba, next subquestion id"""
    solution_trace_str = ""

    for subquestion_id, solution_step in solution_trace.items():
        if subquestion_id == 0:
            continue

        assert subquestion_id > 0
        assert (
            "subquestion" in solution_step.keys()
            and "subanswer" in solution_step.keys()
        )

        solution_trace_str += (
            f"Question {question_index}."
            + str(subquestion_id)
            + ": "
            + solution_step["subquestion"]
        )
        solution_trace_str += "\n"
        solution_trace_str += (
            f"Answer {question_index}."
            + str(subquestion_id)
            + ": "
            + solution_step["subanswer"]["text"]
        )
        solution_trace_str += "\n"

    next_subquestion_id = int(sorted(solution_trace.keys())[-1]) + 1
    return solution_trace_str, next_subquestion_id


def split_user_question(user_question: str):
    user_question = user_question.strip().rstrip(".")
    last_period_id = user_question.rfind(".")
    assert last_period_id < len(user_question) - 1
    user_question_context = user_question[: last_period_id + 1].strip()
    user_question_problem = user_question[last_period_id + 1 :].strip()
    return user_question_context, user_question_problem


def reach_terminal_subquestion(subquestion: str, user_question: str):
    assert subquestion is not None

    if "Now we can answer" in subquestion:  # in the prompt template
        #! remember that: when the original question is answerable, please start the subquestion with "Now we can answer the question: "
        return True

    user_question_2nd_part = split_user_question(user_question)[1]
    if user_question_2nd_part.lower() in subquestion.lower():
        return True

    return False


def reach_terminal_ost_step(ost_step: str):
    assert ost_step is not None

    return "answer is" in ost_step.lower()


def make_hint(
    solution_trace: Dict[int, Dict[str, str]],
    node_type: Node_Type,
    new_subq=None,
    new_suba=None,
    new_ost_step=None,
) -> str:
    if node_type in [Node_Type.SUBQUESTION, Node_Type.RE_SUBANSWER]:
        hint = ""

        for subquestion_id, solution_step in solution_trace.items():
            if subquestion_id == 0:
                continue

            assert subquestion_id > 0
            assert (
                "subquestion" in solution_step.keys()
                and "subanswer" in solution_step.keys()
            )

            hint += f"Hint " + str(subquestion_id) + ": " + solution_step["subquestion"]
            hint += " "
            hint += solution_step["subanswer"]["text"]
            hint += "\n"

        if new_subq is not None and new_suba is not None:
            hint += f"Hint {len(solution_trace)}: " + new_subq + " " + new_suba

        hint = hint.strip("\n")
    elif node_type is Node_Type.OST_STEP:
        hint = "Hint: "
        last_tuple = list(solution_trace.items())[-1]
        last_tuple_recording = last_tuple[1]
        assert last_tuple_recording["ost_step"]
        for step_id, step_text in last_tuple_recording["ost_step"].items():
            hint += step_text + " "

        if new_ost_step is not None:
            hint += new_ost_step

        hint = hint.strip(" ")
    else:
        raise ValueError(f"Invalid node type: {node_type}.")

    return hint


def find_valid_solution_nodes(root_node):
    valid_solution_nodes = []

    def recursion(node):
        if node.is_valid_solution_node():
            valid_solution_nodes.append(node)
            return

        if not node.children:  #! no children
            return

        for child in node.children:
            recursion(child)

    recursion(root_node)

    return valid_solution_nodes


def stochastic_find_best_solution(
    root_node,
    evaluator,
    enable_potential_score,
):
    # todo: what strategy do we use to select best node?
    """The function finds the best solution from the solution nodes in the MCTS tree.
    Return: top answer, top solution, confidence of the top answer, the corresponding node of the answer, all solution nodes
    """
    solution_nodes = find_valid_solution_nodes(root_node)

    if len(solution_nodes) == 0:
        return None, None

    def extract_solution_from_node(node):
        # original repo has bug here, the .is_valid_solution_node consider SUBQUESTION, OST and DIRECT ANSWER,
        # but here it does not consider OST!!!!
        if node.node_type is Node_Type.SUBQUESTION:
            return node.subanswer
        elif node.node_type is Node_Type.DIRECT_ANSWER:
            return node.direct_answer
        elif node.node_type is Node_Type.OST_STEP:
            return node.ost_step
        else:
            return None

    solutions = [extract_solution_from_node(node) for node in solution_nodes]

    def calculate_potential_score_for_solution_node(node):
        model_answer = evaluator.extract_answer_from_model_completion(
            extract_solution_from_node(node)
        )
        potential_answers_history = (
            node.potential_answers_history
        )  # {depth -> [potential answers]}
        assert potential_answers_history[node.depth] is None

        potential_score = 1
        for depth, depth_potential_answers in potential_answers_history.items():
            if depth < node.depth:
                depth_score = sum(
                    evaluator.check_answers_equiv(dpa, model_answer)
                    for dpa in depth_potential_answers
                ) / len(depth_potential_answers)
                potential_score *= depth_score

        node.set_potential_score(potential_score)
        return potential_score

    prior_weights = (
        [calculate_potential_score_for_solution_node(node) for node in solution_nodes]
        if enable_potential_score
        else None
    )
    top_answer, top_completion, top_completion_id, top_confidence = (
        evaluator.stochastic_find_most_confident_answer(
            completions=solutions, prior_weights=prior_weights
        )
    )
    return (
        top_answer,
        top_completion,
        top_confidence,
        solution_nodes[top_completion_id],
        solution_nodes,
        solutions,
    )


class MCTS_Node(ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    def __init__(self) -> None:
        super().__init__()

        # global node_cnt
        self.id = None  # defined when creating one RstarLanguageNode

        self.rollout_id = None

    def set_rollout_id(self, rollout_id: int):
        self.rollout_id = rollout_id

    @abstractmethod
    def find_children(self, rollout_id: int):
        "All possible successors of this board state"
        raise NotImplementedError

    @abstractmethod
    def is_terminal(self):
        "Returns True if the node has no children"
        raise NotImplementedError

    @abstractmethod
    def calculate_reward(self):
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        raise NotImplementedError

    @abstractmethod
    def skip_backprop(self):
        "If True, the reward of this node will not be accumulated in the backpropagation step."
        raise NotImplementedError


class RstarLanguageNode(MCTS_Node):
    def __init__(
        self,
        parent: "Reasoning_MCTS_Node",
        depth: int,
        node_type: Node_Type,
        verbose: bool = False,
        # --- For instantiating root node ---
        node_value: float = None,
        disable_a5: bool = None,
        user_question: str = None,
        max_depth_allowed: int = None,
        disable_a1: bool = None,
        # --- For instantiating REPHRASED_USER_QUESTION node ---
        rephrased_user_question: str = None,
        # ------------------------------------------------------
        expected_answer: str = None,
        # --- For instantiating DIRECT_ANSWER node ---
        direct_answer: str = None,
        # --------------------------------------------
        # --- For instantiating SUBQUESTION node ---
        subquestion: str = None,
        subanswer: str = None,
        is_new_subquestion: bool = None,
        # ------------------------------------------
        # --- For instantiating RE_SUBANSWER node ---
        re_subanswer: str = None,
        # -------------------------------------------
        # --- For instantiating OST_STEP node ---
        ost_step: str = None,
        question_index: int = None,
        id: int = None,
    ) -> None:
        super().__init__()

        # ! sanity checks
        try:
            assert depth is not None
            assert node_type is not None
            if node_value is not None:
                assert node_value > 0, breakpoint()

            if node_type is Node_Type.USER_QUESTION:
                assert depth == 0
                assert all(
                    attr is None
                    for attr in [
                        parent,
                        node_value,
                        rephrased_user_question,
                        direct_answer,
                        subquestion,
                        subanswer,
                        is_new_subquestion,
                        re_subanswer,
                        ost_step,
                    ]
                )
                assert all(
                    attr is not None
                    for attr in [
                        disable_a5,
                        user_question,
                        expected_answer,
                        max_depth_allowed,
                        disable_a1,
                    ]
                )
            elif node_type is Node_Type.REPHRASED_USER_QUESTION:
                assert depth == 1
                assert all(
                    attr is None
                    for attr in [
                        node_value,
                        disable_a5,
                        user_question,
                        expected_answer,
                        direct_answer,
                        subquestion,
                        subanswer,
                        is_new_subquestion,
                        re_subanswer,
                        ost_step,
                        max_depth_allowed,
                        disable_a1,
                    ]
                )
                assert all(
                    attr is not None for attr in [parent, rephrased_user_question]
                )
            elif node_type is Node_Type.DIRECT_ANSWER:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        disable_a5,
                        user_question,
                        expected_answer,
                        subquestion,
                        subanswer,
                        is_new_subquestion,
                        re_subanswer,
                        ost_step,
                        max_depth_allowed,
                        disable_a1,
                    ]
                )
                assert all(
                    attr is not None for attr in [parent, node_value, direct_answer]
                )
            elif node_type is Node_Type.SUBQUESTION:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        disable_a5,
                        user_question,
                        expected_answer,
                        direct_answer,
                        re_subanswer,
                        ost_step,
                        max_depth_allowed,
                        disable_a1,
                    ]
                )
                assert all(
                    attr is not None
                    for attr in [
                        parent,
                        node_value,
                        subquestion,
                        subanswer,
                        is_new_subquestion,
                    ]
                )
            elif node_type is Node_Type.RE_SUBANSWER:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        disable_a5,
                        user_question,
                        expected_answer,
                        direct_answer,
                        subquestion,
                        subanswer,
                        is_new_subquestion,
                        ost_step,
                        max_depth_allowed,
                        disable_a1,
                    ]
                )
                assert all(
                    attr is not None for attr in [parent, node_value, re_subanswer]
                )
            elif node_type is Node_Type.OST_STEP:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        node_value,
                        disable_a5,
                        user_question,
                        rephrased_user_question,
                        expected_answer,
                        direct_answer,
                        subquestion,
                        subanswer,
                        is_new_subquestion,
                        re_subanswer,
                        max_depth_allowed,
                        disable_a1,
                    ]
                )
                assert all(attr is not None for attr in [parent, ost_step])
        except AssertionError:
            print(f"Instantiating node with type {node_type} failed!")
            breakpoint()
            exit()

        self.id = id
        #! attributes
        self.parent = parent  # if parent is None, then the node is the root
        self.children: List["Reasoning_MCTS_Node"] = []
        self.depth = depth
        self.node_type = node_type
        self.node_value = node_value
        self.direct_answer = direct_answer
        self.subquestion = subquestion
        self.subanswer = subanswer
        self.is_new_subquestion = is_new_subquestion
        self.re_subanswer = re_subanswer
        self.ost_step = ost_step

        ## additional parameter
        self._visit_count = 0
        self._value_sum = 0  # self.node_value

        if parent is None:  # root
            self.verbose = verbose
            self.user_question = user_question
            self.expected_answer = expected_answer
            self.disable_a5 = disable_a5
            self.question_index = question_index
            self.max_depth_allowed = max_depth_allowed
            self.disable_a1 = disable_a1
        else:  # inherit from parent
            self.verbose = parent.verbose
            self.user_question = parent.user_question
            self.expected_answer = parent.expected_answer
            self.disable_a5 = parent.disable_a5
            self.question_index = parent.question_index
            self.max_depth_allowed = parent.max_depth_allowed
            self.disable_a1 = parent.disable_a1

        #! keep track of paraphrasing
        if node_type is Node_Type.USER_QUESTION:
            self.paraphrased = False
        elif node_type is Node_Type.REPHRASED_USER_QUESTION:
            self.paraphrased = True
            self.user_question = rephrased_user_question
        else:
            assert parent is not None, breakpoint()
            self.paraphrased = parent.paraphrased

        #! record number of subquestions till now
        if parent is None:  # root
            self.subquestion_counter = 0
        else:
            if node_type is Node_Type.SUBQUESTION and is_new_subquestion:
                self.subquestion_counter = parent.subquestion_counter + 1
            else:
                self.subquestion_counter = parent.subquestion_counter

        #! record number of one-step thought steps till now
        if parent is None:  # root
            self.ost_step_counter = 0
        else:
            if node_type is Node_Type.OST_STEP:
                self.ost_step_counter = parent.ost_step_counter + 1
            else:
                self.ost_step_counter = parent.ost_step_counter

        #! record solution trace from root to the current node. key: subquestion id
        if parent is None:  # root
            assert self.node_type is Node_Type.USER_QUESTION
            self.solution_trace: Dict[int, Dict[str, str]] = {
                0: {"user_question": user_question, "ost_step": {}}
            }
        else:
            assert self.node_type is not Node_Type.USER_QUESTION
            self.solution_trace = deepcopy(parent.solution_trace)

            if node_type is Node_Type.REPHRASED_USER_QUESTION:
                self.solution_trace[0]["user_question"] = rephrased_user_question
            elif node_type is Node_Type.DIRECT_ANSWER:
                assert self.subquestion_counter in self.solution_trace.keys()
                assert self.subquestion_counter == parent.subquestion_counter
                self.solution_trace[self.subquestion_counter]["direct_answer"] = {
                    "text": direct_answer,
                    "value": node_value,
                }
            elif node_type is Node_Type.SUBQUESTION:
                assert (
                    is_new_subquestion
                    and self.subquestion_counter == parent.subquestion_counter + 1
                )
                self.solution_trace[self.subquestion_counter] = {
                    "subquestion": subquestion,
                    "subanswer": {"text": subanswer, "value": node_value},
                    "ost_step": {},
                }
            elif node_type is Node_Type.RE_SUBANSWER:
                assert parent.subquestion is not None
                assert self.subquestion_counter == parent.subquestion_counter
                assert (
                    self.solution_trace[self.subquestion_counter]["subquestion"]
                    == parent.subquestion
                )
                self.solution_trace[self.subquestion_counter]["subanswer"] = {
                    "text": re_subanswer,
                    "value": node_value,
                }
            elif node_type is Node_Type.OST_STEP:
                assert (
                    "ost_step" in self.solution_trace[self.subquestion_counter].keys()
                )
                self.solution_trace[self.subquestion_counter]["ost_step"][
                    self.ost_step_counter
                ] = ost_step

    def __str__(self) -> str:
        type2str = {
            Node_Type.USER_QUESTION: "U",
            Node_Type.REPHRASED_USER_QUESTION: "RU",
            Node_Type.DIRECT_ANSWER: "DA",
            Node_Type.SUBQUESTION: "SQ",
            Node_Type.RE_SUBANSWER: "RS",
            Node_Type.OST_STEP: "TS",
        }
        return f"{type2str[self.node_type]}-{self.id}"

    def is_valid_leaf_node(self):
        #! a valid solution can only be in SUBQUESTION type or DIRECT_ANSWER type
        return (
            self.node_type is Node_Type.SUBQUESTION
            and reach_terminal_subquestion(self.subquestion, self.user_question)
        ) or self.node_type is Node_Type.DIRECT_ANSWER

    def is_valid_solution_node(self):
        #! a valid solution can only be in SUBQUESTION type or DIRECT_ANSWER type or OST_STEP type
        return (
            (
                self.node_type is Node_Type.SUBQUESTION
                and reach_terminal_subquestion(self.subquestion, self.user_question)
                # if the subquestion and already answer the question
            )
            or (
                self.node_type is Node_Type.OST_STEP
                and reach_terminal_ost_step(self.ost_step)
            )  # if the ost contain answer
            or self.node_type is Node_Type.DIRECT_ANSWER
        )

    def value(self) -> float:
        """
        Overview:
            The value of the current node.
        Returns:
            - output (:obj:`Int`): Current value, used to compute ucb score.
        """
        raise NotImplementedError

    def find_children(self, rollout_id: int):
        "All possible successors of this board state"
        raise NotImplementedError

    def is_terminal(self):
        "Returns True if the node has no children"
        raise NotImplementedError

    def calculate_reward(self):
        if self.is_valid_leaf_node():
            assert self.node_value is not None, breakpoint()
            return self.node_value
        else:
            return 0

    def skip_backprop(self):
        "If True, the reward of this node will not be accumulated in the backpropagation step."
        raise NotImplementedError
