"""
Wrapping the base environment with customised
update_legal_actions() function
"""
from heapq import merge
from typing import List, Dict, Tuple
import os
import numpy as np

from envs.MATH.env import CoTEnv
from envs.base_env import NoLegalActionException, ResetException
from enum import Enum, unique

from reason.inference.lm_call import LMCallingConfig
from .rstar_utils import *
from .eval_src.Evaluator import MATHEvaluator
from pathlib import Path

# Get the file path of the current script
CURRENT_DIR = Path(__file__).parent




class RStarEnv(CoTEnv):
    def __init__(
        self,
        config,
        math_problems,
        llm_gen_fn,
        task_desc_str: str,
        cot_example_str: str,
        problem_format_str: str,
        reset=True,
    ):
        """
        rStar call LLM inference in each of the nodes, checking current nodetype and do corresponding thinking
        OpenR apply LLM call in central Env entity (update_legal_action()), so we need to take some records...
        Args:
            config:
            math_problems:
            llm_gen_fn:
            task_desc_str:
            cot_example_str:
            problem_format_str:
            reset:
        """
        super().__init__(config,
                         math_problems,
                         llm_gen_fn,
                         task_desc_str,
                        cot_example_str,
                        problem_format_str,
                        reset)

        self.current_node_type = None
        self.disable_a1 = False
        self.disable_a5 = False
        self.enable_potential_score = False
        # potential score is disable due to https://github.com/zhentingqi/rStar/issues/12
        self.num_a1_steps = 3   # these are generator parameters
        self.mcts_num_last_votes = 32
        self.num_subquestions = 3
        self.num_votes = 10

        self.task_name = "MATH"
        self.sep = "\n\n"

        # loading template
        with open(os.path.join(
                CURRENT_DIR, f"prompts/{self.task_name}/decompose/decompose_template.json"),
                "r") as f:
            decompose_template = json.load(f)
            self.question_index = decompose_template["index"]

        self.decompose_prompt = read_txt(
            os.path.join(CURRENT_DIR, f"prompts/{self.task_name}/decompose/decompose_prompt.txt"))
        self.fewshot_cot_prompt = read_txt(
            os.path.join(CURRENT_DIR, f"prompts/{self.task_name}/fewshot_cot/fewshot_cot_prompt.txt"))
        self.fewshot_cot_config = read_json(
            os.path.join(CURRENT_DIR, f"prompts/{self.task_name}/fewshot_cot/fewshot_cot_config.json"))

        if not self.disable_a1:  # A1: Propose an one-step thought.
            self.fewshot_ost_prompt = read_txt(
                os.path.join(CURRENT_DIR, f"prompts/{self.task_name}/fewshot_ost/fewshot_ost_prompt.txt"))
            self.fewshot_ost_config = read_json(
                os.path.join(CURRENT_DIR, f"prompts/{self.task_name}/fewshot_ost/fewshot_ost_config.json"))

        if not self.disable_a5:  # A5: Rephrase the question/sub-question.
            self.rephrasing_prompt_template = read_txt(
                os.path.join(CURRENT_DIR, f"prompts/{self.task_name}/rephrasing_prompt_template.txt"))
            self.decompose_prompt_rephrased = read_txt(
                os.path.join(CURRENT_DIR, f"prompts/{self.task_name}/decompose/decompose_prompt_rephrased.txt"))
            self.fewshot_cot_prompt_rephrased = read_txt(
                os.path.join(CURRENT_DIR, f"prompts/{self.task_name}/fewshot_cot/fewshot_cot_prompt_rephrased.txt"))
            self.fewshot_ost_prompt_rephrased = read_txt(
                os.path.join(CURRENT_DIR, f"prompts/{self.task_name}/fewshot_cot/fewshot_ost_prompt_rephrased.txt"))

        # load evaluator
        self.evaluator = MATHEvaluator()

    def set_problem(self, idx):
        self.math_problem = self.math_problems[idx]

    def reset(self, update_legal_action=True):
        self.set_problem(idx=0)                     # retrive the first question set {'question': xxx, 'answer': xxx}
        self.action_history = []
        # self._init_query = self.build_query_str(
        #     cot_examples=self._cot_example_str,
        #     cot_task_desc=self._task_desc_str,
        #     problem_format_str=self._problem_format_str,
        #     problem_input=self.math_problem["question"],
        #     is_few_shot=self.is_few_shot,
        # )
        # self.user_question = self.math_problem['question']
        # self.expected_answer = self.math_problem['answer']

        # there are things we need to take records globally, as rStar store and pass them in nodes
        # self.current_node_type = Node_Type.USER_QUESTION        # record current node type
        # self.user_question_trace = []
        # self.solution_trace: Dict[int, Dict[str, str]] = {0: {"user_question": self.get_state(), "ost_step": {}}} # step id: step tuple
        # self.paraphrased = False                    # dynamically changed
        # self.parent_is_subquestion = False          # TODO(yan): in rStar this seems to be false alltime, need to check
        # self.user_question: str = self.get_state()

        # return self.user_question

    def try_update_legal_action(self, node):
        cnt = 0
        while cnt < 3:
            cnt += 1
            try:
                updated_node = (
                    self.update_legal_actions(node)
                )
                break
            except NoLegalActionException as e:
                if cnt == 3:
                    raise ResetException

        # info = {"api_completion_token": api_completion_token}
        return updated_node


    def update_legal_actions(self, current_node):
        """
        Think differently depending on current nodetype (status)
        Returns:

        """
        current_node_type = current_node.node_type

        # depending on type of current node, ask corresponding question
        if current_node_type is Node_Type.USER_QUESTION:
            # A1: Propose an one-step thought.
            if not current_node.disable_a1:
                self.do_action_generate_ost_step(current_node)

            # A2: Propose the remaining thought steps
            self.do_action_generate_direct_answers(current_node)


            # A3: Propose next sub-question along with its answer.
            self.do_action_generate_subquestions(current_node)


            # A5: Rephrase the question/sub-question.
            if not current_node.disable_a5:
                self.do_action_generate_rephrased_user_question(current_node)



        elif current_node_type is Node_Type.REPHRASED_USER_QUESTION:
            # A1: Propose an one-step thought.
            if not current_node.disable_a1:
                self.do_action_generate_ost_step(current_node)

            # A2: Propose the remaining thought steps
            self.do_action_generate_direct_answers(current_node)

            # A3: Propose next sub-question along with its answer.
            self.do_action_generate_subquestions(current_node)


        elif current_node_type is Node_Type.DIRECT_ANSWER:
            raise ValueError("DIRECT_ANSWER node cannot create children!!")

        elif current_node_type is Node_Type.SUBQUESTION:
            # A1: Propose an one-step thought.
            if not current_node.disable_a1:
                self.do_action_generate_ost_step(current_node)

            # A2: Propose the remaining thought steps
            self.do_action_generate_direct_answers(current_node)

            # A3: Propose next sub-question along with its answer.
            self.do_action_generate_subquestions(current_node)


            # A4: Answer the sub-question again.
            self.do_action_generate_re_subanswers(current_node)


        elif current_node_type is Node_Type.RE_SUBANSWER:
            # A1: Propose an one-step thought.
            if not current_node.disable_a1:
                self.do_action_generate_ost_step(current_node)


            # A2: Propose the remaining thought steps
            self.do_action_generate_direct_answers(current_node)


            # A3: Propose next sub-question along with its answer.
            self.do_action_generate_subquestions(current_node)


        elif current_node_type is Node_Type.OST_STEP:
            # A1: Propose an one-step thought.
            if not current_node.disable_a1:
                self.do_action_generate_ost_step(current_node)


            # A2: Propose the remaining thought steps
            self.do_action_generate_direct_answers(current_node)

        return current_node
        # total_completion_tokens = sum([i['completion_tokens'] for i in merge_action_dict.values()])


        # return merge_action_dict, total_completion_tokens


    def is_terminal(self, node):

        def is_valid_leaf_node(n):
            # ! a valid solution can only be in SUBQUESTION type or DIRECT_ANSWER type
            return (
                    n.node_type is Node_Type.SUBQUESTION and reach_terminal_subquestion(n.subquestion,
                                                                                           n.user_question)
            ) or n.node_type is Node_Type.DIRECT_ANSWER


        done = (node.depth >= node.max_depth_allowed) or is_valid_leaf_node(node)

        return done

    def do_action_generate_ost_step(self, node, parent_is_subquestion=False):
        """
        For current state, propose one-step thought, return legal action portion
        Args:
            parent_is_subquestion:

        Returns:

        """
        print(f"---- Generating one-step thought steps for node ...")

        #! ACTION: generate one-step thought step
        ost_step_list = []
        # formating
        if parent_is_subquestion:
            raise NotImplementedError
            #existing_ost_steps, next_ost_step_id = concat_subqs_subas_as_ost_steps(solution_trace)
        else:
            existing_ost_steps, next_ost_step_id = concat_ost_steps(node.solution_trace)

        io_input = (
            self.fewshot_ost_config["prompt_template"].format(
                examples=self.fewshot_ost_prompt if not node.paraphrased else self.fewshot_ost_prompt_rephrased,
                instruction=node.user_question,
            )
            + existing_ost_steps
            + f"Step {next_ost_step_id}:"
        )

        io_output_list = self.llm_gen_fn(
            input_str=io_input,
            config=LMCallingConfig(
                n=self.num_a1_steps,
                stop_str=["\n", "\n\n"],     # check stopping token
                max_tokens=256
            ),
        )
        ost_step_list = [io_output.strip() for io_output in io_output_list.text]
        logps_avg_by_len = io_output_list.logp_avg_by_len
        token_len = io_output_list.num_tokens
        finish_reason_list = io_output_list.finish_reason
        potential_answers_list = [None] * len(ost_step_list)
        completion_tokens = io_output_list.completion_tokens

        for ost_step, potential_answers in zip(ost_step_list, potential_answers_list):
            node.children.append(
                RstarLanguageNode(
                    parent=node,            # TODO[yan]: check over-nesting
                    depth=node.depth + 1,
                    node_type=Node_Type.OST_STEP,
                    ost_step=ost_step,
                )
            )


    def do_action_generate_direct_answers(self, node):
        print(f"---- Generating direct answers for node ...")
        # ! ACTION: generate direct answer for the user question (w/ or w/o hint)
        if (
                node.node_type is not Node_Type.USER_QUESTION
                and node.node_type is not Node_Type.REPHRASED_USER_QUESTION
        ):
            hint = make_hint(node.solution_trace, node.node_type)
        else:
            hint = None

        direct_answer_list, value_list = [], []

        num_return = self.mcts_num_last_votes
        fewshot_cot_prompt = self.fewshot_cot_prompt if not self.paraphrased else self.fewshot_cot_prompt_rephrased
        question = node.user_question + "\n\n" + hint if hint is not None else ""
        io_input = self.fewshot_cot_config["prompt_template"].format(examples=fewshot_cot_prompt,
                                                                     instruction = question)

        io_output_list = self.llm_gen_fn(
            input_str=io_input,
            config=LMCallingConfig(
                n=num_return,
                max_tokens=1024,
                stop_str=self.fewshot_cot_config["stop_tokens"],
            )
        )
        token_len = io_output_list.num_tokens
        finish_reason_list = io_output_list.finish_reason
        completion_tokens = io_output_list.completion_tokens

        cleaned_io_output_list = [io_output.strip() for io_output in io_output_list.text]  # ! cleaning

        try:
            assert len(cleaned_io_output_list) > 0

            if len(cleaned_io_output_list) == 1:
                most_likely_answer = cleaned_io_output_list[0]
                likelihood = 1
            else:
                _, most_likely_answer, _, likelihood = self.evaluator.find_most_confident_answer(
                    cleaned_io_output_list
                )
                assert likelihood>0

        except Exception as e:
            raise GeneratorError(
                source="generate direct answer from: few shot cot",
                io_input=io_input,
                io_output_list=cleaned_io_output_list,
            )

        direct_answer_list.append(most_likely_answer)
        value_list.append(likelihood)

        for direct_answer, value in zip(direct_answer_list, value_list):
            if np.isnan(value) or value <= 0:
                raise NotImplementedError
            node.children.append(
                RstarLanguageNode(
                    parent=node,
                    depth=node.depth + 1,
                    node_type=Node_Type.DIRECT_ANSWER,
                    node_value=value,
                    direct_answer=direct_answer,
                )
            )



    def do_action_generate_subquestions(self, node):
        print(f"---- Generating subquestions for node ...")

        subquestion_list, subanswer_list, value_list = [], [], []
        decompose_prompt = self.decompose_prompt if not node.paraphrased else node.decompose_prompt_rephrased

        # ! generate subquestions
        existing_subquestions_and_subanswers, next_subquestion_id = concat_subqs_and_subas(
            node.solution_trace, self.question_index
        )
        io_input = (
                decompose_prompt
                + "\n\n"
                + f"Question {self.question_index}: {node.user_question}"
                + "\n"
                + existing_subquestions_and_subanswers
                + f"Question {self.question_index}.{next_subquestion_id}:"
        )

        io_output_list = self.llm_gen_fn(
            input_str=io_input,
            config=LMCallingConfig(
                n=self.num_subquestions,
                max_tokens=128,
                stop_str=[
                "\n",
                "\n\n",
                "Answer",
                "Answer ",
                f"Answer {self.question_index}.{next_subquestion_id}",
                f"Answer {self.question_index}.{next_subquestion_id}:",
                f"Answer {self.question_index}.{next_subquestion_id}: ",
            ],
            )
        )

        subquestion_list = [o.strip() for o in io_output_list]

        # ! generate subanswers to the subquestions generated above
        io_input_list = []
        for subquestion in subquestion_list:
            io_input = (
                    decompose_prompt
                    + "\n\n"
                    + f"Question {self.question_index}: {node.user_question}"
                    + "\n"
                    + existing_subquestions_and_subanswers
                    + f"Question {self.question_index}.{next_subquestion_id}: "
                    + subquestion
                    + "\n"
                    + f"Answer {self.question_index}.{next_subquestion_id}:"
            )
            io_input_list.append(io_input)

        if reach_terminal_subquestion(subquestion=subquestion, user_question=node.user_question):
            num_return = self.mcts_num_last_votes
        else:
            num_return = self.num_votes

        io_output_list = self.llm_gen_fn(
            input_str=io_input_list,
            config=LMCallingConfig(
                n=num_return,
                max_tokens=512,
                stop_str=[
                "\n",
                "\n\n",
                f"Question {self.question_index}.{next_subquestion_id + 1}",
            ],
            )
        )
        cleaned_io_output_list = [
            [io_output.strip() for io_output in io_output_group] for io_output_group in io_output_list
        ]
        completion_tokens = io_output_list.completion_tokens


        for i, cleaned_io_output_group in enumerate(cleaned_io_output_list):
            try:
                most_likely_answer, likelihood = self._get_most_likely_answer(cleaned_io_output_group)
            except Exception as e:
                raise GeneratorError(
                    source="generate answer to subquestions",
                    io_input=io_input_list[i],
                    io_output_list=cleaned_io_output_group,
                )
            subanswer_list.append(most_likely_answer)
            value_list.append(likelihood)
        assert len(subquestion_list) == len(subanswer_list) == len(value_list)
        potential_answers_list = [None] * len(subquestion_list)

        for subquestion, subanswer, value, potential_answers in zip(
                subquestion_list, subanswer_list, value_list, potential_answers_list
        ):
            if np.isnan(value) or value <= 0:
                value = 0.01
                # breakpoint()
            node.children.append(
                RstarLanguageNode(
                    parent=node,
                    depth=node.depth + 1,
                    node_type=Node_Type.SUBQUESTION,
                    node_value=value,
                    subquestion=subquestion,
                    subanswer=subanswer,
                    is_new_subquestion=True,
                )
            )





    def do_action_generate_rephrased_user_question(self, node):
        print(f"---- Generating rephrased user question for node ...")

        rephrased_user_question_list = []
        io_input = self.rephrasing_prompt_template
        io_input += "\n\n"
        io_input += "Original Question: " + node.user_question + "\n"
        io_input += "Rephrased Question: Given a list of conditions, please answer the question. Condition 1: "
        io_output = self.llm_gen_fn(
            input_str=io_input,
            config=LMCallingConfig(
                n=1,
                max_tokens=512,
                stop_tokens=["\n", "\n\n"],
            )
        ).text
        token_len = io_output.num_tokens
        finish_reason_list = io_output.finish_reason
        completion_tokens = io_output.completion_tokens


        assert len(io_output)==1
        io_output = "Given a list of conditions, please answer the question. Condition 1: " + io_output
        rephrased_user_question_list.append(io_output)
        potential_answers_list = [None] * len(rephrased_user_question_list)

        # creating children
        for rephrased_user_question, potential_answers in zip(rephrased_user_question_list, potential_answers_list):
            node.children.append(
                RstarLanguageNode(
                    parent=node,
                    depth=node.depth + 1,
                    node_type=Node_Type.REPHRASED_USER_QUESTION,
                    rephrased_user_question=rephrased_user_question,
                )
            )



    def do_action_generate_re_subanswers(self, node):
        print(f"---- Generating re-subanswers for node ...")
        re_subanswer_list, value_list = [], []

        user_question_context, _ = split_user_question(node.user_question)

        last_subquestion_id = int(sorted(node.solution_trace.keys())[-1])
        last_subquestion = node.solution_trace[last_subquestion_id]["subquestion"]
        # ! few shot cot
        question = (
            f"{user_question_context} {last_subquestion}"
            if not node.paraphrased
            else f"{user_question_context} Question: {last_subquestion}"
        )
        fewshot_cot_prompt = self.fewshot_cot_prompt if not node.paraphrased else self.fewshot_cot_prompt_rephrased
        question += "\n\n"  # hint is None
        io_input = self.fewshot_cot_config["prompt_template"].format(examples=fewshot_cot_prompt, instruction=question)
        io_output_list = self.llm_gen_fn(
            input_str=io_input,
            config=LMCallingConfig(
                n=self.num_votes,
                max_tokens=self.max_tokens,
                stop_tokens=self.fewshot_cot_config["stop_tokens"],
            )
        )
        token_len = io_output_list.num_tokens
        finish_reason_list = io_output_list.finish_reason
        completion_tokens = io_output_list.completion_tokens

        cleaned_io_output_list = [io_output.strip() for io_output in io_output_list.text]  # ! cleaning
        try:
            most_likely_answer, likelihood = self._get_most_likely_answer(cleaned_io_output_list)
        except Exception as e:
            raise GeneratorError(
                source="generate re-subanswers: few shot cot",
                io_input=io_input,
                io_output_list=cleaned_io_output_list,
            )
        re_subanswer_list.append(most_likely_answer)
        value_list.append(likelihood)
        potential_answers_list = [None] * len(re_subanswer_list)

        # creating children
        for re_subanswer, value, potential_answers in zip(re_subanswer_list, value_list, potential_answers_list):
            if np.isnan(value) or value <= 0:
                breakpoint()
            node.children.append(
                RstarLanguageNode(
                    parent=node,        # check node chck node, do we need to pass children as well?
                    depth=node.depth + 1,
                    node_type=Node_Type.RE_SUBANSWER,
                    node_value=value,
                    re_subanswer=re_subanswer,
                )
            )


    def _get_most_likely_answer(self, io_output_list: List[str]) -> Tuple[str, float]:
        assert len(io_output_list) > 0

        if len(io_output_list) == 1:
            most_confident_answer_full_completion = io_output_list[0]
            confidence = 1
        else:
            _, most_confident_answer_full_completion, _, confidence = self.evaluator.find_most_confident_answer(
                io_output_list
            )
            assert confidence > 0

        return most_confident_answer_full_completion, confidence







