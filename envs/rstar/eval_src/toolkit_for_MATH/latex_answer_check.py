# ---------------------------------------------------------
# Xwin-Math
# Copyright (c) 2023 Xwin-Math Team
# Licensed under The MIT License [see LICENSE for details]
# Based on ToRA (https://github.com/microsoft/ToRA/blob/main/src/eval/grader.py)
# Modified by Weiqi Wang
# ---------------------------------------------------------

import sys
sys.path.append(".")

from .parsing_lib import *

from typing import Union, Any

from copy import deepcopy
from math import isclose
from sympy import simplify, N
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.latex import parse_latex



def has_numbers(input_string: str) -> bool:
    """
    Checks if a string contains a number.
    """
    return any(char.isdigit() for char in input_string)


def has_structure(input_string: str) -> bool:
    """
    Checks if a string contains structured content.
    """
    if "(" in input_string or ")" in input_string or "[" in input_string or "]" in input_string or "\\" in input_string or "<" in input_string or ">" in input_string or "," in input_string or 'x' in input_string or 'y' in input_string or 'z' in input_string:
        return True
    return False


def sympy_parse(input_string: str) -> Any:
    """
    Parsing strings into mathematical expressions using sympy
    """
    for f in [parse_latex, parse_expr]:
        try:
            return f(input_string)
        except:
            pass
    return input_string


def symbolic_equal(a: str, b: str) -> Union[bool, None]:
    """
    Check if two strings are symbolic equal.
    """
    a = sympy_parse(a)
    b = sympy_parse(b)

    try:
        if simplify(a-b) == 0:
            return True
    except:
        pass

    try:
        if isclose(N(a), float(N(a)), rel_tol=1e-9) and isclose(N(a), float(N(a)), rel_tol=1e-9):
            return False
    except:
        pass

    try:
        if isclose(N(a), N(b), rel_tol=1e-3):
            return True
    except:
        pass
    return None


def convert_to_int(input_string: str) -> Union[int, None]:
    """
    Try to convert a string into int. Return `None` if an error occurs.
    """
    try:
        float_s = float(input_string)
        int_s = int(float_s)

        # If a floating-point number is converted to an integer that is very close to itself, then we consider it to be an integer.
        if isclose(int_s, float_s, rel_tol=1e-9):
            return int_s
        return None
    except:
        return None


def convert_to_float(input_string: str) -> Union[float, None]:
    """
    Try to convert a string into float. Return `None` if an error occurs.
    """
    try:
        float_s = float(input_string)
        return float_s
    except:
        return None


def numerical_equal(a: str, b: str) -> Union[bool, None]:
    """
    Check if two strings are numerical equal.
    """
    a_int = convert_to_int(a)
    b_int = convert_to_int(b)

    if a_int is not None and b_int is not None:
        return a_int == b_int

    a_float = convert_to_float(a)
    b_float = convert_to_float(b)

    if a_float is not None and b_float is not None:
        return isclose(a_float, b_float, rel_tol=1e-3)

    return None


def literal_check(model_generated_answer: str, ground_truth: str) -> Union[bool, None]:
    """
    Check if two strings are the same character by character
    """
    model_remove = deepcopy(model_generated_answer).replace(",", " ").replace(" ", "").replace(" ", "")
    gt_remove = deepcopy(ground_truth).replace(",", " ").replace(" ", "").replace(" ", "")

    if model_remove == gt_remove:
        return True

    if has_numbers(model_generated_answer) == False and has_numbers(ground_truth) == False:
        model_generated_answer = model_remove.strip("[]() ")
        ground_truth = gt_remove.strip("[]() ")
        if model_generated_answer == ground_truth:
            return True

    return None


def number_check(model_generated_answer: str, ground_truth: str) -> None:
    """
    Check if two strings have the same mathematical meaning.
    """
    if "," in model_generated_answer or "," in ground_truth:
        return None

    model_generated_answer = remove_prefix_and_suffix(remove_equals(model_generated_answer))
    ground_truth = remove_prefix_and_suffix(remove_equals(ground_truth))

    numerical_equal_result = numerical_equal(model_generated_answer, ground_truth)
    if numerical_equal_result is not None:
        return numerical_equal_result

    symbolic_equal_result = symbolic_equal(model_generated_answer, ground_truth)

    if symbolic_equal_result is not None:
        return symbolic_equal_result

    return None


def latex_answer_check(model_output, gt_answer, split=None, extract_policy: str="flex", eval_policy: str="aggressive"):
    assert gt_answer is not None
    assert len(gt_answer) > 0

    if model_output is None or model_output == "":
        return False

    # Step 1: Extract answer from response
    if split is not None:
        model_output = extract_answer(model_output, split, extract_policy = extract_policy)

    if model_output is None or model_output == "":
        return False

    # Step 2: Remove boxes and perform literal check
    # Compare strings character by character after simple processing including remove $%.
    # First we remove the boxes in the string but keeps the content
    # \boxed{\frac{13}{4}} --> \frac{13}{4}
    model_ans_norm = string_normalization(model_output)
    model_ans_norm_wo_boxes = remove_boxes_keep_content(model_ans_norm)
    gt_norm = string_normalization(gt_answer)
    gt_norm_wo_boxes = remove_boxes_keep_content(gt_norm)

    literal_check_result = literal_check(remove_prefix_and_suffix(model_ans_norm_wo_boxes), remove_prefix_and_suffix(gt_norm_wo_boxes))
    if literal_check_result is not None:
        return literal_check_result

    # Step 3: Attempt to parse -- single
    # Treat a string as a single number/extract a single number from a string and then compare.
    #
    # If we can accept a few mistakes, we try to extract numbers from the answers and compare them
    if eval_policy == "aggressive":
        # We wan't to use raw model_output to keep the $$
        # $13$ meters --> $13$ --> 13
        model_ans_num_lst = search_for_numbers(model_output)

        # We want the original answer has $$
        # This way we are able to consider the answer as a whole
        # We don't want \frac{13}{4} --> [13, 4] to be considered as 2 numbers
        if gt_answer[0] != "$" or gt_answer[-1] != "$":
            gt_num_lst = search_for_numbers("$" + gt_answer + "$")
        else:
            gt_num_lst = search_for_numbers(gt_answer)

        # We want to judge only those answers that contain only one number that represents the full meaning of the original string.
        # If the string still has LaTeX components or variables in addition to this number, then we believe that this number may not represent the meaning of the answer.
        # Here we must be really really careful.
        # x \\leq -5 vs. x \\geq -5
        # (-\\infty, 5) vs. (5, +\\infty)
        # TODO: We may have better methods to check if the numbers are simple enough
        if len(model_ans_num_lst) == 1 and len(gt_num_lst) == 1 and \
            not has_structure(model_output.replace(model_ans_num_lst[0], "")) and \
            not has_structure(gt_answer.replace(gt_num_lst[0], "")):

            model_num = remove_prefix_and_suffix(remove_boxes_keep_content(remove_text_box_only(model_ans_num_lst[0])))
            gt_num = remove_prefix_and_suffix(remove_boxes_keep_content(remove_text_box_only(gt_num_lst[0])))
            parse_result = number_check(model_num, gt_num)  #! problematic

            # As an additional method of judgment, even if it returns False we can't say that the answer is wrong, it could be caused by an unreasonable extraction of numbers
            if parse_result is True:
                return True

    # Here we do the same thing to the whole string
    model_wo_text = remove_prefix_and_suffix(model_ans_norm)
    gt_wo_text = remove_prefix_and_suffix(gt_norm)
    parse_result = number_check(model_wo_text, gt_wo_text)
    if parse_result is not None:
        return parse_result

    # If none of the above ways can determine whether the answer is correct or incorrect, then return incorrect
    return False
