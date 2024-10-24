# ---------------------------------------------------------
# Xwin-Math
# Copyright (c) 2023 Xwin-Math Team
# Licensed under The MIT License [see LICENSE for details]
# Written by Weiqi Wang
# ---------------------------------------------------------

import sys
sys.path.append(".")

from eval_src.eval_MATH.parsing_lib import *

import re
import math



def cast_to_number(string):
    string = string.replace(":", "/")
    if '.' not in string and '/' not in string:
        return int(string)
    if '/' not in string and abs(int(float(string)) - float(string)) < 1e-10:
        return int(float(string))
    if '/' in string:
        return eval(string)
    return float(string)


def get_simple_numbers(string):
    # pattern_for_percentage = r"(\d+(.\d+)*%)"
    pattern_for_integer = r"-?\d+"
    pattern_for_comma = r"(-?\d{1,3}(,\d{3})+(.\d+)?)"
    pattern_for_float_in_fraction = r"(-?\d+(.\d+)?\s*[/:]\s*\d+(.\d+)?)"
    pattern_for_float_or_fraction = r"-?\d+\s*[\./:]\s*\d+"
    pattern_for_sci = r"(-?\d(.\d+)?[eE][+-]?\d+)"

    pattern_list = [
        pattern_for_sci,
        pattern_for_float_in_fraction,
        pattern_for_comma,
        pattern_for_float_or_fraction,
        pattern_for_integer,
    ]

    result_list = []

    for pattern in pattern_list:
        curr_pattern_result = re.findall(pattern, string)
        for item in curr_pattern_result:
            if isinstance(item, tuple) and item[0] != '':
                item = item[0]
            else:
                item = item
            item = item.replace(",", "")
            result_list.append(cast_to_number(item))
        string = re.sub(pattern, '', string)

    return result_list


def compare_numbers(number_1, number_2):
    if isinstance(number_1, int) and isinstance(number_2, int):
        return number_1 == number_2

    return math.isclose(number_1, number_2, rel_tol=1e-3)

def simple_answer_check(model_generated_answer, ground_truth, extract_policy, eval_policy, split):

    model_generated_answer = extract_answer(model_generated_answer, split=split, extract_policy=extract_policy)
    if model_generated_answer is None:
        return False
    model_generated_answer = model_generated_answer.rstrip('.').rstrip(' ').rstrip('\n').rstrip('.').rstrip(' ').rstrip('\n').replace("$","")


    try:
        ground_truth_number_list = get_simple_numbers(ground_truth)
        model_generated_answer_number_list = get_simple_numbers(model_generated_answer)
    except Exception as e:
        return False

    if len(ground_truth_number_list) == 0 and len(model_generated_answer_number_list) == 0:
        return model_generated_answer.replace("\n","").replace(" ","").lower() == ground_truth.replace("\n","").replace(" ","").lower()

    if len(model_generated_answer_number_list) == 0 or len(ground_truth_number_list) == 0:
        return False

    if eval_policy == "strict" and len(ground_truth_number_list) != len(model_generated_answer_number_list):
        return False

    while len(ground_truth_number_list) != 0 and len(model_generated_answer_number_list) != 0:
        matched = False
        for ground_truth_number in ground_truth_number_list:
            for model_generated_answer_number in model_generated_answer_number_list:
                if compare_numbers(model_generated_answer_number, ground_truth_number):
                    model_generated_answer_number_list.remove(model_generated_answer_number)
                    ground_truth_number_list.remove(ground_truth_number)
                    matched = True
                    break
            if matched:
                break
        if matched:
            continue
        else:
            break

    if eval_policy == "strict":
        if len(ground_truth_number_list) == 0 and len(model_generated_answer_number_list) == 0:
            return True
        return False
    elif eval_policy == "model_include_gt":
        if len(ground_truth_number_list) == 0:
            return True
        return False
    elif eval_policy == "gt_include_model":
        if len(model_generated_answer_number_list) == 0:
            return True
        return False