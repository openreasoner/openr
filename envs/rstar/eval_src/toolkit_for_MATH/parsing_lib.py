# ---------------------------------------------------------
# Xwin-Math
# Copyright (c) 2023 Xwin-Math Team
# Licensed under The MIT License [see LICENSE for details]
# Written by Weiqi Wang
# ---------------------------------------------------------

from pyparsing import *
from typing import List
import os


def extract_answer(
    model_generated_answer: str, split: List[str], extract_policy: str
) -> str:
    """
    Extract answer from response.
    """
    model_generated_answer_split_list = []
    for split_item in split:
        if model_generated_answer.find(split_item) != -1:
            model_generated_answer_split_list.append(
                model_generated_answer.split(split_item)[-1]
            )

    shortest = None
    if len(model_generated_answer_split_list) != 0:
        for ans in model_generated_answer_split_list:
            if shortest is None or len(ans) < len(shortest):
                shortest = ans

        return shortest

    if extract_policy == "flex":
        boxes = search_for_boxes(model_generated_answer)
        if len(boxes) != 0:
            return boxes[-1]

        numbers = search_for_numbers(model_generated_answer)
        if len(numbers) != 0:
            return numbers[-1]

    return None


def remove_prefix_and_suffix(string: str) -> str:
    """
    Remove unnecessary prefixes and suffixes from the input strings
    """
    return string.strip(" \n").rstrip(".").strip(" \n").rstrip(".")


def string_normalization(string: str) -> str:
    """
    Normalize strings, convert to lowercase and remove or replace special symbols
    """
    return (
        string.replace("\\$", "")
        .replace("$", "")
        .replace("\\%", "")
        .replace("%", "")
        .replace("^\\circ", "")
        .replace("^{\\circ}", "")
        .replace("\u03c0", "\\pi")
        .replace("{,}", "")
        .replace("\u00b0", "")
        .lower()
    )


def search_for_intervals(input: str) -> str:
    """
    Extract the interval in the answer
    For example:
        The answer is: (-\infty, 0) \cup (1, +\infty) and [1, 2]. --> ['(-\infty, 0) \cup (1, +\infty)', '[1, 2]']
    """
    latex_objects = Word(alphanums + "+ - * / ^ | % $ ! \\ { }")
    left_bracket = oneOf("[ (")
    right_bracket = oneOf("] )")
    space = Optional(" ")
    interval = (
        left_bracket
        + latex_objects
        + space
        + ","
        + space
        + latex_objects
        + right_bracket
    )
    intervals = Combine(
        delimitedList(interval, delim=space + "\\cup" + space, combine=True)
    )

    result = intervals.searchString(input)
    return [_[0] for _ in result]


def search_for_joint_element_with_bracket(input):
    """
    Extract parallel elements wrapped in parentheses.
    For example:

    """
    nestedParens = Forward()
    nestedBraces = Forward()
    nestedSquareBrackets = Forward()

    nestedParens << (
        "("
        + ZeroOrMore(
            nestedParens | nestedBraces | nestedSquareBrackets | CharsNotIn("()")
        )
        + ")"
    )
    nestedBraces << (
        "{"
        + ZeroOrMore(
            nestedParens | nestedBraces | nestedSquareBrackets | CharsNotIn("{}")
        )
        + "}"
    )
    nestedSquareBrackets << (
        "["
        + ZeroOrMore(
            nestedParens | nestedBraces | nestedSquareBrackets | CharsNotIn("[]")
        )
        + "]"
    )

    parser = nestedParens | nestedBraces | nestedSquareBrackets

    result_list = parser.searchString(input)
    filtered_result_list = []

    for result_item in result_list:
        result_item_str = "".join(result_item)
        if "," in result_item_str:
            filtered_result_list.append(result_item_str)

    return filtered_result_list


def search_for_joint_elements_without_bracket(input):
    if "," in input:
        return ["{" + input + "}"]

    return []


def remove_commas_from_integers(input):
    comma = ","
    not_comma = ~Literal(comma)
    digit_first = Word(nums, max=3)
    digit_others = Word(nums, min=3, max=3)
    sign = Optional(Word("+-", exact=1))

    number_with_comma = Combine(
        sign + digit_first + OneOrMore(comma + digit_others) + not_comma
    )("number")

    def replace_commas(tokens):
        return "".join(tokens[0].split(","))

    number_with_comma.setParseAction(replace_commas)
    return number_with_comma.transformString(input)


def search_for_boxes(input):
    element = originalTextFor(nestedExpr("{", "}"))
    parser = Literal("\\boxed") + element | Literal("\\mbox") + element
    results = parser.searchString(input)
    return [_[1][1:-1] for _ in results]


def search_for_numbers(input):
    integer = Word("-" + nums, nums)
    fraction = Combine(Word("-" + nums, nums) + "/" + Word(nums))
    decimal = Combine(Optional(Word("-" + nums, nums)) + "." + Word(nums))
    scientific = Combine(Word("-" + nums, nums) + "e" + Word("-" + nums, nums))
    latex = Combine(Suppress("$") + SkipTo("$") + Suppress("$"))
    number_with_comma = Combine(
        Optional(Word("+-", exact=1))
        + Word(nums, max=3)
        + OneOrMore(Suppress(",") + Word(nums, min=3, max=3))
        + Suppress(~Literal(","))
    )

    parser = latex | scientific | fraction | decimal | number_with_comma | integer

    return [_[0] for _ in parser.searchString(input)]


def remove_text_box_only(input):
    tex_expr = Literal("\\text") + nestedExpr("{", "}") + Optional(
        "^" + Char(nums)
    ) | Literal("\\mbox") + nestedExpr("{", "}") + Optional("^" + Char(nums))
    return "".join(tex_expr.suppress().transformString(input))


def remove_boxes_keep_content(input_string):
    box_types = ["\\box", "\\boxed", "\\mbox", "\\text"]
    for box in box_types:
        while True:
            start = input_string.find(box)
            if start == -1:  # box type not found
                break
            brace_start = input_string.find("{", start)
            if brace_start == -1:  # "{" not found after box type
                break
            brace_count = 0
            for i in range(brace_start, len(input_string)):
                if input_string[i] == "{":
                    brace_count += 1
                elif input_string[i] == "}":
                    brace_count -= 1
                if brace_count == 0:  # matching "}" found
                    brace_end = i
                    break
            else:  # matching "}" not found
                break
            # remove box type but keep the content inside
            input_string = (
                input_string[:start]
                + input_string[brace_start + 1 : brace_end]
                + input_string[brace_end + 1 :]
            )
    return input_string


def remove_equals(input_string):
    if "=" in input_string and len(input_string.split("=")) == 2:
        left, right = input_string.split("=")
        if (
            right.strip(" \n").rstrip(".").strip(" \n").rstrip(".") == "0"
            and len(left.strip(" \n").rstrip(".").strip(" \n").rstrip(".")) >= 2
        ):
            return left
        else:
            return right
    return input_string


def add_prefix_to_filename(path, prefix):
    dirname, filename = os.path.split(path)
    new_filename = prefix + "_" + filename
    new_path = os.path.join(dirname, new_filename)
    return new_path
