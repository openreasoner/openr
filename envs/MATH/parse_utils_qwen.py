import random
import regex
import re
import sympy
from latex2sympy2 import latex2sympy
from typing import TypeVar, Iterable, List, Union, Any, Dict
from word2number import w2n


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _fix_sqrt(string):
    _string = re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)
    return _string


def convert_word_number(text: str) -> str:
    try:
        text = str(w2n.word_to_num(text))
    except:
        pass
    return text


# units mainly from MathQA
unit_texts = [
    "east",
    "degree",
    "mph",
    "kmph",
    "ft",
    "m sqaure",
    " m east",
    "sq m",
    "deg",
    "mile",
    "q .",
    "monkey",
    "prime",
    "ratio",
    "profit of rs",
    "rd",
    "o",
    "gm",
    "p . m",
    "lb",
    "tile",
    "per",
    "dm",
    "lt",
    "gain",
    "ab",
    "way",
    "west",
    "a .",
    "b .",
    "c .",
    "d .",
    "e .",
    "f .",
    "g .",
    "h .",
    "t",
    "a",
    "h",
    "no change",
    "men",
    "soldier",
    "pie",
    "bc",
    "excess",
    "st",
    "inches",
    "noon",
    "percent",
    "by",
    "gal",
    "kmh",
    "c",
    "acre",
    "rise",
    "a . m",
    "th",
    "π r 2",
    "sq",
    "mark",
    "l",
    "toy",
    "coin",
    "sq . m",
    "gallon",
    "° f",
    "profit",
    "minw",
    "yr",
    "women",
    "feet",
    "am",
    "pm",
    "hr",
    "cu cm",
    "square",
    "v â € ™",
    "are",
    "rupee",
    "rounds",
    "cubic",
    "cc",
    "mtr",
    "s",
    "ohm",
    "number",
    "kmph",
    "day",
    "hour",
    "minute",
    "min",
    "second",
    "man",
    "woman",
    "sec",
    "cube",
    "mt",
    "sq inch",
    "mp",
    "∏ cm ³",
    "hectare",
    "more",
    "sec",
    "unit",
    "cu . m",
    "cm 2",
    "rs .",
    "rs",
    "kg",
    "g",
    "month",
    "km",
    "m",
    "cm",
    "mm",
    "apple",
    "liter",
    "loss",
    "yard",
    "pure",
    "year",
    "increase",
    "decrease",
    "d",
    "less",
    "Surface",
    "litre",
    "pi sq m",
    "s .",
    "metre",
    "meter",
    "inch",
]

unit_texts.extend([t + "s" for t in unit_texts])


def strip_string(string, skip_unit=False):
    string = str(string).strip()
    # linebreaks
    string = string.replace("\n", "")

    # right "."
    string = string.rstrip(".")

    # remove inverse spaces
    # replace \\ with \
    string = string.replace("\\!", "")
    # string = string.replace("\\ ", "")
    # string = string.replace("\\\\", "\\")

    # matrix
    string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
    string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string)
    string = string.replace("bmatrix", "pmatrix")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = (
        string.replace("\\neq", "\\ne")
        .replace("\\leq", "\\le")
        .replace("\\geq", "\\ge")
    )

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("\\{", "{")
    string = string.replace("\\}", "}")

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        string = _string

    if not skip_unit:
        # Remove unit: texts
        for _ in range(2):
            for unit_text in unit_texts:
                # use regex, the prefix should be either the start of the string or a non-alphanumeric character
                # the suffix should be either the end of the string or a non-alphanumeric character
                _string = re.sub(r"(^|\W)" + unit_text + r"($|\W)", r"\1\2", string)
                if _string != "":
                    string = _string

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")
    string = string.replace("\\(", "").replace("\\)", "")

    # convert word number to digit
    string = convert_word_number(string)

    # replace "\\text{...}" to "..."
    string = re.sub(r"\\text\{(.*?)\}", r"\1", string)
    for key in ["x=", "y=", "z=", "x\\in", "y\\in", "z\\in", "x\\to", "y\\to", "z\\to"]:
        string = string.replace(key, "")
    string = string.replace("\\emptyset", r"{}")
    string = string.replace("(-\\infty,\\infty)", "\\mathbb{R}")

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace("%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    # cdot
    # string = string.replace("\\cdot", "")
    if (
        string.startswith("{")
        and string.endswith("}")
        and string.isalnum()
        or string.startswith("(")
        and string.endswith(")")
        and string.isalnum()
        or string.startswith("[")
        and string.endswith("]")
        and string.isalnum()
    ):
        string = string[1:-1]

    # inf
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    # and
    string = string.replace("and", "")
    string = string.replace("\\mathbf", "")

    # use regex to remove \mbox{...}
    string = re.sub(r"\\mbox{.*?}", "", string)

    # quote
    string.replace("'", "")
    string.replace('"', "")

    # i, j
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    # replace a.000b where b is not number or b is end, with ab, use regex
    string = re.sub(r"(\d+)\.0*([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0*$", r"\1", string)

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string

direct_answer_trigger_for_fewshot = ("choice is", "answer is")


def choice_answer_clean(pred: str):
    pred = pred.strip("\n")

    # Determine if this is ICL, if so, use \n\n to split the first chunk.
    ICL = False
    for trigger in direct_answer_trigger_for_fewshot:
        if pred.count(trigger) > 1:
            ICL = True
    if ICL:
        pred = pred.split("\n\n")[0]

    # Split the trigger to find the answer.
    preds = re.split("|".join(direct_answer_trigger_for_fewshot), pred)
    if len(preds) > 1:
        answer_flag = True
        pred = preds[-1]
    else:
        answer_flag = False

    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")

    # Clean the answer based on the dataset
    tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip(".")]

    if len(pred) == 0:
        pred = ""
    else:
        if answer_flag:
            # choose the first element in list ...
            pred = pred[0]
        else:
            # choose the last e
            pred = pred[-1]

    # Remove the period at the end, again!
    pred = pred.rstrip(".").rstrip("/")

    return pred


def find_box(pred_str: str):
    ans = pred_str.split("boxed")[-1]
    if not ans:
        return ""
    if ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()
    return a


def clean_units(pred_str: str):
    """Clean the units in the number."""

    def convert_pi_to_number(code_string):
        code_string = code_string.replace("\\pi", "π")
        # Replace \pi or π not preceded by a digit or } with 3.14
        code_string = re.sub(r"(?<![\d}])\\?π", "3.14", code_string)
        # Replace instances where π is preceded by a digit but without a multiplication symbol, e.g., "3π" -> "3*3.14"
        code_string = re.sub(r"(\d)(\\?π)", r"\1*3.14", code_string)
        # Handle cases where π is within braces or followed by a multiplication symbol
        # This replaces "{π}" with "3.14" directly and "3*π" with "3*3.14"
        code_string = re.sub(r"\{(\\?π)\}", "3.14", code_string)
        code_string = re.sub(r"\*(\\?π)", "*3.14", code_string)
        return code_string

    pred_str = convert_pi_to_number(pred_str)
    pred_str = pred_str.replace("%", "/100")
    pred_str = pred_str.replace("$", "")
    pred_str = pred_str.replace("¥", "")
    pred_str = pred_str.replace("°C", "")
    pred_str = pred_str.replace(" C", "")
    pred_str = pred_str.replace("°", "")
    return pred_str


def extract_theoremqa_answer(pred: str, answer_flag: bool = True):
    if any([option in pred.lower() for option in ["yes", "true"]]):
        pred = "True"
    elif any([option in pred.lower() for option in ["no", "false"]]):
        pred = "False"
    elif any(
        [
            option in pred.lower()
            for option in ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
        ]
    ):
        pass
    else:
        # Some of the models somehow get used to boxed output from pre-training
        if "boxed" in pred:
            pred = find_box(pred)

        if answer_flag:
            # Extract the numbers out of the string
            pred = pred.split("=")[-1].strip()
            pred = clean_units(pred)
            try:
                tmp = str(latex2sympy(pred))
                pred = str(eval(tmp))
            except Exception:
                if re.match(r"-?[\d\.]+\s\D+$", pred):
                    pred = pred.split(" ")[0]
                elif re.match(r"-?[\d\.]+\s[^\s]+$", pred):
                    pred = pred.split(" ")[0]
        else:
            # desparate search over the last number
            preds = re.findall(r"-?\d*\.?\d+", pred)
            if len(preds) >= 1:
                pred = preds[-1]
            else:
                pred = ""

    return pred

def check_boxed(pred_str):
    ans = pred_str.split("boxed")[-1]
    if len(ans) == 0:
        return ""
    elif ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()

    # if the answer is a equation
    if "=" in a:
        a = a.split('=')[-1]

    return a

def extract_answer(pred_str, data_name, use_last_number=True):
    pred_str = pred_str.replace("\u043a\u0438", "")

    if "final answer is $" in pred_str and "$. I hope" in pred_str:
        # minerva_math
        tmp = pred_str.split("final answer is $", 1)[1]
        pred = tmp.split("$. I hope", 1)[0].strip()
        if "boxed" in pred:         # llama3 case, check boxed
            pred = check_boxed(pred)
    elif "final answer is" in pred_str and ". I hope" in pred_str:      # llama3-8b-instruct case
        tmp = pred_str.split("final answer is", 1)[1]
        pred = tmp.split(". I hope", 1)[0].strip()
    elif "final answer is" in pred_str:
        tmp = pred_str.split("final answer is", 1)[1]
        pred = tmp.strip()
    elif "boxed" in pred_str:
        pred = check_boxed(pred_str)
    elif "he answer is" in pred_str:
        pred = pred_str.split("he answer is")[-1].strip()
    elif "final answer is" in pred_str:
        pred = pred_str.split("final answer is")[-1].strip()
    elif "答案是" in pred_str:
        # Handle Chinese few-shot multiple choice problem answer extraction
        pred = pred_str.split("答案是")[1].strip().split("\n\n")[0].strip()
    else:  # use the last number
        if use_last_number:
            pattern = "-?\d*\.?\d+"
            pred = re.findall(pattern, pred_str.replace(",", ""))
            if len(pred) >= 1:
                pred = pred[-1]
            else:
                pred = ""
        else:
            pred = ""

    # multiple line
    # pred = pred.split("\n")[0]
    pred = re.sub(r"\n\s*", "", pred)
    if pred != "" and pred[0] == ":":
        pred = pred[1:]
    if pred != "" and pred[-1] == ".":
        pred = pred[:-1]
    if pred != "" and pred[-1] == "/":
        pred = pred[:-1]
    pred = strip_string(pred, skip_unit=data_name in ["carp_en", "minerva_math"])
    return pred


STRIP_EXCEPTIONS = ["carp_en", "minerva_math"]


def parse_ground_truth(groudtruth_solution: str, data_name):
        gt_ans = extract_answer(groudtruth_solution, data_name)
        return gt_ans


def parse_question(example, data_name):
    question = ""
    if data_name == "asdiv":
        question = f"{example['body'].strip()} {example['question'].strip()}"
    elif data_name == "svamp":
        body = example["Body"].strip()
        if not body.endswith("."):
            body = body + "."
        question = f'{body} {example["Question"].strip()}'
    elif data_name == "tabmwp":
        title_str = (
            f'regarding "{example["table_title"]}" ' if example["table_title"] else ""
        )
        question = f"Read the following table {title_str}and answer a question:\n"
        question += f'{example["table"]}\n{example["question"]}'
        if example["choices"]:
            question += (
                f' Please select from the following options: {example["choices"]}'
            )
    elif data_name == "carp_en":
        question = example["content"]
    elif data_name == "mmlu_stem":
        options = example["choices"]
        assert len(options) == 4
        for i, (label, option) in enumerate(zip("ABCD", options)):
            options[i] = f"({label}) {str(option).strip()}"
        options = " ".join(options)
        # question = f"{example['question'].strip()}\nWhat of the following is the right choice? Explain your answer.\n{options}"
        question = f"{example['question'].strip()}\nAnswer Choices: {options}"
    elif data_name == "sat_math":
        options = example["options"].strip()
        assert "A" == options[0]
        options = "(" + options
        for ch in "BCD":
            if f" {ch}) " in options:
                options = regex.sub(f" {ch}\) ", f" ({ch}) ", options)
        # question = f"{example['question'].strip()}\nWhat of the following is the right choice? Explain your answer.\n{options.strip()}"
        question = f"{example['question'].strip()}\nAnswer Choices: {options}"
    elif "aqua" in data_name:
        options = example["options"]
        choice = "(" + "(".join(options)
        choice = choice.replace("(", " (").replace(")", ") ").strip()
        choice = "\nAnswer Choices: " + choice
        question = example["question"].strip() + choice
    elif data_name == "gaokao_math_qa":
        options_dict = example["options"]
        options = []
        for key in options_dict:
            options.append(f"({key}) {options_dict[key]}")
        options = " ".join(options)
        question = f"{example['question'].strip()}\n选项: {options}"
    else:
        for key in ["question", "problem", "Question", "input"]:
            if key in example:
                question = example[key]
                break
    # assert question != ""
    # Yes or No question
    _, gt_ans = parse_ground_truth(example, data_name)
    if isinstance(gt_ans, str):
        gt_lower = gt_ans.lower()
        if gt_lower in ["true", "false"]:
            question += " (True or False)"
        if gt_lower in ["yes", "no"]:
            question += " (Yes or No)"
    return question.strip()

def extract_groundtruth(groundtruth_str: str) -> str:
    return parse_ground_truth(groundtruth_str, data_name='math')


if __name__ == "__main__":
    # run examples
    # test_text = "n\nLet's figure out how much the math club made:\n\\begin{align*} \n &\\text{Cookies: } \\frac{54}{3}=18 \\text{ sets of } 3 cookies, \\text{ so } 18 \\cdot \\$1 = \\$18. \\\\\n &\\text{Cupcakes: } 20 \\cdot \\$2 = \\$40. \\\\\n &\\text{Brownies: } 35 \\cdot \\$1 = \\$35. \\\\\n &\\text{Total: } \\$18 + \\$40 + \\$35 = \\$93.\n\\end{align*}The math club spent $\\$15$to bake these items, so their profit is $\\boxed{\\$93-\\$15 = \\$78}.$\n\n"
    # test_text = "\nStep 1: List all the possible ways to insert parentheses. There are 6 possible ways to insert 3 pairs of parentheses into 4 terms. The 6 ways are: \\begin{align*}\n&(2\\cdot 3)\\cdot(4\\cdot 5)+1\\\\\n&2\\cdot(3\\cdot 4)\\cdot (5+1)\\\\\n&(2\\cdot 3)\\cdot(4\\cdot (5+1))\\\\\n&(2\\cdot (3\\cdot 4)\\cdot 5)+1\\\\\n&2\\cdot (3\\cdot (4\\cdot 5))+1\\\\\n&(2\\cdot ((3\\cdot 4)\\cdot 5))+1\n\\end{align*} Step \"\nFinal Answer: The final answer is 6. I hope it is correct.\n"
    # test_text = "\nWe can simplify $f(x)$ as follows:\n$f(x) = \\frac{3x-2}{x-2} = \\frac{3x-2}{x-2} \\cdot \\frac{x+2}{x+2} = \\frac{3x^2 - 2x - 6}{x^2-4} = 3 - \\frac{8}{x^2-4}$.\nThen, we can find the values of $f(-2)$, $f(-1)$, and $f(0)$:\n$f(-2) = 3 - \\frac{8}{(-2)^2-4} = 3 - \\frac{8}{4} = 1$,\n$f(-1) = 3 - \\frac{8}{(-1)^2-4} = 3 - \\frac{8}{1-4} = 3 - (-8) = 11$,\n$f(0) = 3 - \\frac{8}{0^2-4}$ is undefined, because the denominator is zero.\nSo, the sum of the values is: $\\boxed{\\frac{1 + 11}{1} = 12}$.\n"
    # test_text = "\nLet $n = 7k+2$ for some integer $k$. Then we have\n\\begin{align*}\n(n+2)(n+4)(n+6) & = (7k+2+2)(7k+2+4)(7k+2+6) \\\\\n& = (7k+4)(7k+6)(7k+8)\n\\end{align*}By Fermat's Little Theorem, $7^3 \\equiv 1 \\pmod{7}$. Therefore, we have\n$$(7k+4)(7k+6)(7k+8) = 7k(7k+1)(7k+2)$$ $$\\equiv k(k+1)(k+2) \\pmod{7}$$ $$\\equiv k(k+1)(k+3) \\pmod{7}$$ $$\\equiv k(k+1)(-1) \\pmod{7}$$ $$\\equiv -k(k+1) \\pmod{7}$$ $$\\equiv -(k^2+k) \\pmod{7}$$ $$\\equiv -(k^2+k-1+1) \\pmod{7}$$ $$\\equiv -(k^2+k-1)+1 \\pmod{7}$$ $$\\equiv -(k^2+k) + (k-1) + 1 \\pmod{7}$$ $$\\equiv (-k)(k-1) + (k-1) + 1 \\pmod{7}$$ $$\\equiv -k^2+k+k-1+1 \\pmod{7}$$ $$\\equiv -k^2+2k \\pmod{7}$$ $$\\equiv -k^2+2k-2+2 \\pmod{7}$$ $$\\equiv -(k^2+2k-2) + 2 \\pmod{7}$$ $$\\equiv -(k+1)^2+1+2 \\pmod{7}$$ $$\\equiv -(k+1)^2+3 \\pmod{7}$$ $$\\equiv -(k+1)(k+1)+3 \\pmod{7}$$ $$\\equiv -(k+1)(k+1)+2+1 \\pmod{7}$$ $$\\equiv -(k+1)(k+1)+(k+1)+1 \\pmod{7}$$ $$\\equiv -((k+1)^2+(k+1)) + 1 \\pmod{7}$$ $$\\equiv -((k+1)(k+1)+k+1) + 1 \\pmod{7}$$ $$\\equiv -((k+1)(k+1)+k+1-1+1) \\pmod{7}$$ $$\\equiv -((k+1)(k+1)+k) + 2 \\pmod{7}$$ $$\\equiv -((k+1)(k+1)-k)+2 \\pmod{7}$$ $$\\equiv -(k+1)^2+2 \\pmod{7}$$ $$\\equiv -(k+1)^2+2-2+2 \\pmod{7}$$ $$\\equiv -(k+1)^2+0+2 \\pmod{7}$$ $$\\equiv -(k+1)^2+2 \\pmod{7}$$ $$\\equiv -1^2+2 \\pmod{7}$$ $$\\equiv -1+2 \\pmod{7}$$ $$\\equiv 1 \\pmod{7}$$ $$\\boxed{\\equiv 1}$$\nFinal Answer: The final answer is 1. I hope it is correct.\n"
    # test_text = "\\boxed{\\text{Step 1:} }  x^6 - 3 = x^6 + 0x^5 + 0x^4 + 0x^3 + 0x^2 + 0x - 3. \n\\boxed{\\text{Step 2:} } The dividend has been written in descending powers of $x$. Let us long divide.\n\\boxed{\\text{Step 3:} } \\begin{array}{r} x^5- x^4+x^3-x^2+x-1 \\\\ x + 1 \\enclose{longdiv}{ x^6-0x^5+0x^4+0x^3+0x^2-3} \\\\ \\underline{x^6+ x^5} \\\\ -x^5+0x^4+0x^3+0x^2-3 \\\\ \\underline{-x^5- x^4} \\\\ x^4+0x^3+0x^2-3 \\\\ \\underline{x^4+ x^3} \\\\ -x^3+0x^2-3 \\\\ \\underline{-x^3- x^2} \\\\ x^2-3 \\\\ \\underline{x^2+ x} \\\\ -x-3 \\\\ \\underline{-x-1} \\\\ -2 \\end{array} \n\\boxed{\\text{Step 4:} }  The quotient is $x^5- x^4+x^3-x^2+x-1$. The remainder is $-2$. \n\\boxed{\\text{Final Answer: } x^5- x^4+x^3-x^2+x-1}\n"
    # test_text = "\\step{Step 1}\nWe are given that $2^8=4^x$. To solve for $x$, we can rewrite $4$ as $2^2$. \n$2^8=(2^2)^x$\nWe can now rewrite the equation as:\n$\\boxed{(2^2)^x=2^8}$\n\\step{Step 2}\nWe know that the exponent of $2$ is equal on both sides of the equation, so we can equate them:\n$2^x=2^8$\nTo solve for $x$, we can take the logarithm base 2 of both sides:\n$\\log_22^x=\\log_22^8$\nUsing the property of logarithms that $\\log_ab^x=x$, we get:\n$x\\log_22=8\\log_22$\nSimplifying:\n$x=8$\n\\step{Final Answer}\nThe final answer is $\\boxed{x=8}$. I hope it is correct.\n"
    # test_text = "\n\\begin{step} \"Step 1: Split the equation into two cases\"\nThe equation $|5x-1|=x+3$ can be written as either $5x-1=x+3$ or $-(5x-1)=x+3$. The first equation becomes $4x=4$, which has the solution $x=1$. The second equation becomes $-5x+1=x+3$, which has the solution $x=-2$. So we have two solutions, $x=1$ and $x=-2$.\n\\end{step}\n\\begin{step} \"Step 2: Determine the larger solution\"\nThe solution $x=1$ is greater than $x=-2$, so it is the larger solution.\n\\end{step}\n\\begin{boxed} $\\boxed{x=1}$ \\end{boxed}\nFinal Answer: The final answer is 1. I hope it is correct.\n"
    # test_text = "\\boxed{\\text{Step 1:}}\nThere are $6\\times 6=36$ possible outcomes for the two dice.  The product of the numbers on the dice can only be a multiple of 5 if one of the numbers is 5 and the other number is 1, or if both numbers are 5.  There are $2\\times1=2$ ways to get the first outcome, and one way to get the second outcome.\n\\boxed{\\text{Step 2:}}Therefore, there are 3 ways for the product to be a multiple of 5.  The probability of the product being a multiple of 5 is then \\begin{align*}\n\\frac{3}{36}&=\\frac{1}{12}.\n\\end{align*}\nFinal Answer: The final answer is \\frac{1}{12}.\n"
    test_text = "}\\boxed{\\text{Step 1: Count the number of ways to choose 3 boys}}\\text{Mr. Brennan has 7 boys and he needs to choose 3 of them. We can do this in }\\binom{7}{3}=\\frac{7!}{3!(7-3)!}=\\frac{7!}{3!4!}=\\frac{7\\cdot6\\cdot5}{3\\cdot2\\cdot1}=35\\text{ ways.}\\text{Step }\"\n\\}\\boxed{\\text{Step 2: Count the number of ways to choose 2 girls}}\\text{Mr. Brennan has 4 girls and he needs to choose 2 of them. We can do this in }\\binom{4}{2}=\\frac{4!}{2!(4-2)!}=\\frac{4!}{2!2!}=\\frac{4\\cdot3}{2\\cdot1}=6\\text{ ways.}\\text{Step }\"\n\\}\\boxed{\\text{Step 3: Combine the results}}\\text{The total number of ways Mr. Brennan can pick 3 boys and 2 girls is the product of the number of ways to choose each set. Therefore, the total number of ways is }35\\cdot 6=210.\\text{Step }\"\nFinal Answer: The final answer is 210. I hope it is correct.\n"

    # true_text = "To find the profit, we want to find out how much the math club earned from selling the various baked goods and subtract the cost of producing those goods, $\\$15$, from the number we get.\n\nFirst let's calculate how much the math club earned from selling cookies. The cookies were sold at a price of three for $\\$1$, so the math club earned $54\\div 3\\cdot\\$1=18\\cdot\\$1=\\$18$ from selling cookies.\n\nNext, let's calculate how much the club earned from selling cupcakes. At a price of $\\$2$ each, the club earned $20\\cdot \\$2=\\$40$ from selling cupcakes.\n\nFinally, let's calculate how much the club earned from selling brownies. At a price of $\\$1$ each, the club earned $35\\cdot\\$1=\\$35$ from selling brownies.\n\nNow let's add up these numbers to find out how much the club earned in total and subtract $\\$15$ from that number to find the club's profit. We obtain \\begin{align*}\n\\$18+\\$40+\\$35-\\$15&=\\$18+\\$40+\\$35-\\$15\\\\\n&=\\$18+\\$40+\\$35+(-\\$15)\\\\\n&=\\$18+\\$40+(\\$35+(-\\$15))\\\\\n&=\\$18+\\$40+(\\$20)\\\\\n&=\\boxed{78}.\n\\end{align*}Notice how we used the definition of subtraction, $a-b=a+(-b)$ to $\\$35-\\$15$ as $\\$35+(-\\$15)$ and the associative property of addition to group the numbers together."
    true_text = "\nStep 1: The skater spins 2250 degrees to her right, which means she turns 6.25 full rotations to her right (since 2250 degrees is equal to 6.25 * 360 degrees).\nStep 2: Since she starts facing north, after each full rotation, she will be facing the original direction (north) again. So, after 6.25 full rotations, she will still be facing the original direction, which is north, but rotated 6.25 * 90 = 562.5 degrees to the right.\nStep 3: Since 562.5 degrees is equivalent to 1.5625 full rotations, and she turns 1.5625 * 90 = 140.625 degrees to the right, she will be facing slightly east of north.\nStep 4: So, when she finishes her spin, she will be facing east, since east is slightly more than 90 degrees to the right of north.\n\\boxed{East}\n"

    answer = extract_answer(test_text, "MATH")
    true_answer = extract_groundtruth(true_text)
    print(f"answer = {answer}, true label = {true_answer}")