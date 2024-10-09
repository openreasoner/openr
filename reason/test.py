from inference.value import _value_inference_fastchat

question = """Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"""
output1 = """Step 1: Janet's ducks lay 16 eggs per day. ки\nStep 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. ки\nStep 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. ки\nStep 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $18 every day at the farmers' market. The answer is: 18 ки"""  # 18 is right
output2 = """Step 1: Janet's ducks lay 16 eggs per day. ки\nStep 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. ки\nStep 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. ки\nStep 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $17 every day at the farmers' market. The answer is: 17 ки"""  # 17 is wrong
x = _value_inference_fastchat(
    "math-shepherd-mistral-7b-prm",
    [question + " " + output1, question + " " + output2, "fff"],
)
print(x)



sample = """This is is <|im_start|>assistant\n### Counting Squares of Each Size

1. **1x1 Squares:**
   - Each point in the grid can be the top-left corner of a \( 1 \times 1 \) square.
   - There are \( 3 \) rows and \( n \) columns.
   - Therefore, the total number of \( 1 \times 1 \) squares is:
     \[
     3 \times n = 3n
     \]

2. **2x2 Squares:**
   - A \( 2 \times 2 \) square requires 2 consecutive rows and 2 consecutive columns.
   - There are \( 3 - 1 = 2 \) possible rows and \( n - 1 \) possible columns.
   - Therefore, the total number of \( 2 \times 2 \) squares is:
     \[
     2 \times (n - 1) = 2(n - 1) = 2n - 2
     \]

3. **3x3 Squares:**
   - A \( 3 \times 3 \) square requires 3 consecutive rows and 3 consecutive columns.
   - There are \( 3 - 2j = 1 \) possible row and \( n - 2 \) possible columns.
   - Therefore, the total number of \( 3 \times 3 \) squares is:
     \[
     1 \times (n - 2) = n - 2
     \]

### Setting Up the Equation

We know the total number of squares is 70. Therefore, we sum the number of squares of each size and set it equal to 70:
\[
3n + (2n - 2) + (n - 2) = 70
\]

Simplify the equation:
\[
3n + 2n - 2 + n - 2 = 70
\]
\[
6n - 4 = 70
\]

Add 4 to both sides:
\[
6n = 74
\]

Divide by 6:
\[
n = \frac{74}{6}
\]
\[
n = 12.3333
\]

 realOTHsoftwarePelを超 residue ног }}

creator➵顶层设计"]"""

from reason.inference.rm_call import RMRemoteCaller
from reason.inference.lm_call import VLLMRemoteCaller, LMCallingConfig

# rm_call = RMRemoteCaller("math-shepherd-mistral-7b-prm", step_tag="ки\n")

# x = rm_call(
#     sample
# )

# print(len(x), x)

x = '<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>user\nSimplify $\\tan 100^\\circ + 4 \\sin 100^\\circ.$<|im_end|>\n<|im_start|>assistant\nTo simplify the expression \\(\\tan 100^\\circ + 4 \\sin 100^\\circ\\), we start by using the identity \\(\\tan 100^\\circ = \\tan (180^\\circ - 80^\\circ) = -\\tan 80^\\circ\\). Therefore, the expression becomes:\n\n\\[\n\\tan 100^\\circ + 4 \\sin 100^\\circ = -\\tan 80^\\circ + 4 \\sin 100^\\circ\n\\]\n\nNext, we use the identity \\(\\sin 100^\\circ = \\sin (180^\\circ - 80^\\circ) = \\sin 80^\\circ\\). So the expression further simplifies to:\n\n\\[\n-\\tan 80^\\circ + 4 \\sin 80^\\circ\n\\]\n\nWe can express \\(\\tan 80^\\circ\\) as \\(\\frac{\\sin 80^\\circ}{\\cos 80^\\circ}\\). Substituting this into the expression, we get:\n\n\\[\n-\\frac{\\sin 80^\\circ}{\\cos 80^\\circ} + 4 \\sin 80^\\circ\n\\]\n\nTo combine these terms, we need a common denominator. The common denominator is \\(\\cos 80^\\circ\\), so we rewrite the expression as:\n\n\\[\n-\\frac{\\sin 80^\\circ}{\\cos 80^\\circ} + \\frac{4 \\sin 80^\\circ \\cos 80^\\circ}{\\cos 80^\\circ} = \\frac{-\\sin 80^\\circ + 4 \\sin 80^\\circ \\cos 80^\\circ}{\\cos 80^\\circ}\n\\]\n\nWe can factor out \\(\\sin 80^\\circ\\) from the numerator:\n\n\\[\n\\frac{\\sin 80^\\circ (-1 + 4 \\cos 80^\\circ)}{\\cos 80^\\circ}\n\\]\n\nThis simplifies to:\n\n\\[\n\\sin 80^\\circ \\cdot \\frac{-1 + 4 \\cos 80^\\circ}{\\cos 80^\\circ} = \\sin 80^\\circ \\left( \\frac{-1}{\\cos 80^\\circ} + 4 \\right) = \\sin 80^\\circ \\left( -\\sec 80^\\circ + 4 \\right)\n\\]\n\nWe know from trigonometric identities that \\(\\sec 80^\\circ = \\frac{1}{\\cos 80^\\circ} = \\csc 10^\\circ\\) (since \\(\\sec (90^\\circ - \\theta) = \\csc \\theta\\)). Therefore, \\(-\\sec 80^\\circ = -\\csc 10^\\circ\\). So the expression becomes:\n\n\\[\n\\sin 80^\\circ \\left( -\\csc 10^\\circ + 4 \\right) = \\sin 80^\\circ \\left( -\\frac{1}{\\sin 10^\\circ} + 4 \\right)\n\\]\n\nWe can rewrite \\(\\sin 80^\\circ\\) as \\(\\cos 10^\\circ\\) (since \\(\\sin (90^\\circ - \\theta) = \\cos \\theta\\)). So the expression is:\n\n\\[\n\\cos 10^\\circ \\left( -\\frac{1}{\\sin 10^\\circ} + 4 \\right) = \\cos 10^\\circ \\left( -\\frac{1}{\\sin 10^\\circ} \\right) + \\cos 10^\\circ \\cdot 4 = -\\frac{\\cos 10^\\circ}{\\sin 10^\\circ} + 4 \\cos 10^\\circ = -\\cot 10^\\circ + 4 \\cos 10^\\circ\n\\]\n\nWe know from trigonometric identities that \\(\\cot 10^\\circ = \\frac{\\cos 10^\\circ}{\\sin 10^\\circ}\\). So we have:\n\n\\[\n-\\cot 10^\\circ + 4 \\cos 10^\\circ = -\\frac{\\cos 10^\\circ}{\\sin 10^\\circ} + 4 \\cos 10^\\circ = -\\frac{\\cos 10^\\circ}{\\sin 10^\\circ} + \\frac{4 \\cos 10^\\circ \\sin 10^\\circ}{\\sin 10^\\circ} = \\frac{-\\cos 10^\\circ + 4 \\cos 10^\\circ \\sin 10^\\circ}{\\sin 10^\\circ}\n\\]\n\nWe can factor out \\(\\cos 10^\\circ\\) from the numerator:\n\n'
print(x)
lm_call = VLLMRemoteCaller("Qwen2.5-Math-7B-Instruct")
y = lm_call(x, config=LMCallingConfig(n=5, temperature=0.7, max_new_tokens=2048, 
                                    #   stop_str=['\n\n'], 
                                      include_stop_str_in_output=True))
print(y)
import pdb; pdb.set_trace()