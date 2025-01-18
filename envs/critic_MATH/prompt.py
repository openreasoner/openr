COT_TASK_DESC = '<|im_start|>system\nPlease wrie a answer for this question. The Answer should format as "\n#### Reasoning Process\n...\n#### Verification\n...\n#### Final Answer\n...\n"The Final Answer should format as \\boxed{{}}.<|im_end|>'

CRITIQUE_TASK_DESC = "<|im_start|>system\nPlease analyze this weak Answer, write a strict Critic/Reflection for error re-correct and Hints/Guidelines for maximum improvement. Let’s think step by step.<|im_end|>"

REWRITE_TASK_DESC = "<|im_start|>system\nPlease write a better answer for this question refer to the comments. Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>"

COT_FORMAT_STR = (
    """<|im_start|>user\nQuestion: {question}<|im_end|>\n<|im_start|>assistant\n"""
)
CRITIQUE_FORMAT_STR = """<|im_start|>user\nQuestion: {question}\nAnswer: {answer}<|im_end|>\n<|im_start|>assistant\n"""
REWRITE_FORMAT_STR = """<|im_start|>user\nQuestion: {question}\nAnswer: {answer}\nReview: {review}<|im_end|>\n<|im_start|>assistant\n"""


SEP = None
