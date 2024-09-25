COT_TASK_DESC = """Given a problem statement as contexts, the task is to answer a logical reasoning question step by step."""

# 5-shot
COT_EXAMPLES = f"""Question: Lepidopterans are insects. Every animal is multicellular. Each insect is an arthropod. Each invertebrate is an animal. Insects are six-legged. Arthropods are small. Arthropods are invertebrates. Each butterfly is a lepidopteran. Whales are not small. Polly is a lepidopteran. Is the statement \"Polly is not small\" true or false?
Steps:
Polly is a lepidopteran.
Lepidopterans are insects.
Polly is an insect.
Each insect is an arthropod.
Polly is an arthropod.
Arthropods are small.
Polly is small.
The answer is false.

Question: Every natural number is positive. Real numbers are numbers. Mersenne primes are prime. Natural numbers are integers. Prime numbers are prime. Mersenne primes are prime numbers. Prime numbers are natural numbers. Every integer is a real number. Real numbers are not imaginary. Every complex number is imaginary. 131071 is a Mersenne prime. Is the statement \"131071 is not imaginary\" true or false?
Steps:
131071 is a Mersenne prime.
Mersenne primes are prime numbers.
131071 is a prime number.
Prime numbers are natural numbers.
131071 is a natural number.
Natural numbers are integers.
131071 is an integer.
Every integer is a real number.
131071 is a real number.
Real numbers are not imaginary.
131071 is not imaginary.
The answer is true.

Question: Every whale is bony. Every insect is an arthropod. Animals are not unicellular. Each butterfly is a lepidopteran. Every lepidopteran is an insect. Each arthropod is an invertebrate. Insects are not eight-legged. Arthropods are not bony. Every invertebrate is an animal. Rex is a lepidopteran. Is the statement \"Rex is bony\" true or false?
Steps:
Rex is a lepidopteran.
Every lepidopteran is an insect.
Rex is an insect.
Every insect is an arthropod.
Rex is an arthropod.
Arthropods are not bony.
Rex is not bony.
The answer is false.

Question: Every whale is bony. Each arthropod is an invertebrate. Insects are arthropods. Each lepidopteran is an insect. Each butterfly is a lepidopteran. Invertebrates are animals. Animals are not unicellular. Insects are not eight-legged. Arthropods are not bony. Polly is a butterfly. Is the statement \"Polly is bony\" true or false?
Steps:
Polly is a butterfly.
Each butterfly is a lepidopteran.
Polly is a lepidopteran.
Each lepidopteran is an insect.
Polly is an insect.
Insects are arthropods.
Polly is an arthropod.
Arthropods are not bony.
Polly is not bony.
The answer is false.

Question: Integers are real numbers. Mersenne primes are prime. Each natural number is positive. Every imaginary number is not real. Each prime number is a natural number. Prime numbers are not composite. Every real number is real. Real numbers are numbers. Each natural number is an integer. Every Mersenne prime is a prime number. 127 is a natural number. Is the statement \"127 is not real\" true or false?
Steps:
127 is a natural number.
Each natural number is an integer.
127 is an integer.
Integers are real numbers.
127 is a real number.
Every real number is real.
127 is real.
The answer is false."""

PROBLEM_FORMAT_STR = "Question: {question}\nSteps:"
SEP = "\n"
# 5-shot
standard_task_desc = """Given a problem statement as contexts, the task is to answer a logical reasoning question."""
standard_5shot_examples = f"""Question: Lepidopterans are insects. Every animal is multicellular. Each insect is an arthropod. Each invertebrate is an animal. Insects are six-legged. Arthropods are small. Arthropods are invertebrates. Each butterfly is a lepidopteran. Whales are not small. Polly is a lepidopteran. Is the statement \"Polly is not small\" true or false?
The answer is false.
Question: Every natural number is positive. Real numbers are numbers. Mersenne primes are prime. Natural numbers are integers. Prime numbers are prime. Mersenne primes are prime numbers. Prime numbers are natural numbers. Every integer is a real number. Real numbers are not imaginary. Every complex number is imaginary. 131071 is a Mersenne prime. Is the statement \"131071 is not imaginary\" true or false?
The answer is true.
Question: Every whale is bony. Every insect is an arthropod. Animals are not unicellular. Each butterfly is a lepidopteran. Every lepidopteran is an insect. Each arthropod is an invertebrate. Insects are not eight-legged. Arthropods are not bony. Every invertebrate is an animal. Rex is a lepidopteran. Is the statement \"Rex is bony\" true or false?
The answer is false.
Question: Every whale is bony. Each arthropod is an invertebrate. Insects are arthropods. Each lepidopteran is an insect. Each butterfly is a lepidopteran. Invertebrates are animals. Animals are not unicellular. Insects are not eight-legged. Arthropods are not bony. Polly is a butterfly. Is the statement \"Polly is bony\" true or false?
The answer is false.
Question: Integers are real numbers. Mersenne primes are prime. Each natural number is positive. Every imaginary number is not real. Each prime number is a natural number. Prime numbers are not composite. Every real number is real. Real numbers are numbers. Each natural number is an integer. Every Mersenne prime is a prime number. 127 is a natural number. Is the statement \"127 is not real\" true or false?
The answer is false.
"""
