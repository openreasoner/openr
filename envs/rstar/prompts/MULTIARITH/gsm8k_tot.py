vote_prompt = '''Given a question and several choices of next steps, analyze each choice in detail and compare them to decide which choice is the most promising to be the next step to solve the question. After analyzing each choice in detail and comparing them, conclude your final choice with \"Therefore, the best choice is\".
Example:

Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Choice 1: There are 15 trees originally.
Choice 2: There are originally 3 cars.
Choice 3: There are 20 trees originally.
Response: Choice 1 logically follows the progression of the problem: recognizing the original amount. However, Choice 2 involves irrelevant information (3 cars) and Choice 3 contains incorrect information (20 trees). Therefore, the best choice is 1.

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Choice 1: Start with 3 cars. 2 more cars arrive, making it 3 + 2 = 5.
Choice 2: There are 3 cars. If 3 more cars leave, it's 3 - 3 = 0.
Choice 3: Originally 3 cars. Double the number, 3 * 2 = 6.
Response: Choice 1 is the only one that accurately reflects the scenario: starting with the initial number of cars and adding the ones that arrive. Choices 2 and 3 involve operations that are not relevant to the situation described in the question. Therefore, the best choice is 1.

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Choice 1: Leah had 32 chocolates. Her sister had 42. Total is 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.
Choice 2: Leah had 32 chocolates. If she gives half to her sister, it's 32 / 2 = 16. Not relevant to the question.
Choice 3: They start with 32 and 42 chocolates. If they lose some, say 10, it's 32 - 10 = 22 and 42 - 10 = 32. Total 22 + 32 = 54.
Response: Choice 1 correctly calculates the total number of chocolates and then subtracts the amount they ate, providing the correct remaining total. Choices 2 and 3 involve calculations or scenarios that do not align with the question. Therefore, the best choice is 1.

Question: {instruction}
'''


cot_prompt = '''Imagine you are trying to solve a math problem with a step-by-step approach. At each step, you should propose a single next best step to solve the problem which may involve arithmetic calculation. Please start your answer with \"Let's think step by step\" in the very first sentence. When the original question is answerable, please start the subquestion with \"The answer is\".
IMPORTANT: MAKE SURE NOT TO HAVE THE DIRECT ANSWER IN YOUR POSSIBLE STEPS OUTPUT, JUST MAKE ONE STEP AT A TIME.
Solved Example:

Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Step 1: There are 15 trees originally.
Step 2: Then there were 21 trees after some more were planted.
Step 3: So there must have been 21 - 15 = 6.
Step 4: The answer is 6.

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Step 1: There are originally 3 cars.
Step 2: 2 more cars arrive.
Step 3: 3 + 2 = 5.
Step 4: The answer is 5.

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Step 1: Originally, Leah had 32 chocolates.
Step 2: Her sister had 42.
Step 3: So in total they had 32 + 42 = 74.
Step 4: After eating 35, they had 74 - 35 = 39.
Step 5: The answer is 39.

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Step 1: Jason started with 20 lollipops.
Step 2: Then he had 12 after giving some to Denny.
Step 3: So he gave Denny 20 - 12 = 8.
Step 4: The answer is 8.

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Step 1: Shawn started with 5 toys.
Step 2: If he got 2 toys each from his mom and dad, then that is 4 more toys.
Step 3: 5 + 4 = 9.
Step 4: The answer is 9.

Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
Step 1: There were originally 9 computers.
Step 2: For each of 4 days, 5 more computers were added.
Step 3: So 5 * 4 = 20 computers were added.
Step 4: 9 + 20 is 29.
Step 5: The answer is 29.

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Step 1: Michael started with 58 golf balls.
Step 2: After losing 23 on tues- day, he had 58 - 23 = 35.
Step 3: After losing 2 more, he had 35 - 2 = 33 golf balls.
Step 4: The answer is 33.

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Step 1: Olivia had 23 dollars.
Step 2: 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars.
Step 3: So she has 23 - 15 dollars left.
Step 4: 23 - 15 is 8.
Step 5: The answer is 8.

Question: {input}
'''

vote_prompt_2 = '''Given a question and several choices of next steps, decide which choice is the most promising to solve the question. Analyze each choice in detail, then conclude in the last line "The best choice is s", where s the integer id of the choice.
Instruction: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Choice 1: There are 15 trees originally.
Choice 2: There are originally 3 cars.333
Choice 3: Originally, Leah had 32 chocolates.
Response: The best choice is 1.

Instruction: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Choice 1: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74.
Choice 2: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. 3 + 2 = 5.
Response: The best choice is 2.

Instruction: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Choice 1: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.
Response: The best choice is 1.

Insurction: {instruction}
'''

cot_prompt_2 = '''Imagine you are trying to solve a math problem with a step-by-step approach. At each step, you should propose a single next best step to solve the problem which may involve arithmetic calculation. When the original question is answerable, please start the subquestion with \"The answer is\".
IMPORTANT: MAKE SURE NOT TO HAVE THE DIRECT ANSWER IN YOUR POSSIBLE STEPS OUTPUT, JUST MAKE ONE STEP AT A TIME.
Solved Example:

Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Steps:
There are 15 trees originally.
Then there were 21 trees after some more were planted.
So there must have been 21 - 15 = 6.
The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Steps:
There are originally 3 cars.
2 more cars arrive.
3 + 2 = 5.
The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Steps:
Originally, Leah had 32 chocolates.
Her sister had 42.
So in total they had 32 + 42 = 74.
After eating 35, they had 74 - 35 = 39.
The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Steps:
Jason started with 20 lollipops.
Then he had 12 after giving some to Denny.
So he gave Denny 20 - 12 = 8.
The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Steps:
Shawn started with 5 toys.
If he got 2 toys each from his mom and dad, then that is 4 more toys.
5 + 4 = 9.
The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
Steps:
There were originally 9 computers.
For each of 4 days, 5 more computers were added.
So 5 * 4 = 20 computers were added.
9 + 20 is 29.
The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Steps:
Michael started with 58 golf balls.
After losing 23 on tuesday, he had 58 - 23 = 35.
After losing 2 more, he had 35 - 2 = 33.
The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Steps:
Olivia had 23 dollars.
5 bagels for 3 dollars each will be 5 x 3 = 15 dollars.
So she has 23 - 15 dollars left.
23 - 15 is 8.
The answer is 8.

Q: {input}
'''
