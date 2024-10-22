import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from core import State
import random
import json
import re
import os
os.environ["HUGGINGFACE_HUB_TOKEN"] = "hf_yourkey"

set_seed(1234)  # For reproducibility
random.seed(42)

def complete_answer(problem, partial_answer, checkpoint = "Qwen/Qwen2-Math-7B-Instruct"): #"Qwen/Qwen2-1.5B-Instruct"):
    # Define the prompt with the problem and partial answer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float16, device_map="cuda")
    prompt = problem + partial_answer
    #print("prompt:", prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    temperatures = [0.7, 1.0]
    temp = random.choice(temperatures)
    outputs = model.generate(**inputs, do_sample=True, max_new_tokens=200, temperature=temp)
    completion_only = outputs[0][inputs['input_ids'].shape[1]:]
    result = tokenizer.decode(completion_only, skip_special_tokens=True)

    #print("result:", result)
    return result


def check_answer(groundtruth_answer, response):
    # Use regular expressions to split the text into sentences
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', response.strip())
    
    # Extract the last sentence from the list of sentences
    last_sentence = sentences[-1] if sentences else ''
    
    # Check if groundtruth_answer is a substring of the last sentence
    return groundtruth_answer.strip() in last_sentence.strip()


# def getrollouts(s, n = 20):
def getrollouts(s, n = 5):
  corrs = []
  q = s.q
  pa = s.pa
  for i in range(n):
    re = complete_answer(q, pa)
    s.add_rollout(re)
    #check the answer
    a = s.a
    if check_answer(a, re):
      corrs.append(1)
    else:
      corrs.append(0)
  return s.rollouts, corrs

def cal_mc(s):
    corr = 0
    for r in s.rollouts:
        if check_answer(s.a, r):
            corr += 1
    return corr * 1.0 / len(s.rollouts)
 
def cal_mc_bs(s, bs = 5):
    n = len(s.rollouts)
    subn = max(1,random.randint(n//2, n))
    mc = 0
    for i in range(bs):
        corr = 0
        sub = random.sample(s.rollouts, subn)
        for r in sub:
            if check_answer(s.a, r):
                corr += 1
        mc += corr * 1.0 / len(sub)
    return mc / bs 


# def select(states):
#     ms, mr, maxqu = states[0], "", -1
#     for s in states:
#         mcs = cal_mc(s)
#         s.mc = mcs
#         for r in s.rollouts:
#             q = Q(r, mcs)
#             u = U(s,states)
#             qu = q + u
#             if qu > maxqu:
#                 ms = s
#                 mr = r
#                 maxqu = qu 
#     return ms, mr, maxqu

def select(states):
    best_st = None
    best_roll_idx = -1
    best_qu = -1
    for s in states:
        # mcs = cal_mc(s) if s.mc is None else s.mc
        mcs = cal_mc_bs(s) if s.mc is None else s.mc
        if mcs == 0 or mcs==1.0:
            continue
        for i,r in enumerate(s.rollouts):
            if s.rollout_was_visited[i]:
                continue
            q = Q(r, mcs)
            u = U(s,states)
            qu = q + u
            if qu > best_qu:
                best_st = s
                best_roll_idx = i
                best_qu = qu
    #
    if best_roll_idx != -1:
        best_st.rollout_was_visited[best_roll_idx] = True
    return best_st,best_st.rollouts[best_roll_idx],best_qu

def split_sentence_middle(sentence):
    # Remove leading/trailing whitespace
    sentence = sentence.strip()

    # Calculate the middle index
    middle_index = len(sentence) // 2

    # Find the nearest space to the middle index to split the sentence cleanly
    if sentence[middle_index] != ' ':
        left_space = sentence.rfind(' ', 0, middle_index)
        right_space = sentence.find(' ', middle_index)
        
        # Choose the nearest space to split
        if left_space == -1:
            split_index = right_space
        elif right_space == -1:
            split_index = left_space
        else:
            split_index = left_space if (middle_index - left_space) <= (right_space - middle_index) else right_space
    else:
        split_index = middle_index

    # Split the sentence into two parts
    part1 = sentence[:split_index].strip()
    part2 = sentence[split_index:].strip()
    return part1, part2

# def error_locate(s, rollout):
#     spliti = 1
#     print("error locate:", rollout)
#     p1, p2 = split_sentence_middle(rollout)
#     print("error locate split:", p1)
#     print("error locate split:", p2)
#     st = State(s.q, p1, s.a)
#     rollouts, corrs = getrollouts(st)
#     mcst = cal_mc(st)
#     print("error locate mc:", mcst)
#     while mcst > 0:
#         word_count = len(p2.split())
#         if word_count < 2:
#             break
#         spliti += 1
#         np1, np2 = split_sentence_middle(p2)
#         print("error locate split:", np1)
#         print("error locate split:", np2)
#         #op1 = p1
#         p1 = p1 + np1
#         st2 = State(s.q, p1, s.a)
#         rollouts, corrs = getrollouts(st2)
#         mcst2 = cal_mc(st2)
#         print("error locate mc:", mcst2)
#         if mcst2 == 0:
#             return st
#         else:
#             p2 = np2
#     return st

def error_locate(s, rollout):
    current_span = rollout
    prev = ""
    divide_roll_pos_st = []
    leaf_st = []
    while True:
        word_count = len(current_span.split())
        if word_count < 2:
            break
        np1, np2 = split_sentence_middle(current_span)
        print("----")
        print(" BS[l]=", np1)
        print(" BS[r]=", np2)
        st = State(s.q, prev + np1, s.a)
        rollouts, corrs = getrollouts(st)
        # mcst = cal_mc(st)
        mcst = cal_mc_bs(st)
        st.mc = mcst
        # case 1: always correct (we are not interested in this kind of state)
        if mcst == 1:
            # leaf_st.append(st)
            break
        # case 2: right span
        elif mcst > 0:
            current_span = np2
            prev = prev + np1
            divide_roll_pos_st.append(st)
        # case 3: left span
        elif mcst == 0:
            current_span = np1
            leaf_st.append(st)
        
    #
    print("----")
    return divide_roll_pos_st,leaf_st


import math

def Q(r, mc, alpha  = 0.5, beta = 0.9, L = 500):
    part1 = alpha ** (1 - mc)
    part2 = beta ** (len(r) / L)
    Q_value = part1 * part2
    return Q_value

def U(s, states, c_puct = 0.125):
    N_sum = 0
    for item in states:
        N_sum += item.v
    numerator = math.sqrt(N_sum)
    denominator = 1 + s.v
    U_value = c_puct * (numerator / denominator)
    return U_value

def qu(i, r, mc, ncs):
    q = Q(r, mc)
    u = U(i, ncs)
    return q+u


# Function to append data to the JSON file
def append_to_json_file(filename, new_data):
    # Check if the file already exists
    if os.path.exists(filename):
        # Load existing data
        with open(filename, 'r') as file:
            data = json.load(file)
    else:
        # If the file does not exist, start with an empty list
        data = []

    # Append the new data
    data.append(new_data)

    # Save the updated data back to the file
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

    print(f"Data appended to {filename}")

def process_annotation(q, a, states, filename = 'states_list.json'):
    print("++++++")
    it = 0
    leaf_states = []
    while True:
        s, rollout, maxqu = select(states)
        if s is not None and s.pa!='':
            new_data = {
                "q": q,           # Ensure q is serializable
                "states": s.pa, # Ensure states is serializable
                "mcs": s.mc        # Ensure mcs is serializable
            }
            # Call the function to append the new data
            append_to_json_file(filename, new_data)
            it += 1
            if it > 100:
                break
        # all state-rolls pairs were exhausted
        if s is None:
            break
        print()
        print("[sel]")
        print(s)
        print("  roll=",rollout," || qu=", maxqu)
        
        s.add_visit()
        div_roll_sts,leaf_sts = error_locate(s, rollout)
        if len(div_roll_sts)==0:
            continue
        
        states.extend([s for s in div_roll_sts if s!=None and s.pa != ''])
        leaf_states.extend(leaf_sts)
    #
    ## add leaf states to data
    for s in leaf_states:
        new_data = {
            "q": q,           # Ensure q is serializable
            "states": s.pa, # Ensure states is serializable
            "mcs": s.mc        # Ensure mcs is serializable
        }
        # Call the function to append the new data
        append_to_json_file(filename, new_data)
    print("++++++")
