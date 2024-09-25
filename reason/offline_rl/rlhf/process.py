import json
from argparse import ArgumentParser

def load_jsonl(file_path):
    data_list = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line.strip())
            data_list.append(data)
    return data_list

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    config = parser.parse_args()

    all_data = []
    for i in range(8):
        # merge all data, you can modify the path to your save path
        all_data.extend(load_jsonl(f"{config.input_dir}/rlhf/args0/cot_sc/{i}.jsonl"))

    d = []
    for data_dict in all_data:
        question = data_dict['prompt']
        answer_list = []
        for o in data_dict['output']:
            answer_list.append({'text': o[0], 'reward':o[1]})
        data_dict_new = {'question': question, 'answer': answer_list}
        d.append(data_dict_new)

    save_dir = config.output_dir + "/rlhf_data.jsonl"
    with open(save_dir, 'w') as file:
        for data in d:
            json_str = json.dumps(data)
            file.write(json_str + '\n')