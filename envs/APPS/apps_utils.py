import json
import os


def get_test_cases(prob_dir, public_cases_type):
    if public_cases_type == 'desc':
        train_io_file = "public_input_output.json"
        test_io_file = "input_output.json"

        if os.path.exists(os.path.join(prob_dir, train_io_file)):
            with open(os.path.join(prob_dir, train_io_file)) as f:
                train_in_outs = json.load(f)
        else:
            raise Exception('using public test cases in problem description, but public test cases missing.')

        if os.path.exists(os.path.join(prob_dir, test_io_file)):
            with open(os.path.join(prob_dir, test_io_file)) as f:
                test_in_outs = json.load(f)
        else:
            test_in_outs = None

    else:
        if os.path.exists(os.path.join(prob_dir, "input_output.json")):
            with open(os.path.join(prob_dir, "input_output.json")) as f:
                in_outs = json.load(f)
                in_out_len = len(in_outs['inputs'])

                train_in_outs, test_in_outs = {}, {}
                if public_cases_type == 'half':
                    # split evenly by default
                    public_test_cases = in_out_len // 2
                    # if public_test_cases > 3:
                    #     public_test_cases = 3
                elif public_cases_type.isdigit():
                    public_test_cases = int(public_cases_type)
                else:
                    raise Exception(f"Can't understand public_test_cases {public_cases_type}")
                private_test_cases = in_out_len - public_test_cases

                if public_test_cases < 1 or private_test_cases < 1:
                    raise Exception(f"Not enough test cases: {public_test_cases}, {private_test_cases}.")

                train_in_outs['inputs'] = in_outs['inputs'][:public_test_cases]
                train_in_outs['outputs'] = in_outs['outputs'][:public_test_cases]
                test_in_outs['inputs'] = in_outs['inputs'][public_test_cases:]
                test_in_outs['outputs'] = in_outs['outputs'][public_test_cases:]
        else:
            train_in_outs, test_in_outs = None, None  # will raise an Exception later
    return train_in_outs, test_in_outs

