# Open O1 dev

A simple implementation of data generation of [Improve Mathematical Reasoning in Language Models by Automated Process Supervision](https://arxiv.org/pdf/2406.06592)

## Generating data

Edit the `config.yaml` file to set the model and other parameters. Input json file should be a list of problems and answers with key `problem` and `final_answer`.

```yaml
input:
  json_file_path: 'extracted_problems_and_answers.json'

output:
  file_prefix: 'math'
  log_file_path: 'processing_log.log'

processing:
  initial_rollouts: 20
  num_rollouts: 20
  max_iterations: 100

model:
  # supported model_types: "hf", "openai", "anthropic"
  model_type: "hf"
  model_name: "Qwen/Qwen2.5-Math-7B-Instruct"
  model_args:
    max_tokens: 200
    temperature_range: [0.7, 1.0]
```

```bash
python gen_data.py
```

## Data 

Download the [**MATH dataset**](https://people.eecs.berkeley.edu/~hendrycks/MATH.tar) here.