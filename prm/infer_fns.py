import torch

@torch.inference_mode()
def _qwen_math_infer_fn(input_str: str, model, tokenizer, device):
    GOOD_TOKEN = '+'
    BAD_TOKEN = '-'
    STEP_TAG = '\n\n\n\n\n'

    candidate_tokens = tokenizer.encode(f" {GOOD_TOKEN} {BAD_TOKEN}") # [488, 481]
    step_tag_id = torch.tensor([tokenizer.encode(f" {STEP_TAG}")], device=device) # 76325
    input_id = torch.tensor(
        [tokenizer.encode(input_str)], device=device)
    logits = model(input_id).logits[:,:,candidate_tokens]

    scores = logits.softmax(dim=-1)[:,:,0]
    mask = input_id == step_tag_id
    step_scores = scores[mask]
    return step_scores


@torch.inference_mode()
def _math_shepherd_infer_fn(input_str: str, model, tokenizer, device):
    GOOD_TOKEN = '+'
    BAD_TOKEN = '-'
    STEP_TAG = 'ки'
    candidate_tokens = tokenizer.encode(f"{GOOD_TOKEN} {BAD_TOKEN}")[1:] # [648, 387]
    step_tag_id = tokenizer.encode(f"{STEP_TAG}")[-1] # 12902

    input_id = torch.tensor(
        [tokenizer.encode(input_str)], device=device)
    logits = model(input_id).logits[:,:,candidate_tokens]
    scores = logits.softmax(dim=-1)[:,:,0] 
    step_scores = scores[input_id == step_tag_id]
    return step_scores