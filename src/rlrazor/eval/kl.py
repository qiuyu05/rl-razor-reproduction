import torch
from datasets import Dataset

from rlrazor.data.gsm8k import load_gsm8k

N_KL_PROMPTS = 256


def kl_prompts(seed: int = 0) -> Dataset:
    """256 fixed GSM8K test prompts, sampled without replacement."""
    ds = load_gsm8k("test")
    return ds.shuffle(seed=seed).select(range(N_KL_PROMPTS))


def forward_kl(base_logps: torch.Tensor, model_logps: torch.Tensor) -> float:
    """MC estimate of KL(pi0 || pi) from pi0-sampled responses."""
    return (base_logps - model_logps).mean().item()


def response_token_counts(tokenizer, responses) -> list[int]:
    """Scored tokens per response: exactly the tokens sequence_logprobs sums over (R46: per-token KL)."""
    return [len(tokenizer(r, add_special_tokens=False)["input_ids"]) for r in responses]


@torch.no_grad()
def sequence_logprobs(model, tokenizer, prompts, responses, batch_size: int = 8) -> torch.Tensor:
    """Sum of response-token log-probs per (prompt, response); prompt/padding masked.

    The response's terminating EOS/<|im_end|> token is intentionally not scored
    (responses are decoded with skip_special_tokens), so the resulting KL slightly
    understates the full-sequence KL; this is consistent across all checkpoints.
    """
    assert len(prompts) == len(responses)
    model.eval()
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    p_ids, r_ids = [], []
    for prompt, resp in zip(prompts, responses):
        text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        p_ids.append(tokenizer(text, add_special_tokens=False)["input_ids"])
        r_ids.append(tokenizer(resp, add_special_tokens=False)["input_ids"])

    out = torch.zeros(len(prompts), dtype=torch.float32)
    for start in range(0, len(prompts), batch_size):
        idx = range(start, min(start + batch_size, len(prompts)))
        seqs = [p_ids[i] + r_ids[i] for i in idx]
        L = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
        attn = torch.zeros((len(seqs), L), dtype=torch.long)
        resp_mask = torch.zeros((len(seqs), L), dtype=torch.bool)
        for j, i in enumerate(idx):
            n, pl = len(seqs[j]), len(p_ids[i])
            input_ids[j, :n] = torch.tensor(seqs[j])
            attn[j, :n] = 1
            resp_mask[j, pl:n] = True
        logits = model(input_ids=input_ids.to(device), attention_mask=attn.to(device)).logits
        logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        tok_lp = logp.gather(-1, input_ids[:, 1:].to(device).unsqueeze(-1)).squeeze(-1)
        tok_lp = tok_lp * resp_mask[:, 1:].to(device)
        out[start : start + len(seqs)] = tok_lp.sum(-1).cpu()
    return out
