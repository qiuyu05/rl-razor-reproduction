"""GSM8K test accuracy under greedy decoding."""
from rlrazor.data.gsm8k import load_gsm8k
from rlrazor.gen import generate
from rlrazor.reward import is_correct

MAX_TOKENS = 512


def gsm8k_accuracy(model_path: str, limit: int | None = None, backend: str = "vllm") -> float:
    """Fraction of the GSM8K test split (all 1,319 rows unless `limit`) answered correctly."""
    ds = load_gsm8k("test", limit=limit)
    outs = generate(model_path, list(ds["prompt"]), n=1, temperature=0.0,
                    max_tokens=MAX_TOKENS, backend=backend)
    correct = [is_correct(o[0], gold) for o, gold in zip(outs, ds["answer"])]
    return sum(correct) / len(correct)
