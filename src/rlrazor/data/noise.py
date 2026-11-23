import random

from datasets import Dataset

from rlrazor.reward import completion_text, extract_boxed, normalize_number


def final_number(text: str) -> float | None:
    """The number in a completion's last \\boxed{} (None if there is none or it does not parse)."""
    boxed = extract_boxed(text)
    return None if boxed is None else normalize_number(boxed)


def corrupt_labels(ds: Dataset, p: float, seed: int) -> Dataset:
    """Swap in another row's completion for round(p*len) rows (spec Q2; R54).

    The donor is re-drawn until its completion's own final answer (last \\boxed{}) parses and is numerically
    different from the row's true answer, so every corrupted label is actually wrong. The check reads the donor's
    completion rather than the donor row's `answer`, so it holds for any target format: GSM8K gold targets and
    the teacher SFT set (other problems' teacher answers) alike. `answer` keeps the true gold. Adds a `corrupted`
    bool column. Raises ValueError if a chosen row's gold answer does not parse as a number.
    """
    n = len(ds)
    golds = list(ds["answer"])
    completions = ds["completion"]
    donor_ans = [final_number(completion_text(c)) for c in completions]
    true = [normalize_number(a) for a in golds]
    if len(set(true)) < 2:
        raise ValueError("need at least 2 distinct answers to corrupt labels")
    rng = random.Random(seed)
    chosen = set(rng.sample(range(n), round(p * n)))
    new_completions, flags = [], []
    for i in range(n):
        if i in chosen:
            if true[i] is None:  # without a parseable gold answer a donor cannot be shown to be wrong
                raise ValueError(f"row {i}: gold answer {golds[i]!r} does not parse as a number")
            for _ in range(100 * n):
                j = rng.randrange(n)
                if j != i and donor_ans[j] is not None and abs(donor_ans[j] - true[i]) > 1e-6:
                    break
            else:
                raise ValueError(f"row {i}: no donor with a different final answer")
            new_completions.append(completions[j])
            flags.append(True)
        else:
            new_completions.append(completions[i])
            flags.append(False)
    return ds.remove_columns("completion").add_column("completion", new_completions).add_column(
        "corrupted", flags
    )
