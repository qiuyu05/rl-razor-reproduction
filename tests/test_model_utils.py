import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rlrazor.data.gsm8k import build_prompt
from rlrazor.eval.kl import forward_kl, sequence_logprobs
from rlrazor.eval.wise import interpolate_state_dicts
from rlrazor.gen import generate

PROMPTS = [build_prompt(q) for q in ["What is 1+1?", "Tom has 3 apples and eats 1. How many remain?", "Say hi.", "7*8?"]]
RESPONSES = [
    "2",
    "Tom has 3 - 1 = 2 apples left. The answer is \\boxed{2}. And some more words to be long.",
    "",
    "56 because seven times eight is fifty-six.",
]


def _load(path):
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
    m.eval()
    return m, AutoTokenizer.from_pretrained(path)


def test_kl_zero_for_identical_models(tiny_model_dir):
    m, tok = _load(tiny_model_dir)
    a = sequence_logprobs(m, tok, PROMPTS, RESPONSES)
    b = sequence_logprobs(m, tok, PROMPTS, RESPONSES)
    assert abs(forward_kl(a, b)) < 1e-5
    assert a[2].item() == 0.0


def test_kl_positive_for_perturbed_model(tiny_model_dir):
    m, tok = _load(tiny_model_dir)
    # responses must be samples from the base model for the MC estimate to be a KL
    samples = generate(tiny_model_dir, PROMPTS, n=8, temperature=1.0, max_tokens=16, seed=0, backend="hf")
    prompts = [p for p, rs in zip(PROMPTS, samples) for _ in rs]
    responses = [r for rs in samples for r in rs]
    base = sequence_logprobs(m, tok, prompts, responses)
    torch.manual_seed(1)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.02)
    pert = sequence_logprobs(m, tok, prompts, responses)
    assert forward_kl(base, pert) > 0


def test_kl_padding_invariant(tiny_model_dir):
    m, tok = _load(tiny_model_dir)
    b4 = sequence_logprobs(m, tok, PROMPTS, RESPONSES, batch_size=4)
    b1 = sequence_logprobs(m, tok, PROMPTS, RESPONSES, batch_size=1)
    assert b4.shape == (4,)
    assert torch.allclose(b4, b1, atol=1e-4)


def test_wise_endpoints():
    sd0 = {"w": torch.randn(3, 3), "i": torch.tensor([1, 2])}
    sd1 = {"w": torch.randn(3, 3), "i": torch.tensor([5, 6])}
    assert torch.equal(interpolate_state_dicts(sd0, sd1, 0.0)["w"], sd0["w"])
    assert torch.allclose(interpolate_state_dicts(sd0, sd1, 1.0)["w"], sd1["w"])
    assert torch.allclose(interpolate_state_dicts(sd0, sd1, 0.5)["w"], (sd0["w"] + sd1["w"]) / 2)
    assert torch.equal(interpolate_state_dicts(sd0, sd1, 0.5)["i"], sd1["i"])


def test_wise_key_mismatch():
    import pytest

    with pytest.raises(ValueError):
        interpolate_state_dicts({"a": torch.zeros(1)}, {"b": torch.zeros(1)}, 0.5)


def test_save_interpolated(tiny_model_dir, tmp_path):
    from rlrazor.eval.wise import save_interpolated

    out = save_interpolated(tiny_model_dir, tiny_model_dir, 0.5, str(tmp_path / "o"))
    assert AutoModelForCausalLM.from_pretrained(out) is not None


def test_generate_hf_shapes(tiny_model_dir):
    out = generate(tiny_model_dir, PROMPTS[:2], n=3, temperature=1.0, max_tokens=8, seed=0, backend="hf")
    assert len(out) == 2 and all(len(r) == 3 for r in out)
    assert all(isinstance(s, str) for r in out for s in r)


def test_sequence_logprobs_matches_reference_loop(tiny_model_dir):
    m, tok = _load(tiny_model_dir)
    prompts = [PROMPTS[1], PROMPTS[0]]
    responses = ["Tom has 3 - 1 = 2 apples left. The answer is \\boxed{2}.", "\nStep 1: 1+1=2"]
    got = sequence_logprobs(m, tok, prompts, responses, batch_size=2)
    for k, (p, r) in enumerate(zip(prompts, responses)):
        text = tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        pi = tok(text, add_special_tokens=False)["input_ids"]
        ri = tok(r, add_special_tokens=False)["input_ids"]
        ids = torch.tensor([pi + ri])
        with torch.no_grad():
            logits = m(input_ids=ids).logits[0].float()
        ref = sum(torch.log_softmax(logits[pos - 1], -1)[ids[0, pos]].item() for pos in range(len(pi), len(pi) + len(ri)))
        assert abs(got[k].item() - ref) < 1e-4


def test_generate_hf_untruncated_and_greedy_guard(tiny_model_dir, monkeypatch):
    import pytest
    from transformers import GenerationMixin

    seen = {}
    orig = GenerationMixin.generate

    def spy(self, *a, **kw):
        seen.update(kw)
        return orig(self, *a, **kw)

    monkeypatch.setattr(GenerationMixin, "generate", spy)
    generate(tiny_model_dir, PROMPTS[:1], n=2, temperature=1.0, max_tokens=4, backend="hf")
    assert seen["top_k"] == 0 and seen["top_p"] == 1.0 and seen["repetition_penalty"] == 1.0
    with pytest.raises(ValueError, match="greedy"):
        generate(tiny_model_dir, PROMPTS[:1], n=2, temperature=0.0, max_tokens=4, backend="hf")
