from rlrazor.reward import (
    completion_text,
    extract_boxed,
    gsm8k_reward,
    is_correct,
    make_flipped_reward,
    normalize_number,
)


def test_extract_and_normalize_edge_cases():
    assert extract_boxed("a \\boxed{1,200} b") == "1,200"
    assert normalize_number("1,200") == 1200.0
    assert normalize_number("$72.00") == 72.0
    assert normalize_number("72 clips") == 72.0
    assert extract_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert normalize_number(r"\frac{1}{2}") is None
    assert extract_boxed(r"\boxed{3} then \boxed{4}") == "4"
    assert extract_boxed(r"\boxed{a{b}c}") == "a{b}c"
    assert extract_boxed("no box") is None
    assert extract_boxed(r"\boxed{unclosed") is None
    assert normalize_number("-5") == -5.0
    assert normalize_number("abc") is None


def test_is_correct():
    assert is_correct("so \\boxed{72}", "72")
    assert is_correct("so \\boxed{72.0}", "72")
    assert not is_correct("\\boxed{71}", "72")
    assert not is_correct("truncated...", "72")
    assert not is_correct(r"\boxed{\frac{1}{2}}", "0.5")


def test_reward_accepts_conversational_completions():
    out = gsm8k_reward(
        [[{"role": "assistant", "content": "\\boxed{5}"}], "\\boxed{4}"],
        answer=["5", "5"],
    )
    assert out == [1.0, 0.0]


def test_flipped_reward_rate_and_determinism():
    def base(completions, **kw):
        return [1.0] * len(completions)

    comps = ["x"] * 10_000
    out = make_flipped_reward(base, 0.2, seed=0)(comps)
    assert 0.18 <= out.count(0.0) / len(out) <= 0.22
    assert out == make_flipped_reward(base, 0.2, seed=0)(comps)
    assert make_flipped_reward(base, 0.0, seed=0)(comps) == base(comps)


def test_latex_wrappers_parse():
    for t, v in [
        (r"\$72", 72.0),
        (r"72\%", 72.0),
        (r"72\text{ clips}", 72.0),
        (r"72 \text{clips}", 72.0),
        (r"72\mbox{ clips}", 72.0),
        ("72.", 72.0),
        (r"1\,200", 1200.0),
        (r"1\!200", 1200.0),
    ]:
        assert normalize_number(t) == v, t
    assert extract_boxed(r"\boxed {5}") == "5"
    assert normalize_number(r"\dfrac{1}{2}") is None
    assert normalize_number(r"\frac{1}{2}") is None
    assert is_correct("\\boxed{\\$1,200}", "1200")
    assert is_correct("\\boxed{72 clips}", "72")


def test_ambiguous_numbers_rejected():
    for t in ["12 and 5", "3 apples 5 oranges", "1,2,3", "1,20"]:
        assert normalize_number(t) is None, t


def test_completion_text_none_content():
    assert completion_text([{"role": "assistant", "content": None}]) == ""
