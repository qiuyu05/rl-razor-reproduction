from dataclasses import replace

import pytest

from rlrazor.config import (
    RunConfig,
    config_from_yaml,
    config_to_yaml,
    dump_config,
    load_config,
    run_id,
)
import json

from rlrazor.results import append_row, export_jsonl, has_row, load_rows


def _cfg(**kw):
    base = dict(arm="a", method="sft", lr=3e-5, seed=1)
    base.update(kw)
    return RunConfig(**base)


def test_run_id_stable_and_distinct():
    c = _cfg(noise_p=0.2)
    assert run_id(c) == run_id(_cfg(noise_p=0.2))
    assert run_id(c) != run_id(replace(c, seed=2))
    assert run_id(c) != run_id(replace(c, noise_p=0.3))
    assert "n0.2" in run_id(c)
    assert run_id(c).startswith("sft-lr3e-05-s1-n0.2-")
    assert len(run_id(c).rsplit("-", 1)[1]) == 6


def test_run_id_optional_suffixes():
    rid = run_id(_cfg(mix_f=0.5, lora_r=8))
    assert "-f0.5-r8-" in rid
    assert "-n" not in rid


def test_run_id_wise_alpha_suffix():
    base = _cfg()
    w = _cfg(wise_alpha=0.5)
    assert "-w0.5-" in run_id(w)
    assert "-w" not in run_id(base)
    assert run_id(w) != run_id(base)


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        _cfg(method="bogus")


def test_config_roundtrip(tmp_path):
    c = _cfg(noise_p=0.2, init_from="x/y", smoke=True, wise_alpha=0.5)
    p = tmp_path / "c.yaml"
    dump_config(c, p)
    assert load_config(p) == c
    assert config_from_yaml(config_to_yaml(c)) == c


def test_append_is_idempotent(tmp_path):
    d = tmp_path / "sub" / "results"
    assert load_rows(d) == []
    assert not has_row(d, "r", 0)
    assert append_row(d, {"run_id": "r", "step": 0, "acc": 0.5}) is True
    assert append_row(d, {"run_id": "r", "step": 0, "acc": 0.9}) is False
    assert append_row(d, {"run_id": "r", "step": 1, "acc": 0.6}) is True
    assert has_row(d, "r", 0)
    assert sorted(f.name for f in d.iterdir()) == ["r__step0.json", "r__step1.json"]
    rows = load_rows(d)
    assert len(rows) == 2 and rows[0]["acc"] == 0.5


def test_load_rows_sorted_and_ignores_tmp(tmp_path):
    d = tmp_path / "res"
    append_row(d, {"run_id": "b", "step": 0})
    append_row(d, {"run_id": "a", "step": 2})
    append_row(d, {"run_id": "a", "step": 1})
    (d / "a__step9.json.tmp").write_text("{garbage")
    (d / ".x.tmp123").write_text("{garbage")
    assert [(r["run_id"], r["step"]) for r in load_rows(d)] == [("a", 1), ("a", 2), ("b", 0)]


def test_export_jsonl_roundtrip(tmp_path):
    d = tmp_path / "res"
    append_row(d, {"run_id": "b", "step": 0, "v": 1})
    append_row(d, {"run_id": "a", "step": 0, "v": 2})
    out = tmp_path / "o" / "results.jsonl"
    assert export_jsonl(d, out) == 2
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    assert lines == load_rows(d) and lines[0]["run_id"] == "a"


def test_config_from_handwritten_yaml_coerces_types():
    c = config_from_yaml("arm: q0\nmethod: sft\nlr: 3e-5\nseed: 1\n")
    assert isinstance(c.lr, float) and c.lr == 3e-05
    assert run_id(c) == run_id(RunConfig(arm="q0", method="sft", lr=3e-5, seed=1))
    c2 = config_from_yaml("arm: q0\nmethod: sft\nlr: 1\nnoise_p: 0\n")
    assert isinstance(c2.lr, float) and isinstance(c2.noise_p, float)


def test_config_unknown_key_raises():
    with pytest.raises(ValueError, match="bogus"):
        config_from_yaml("arm: q0\nmethod: sft\nlr: 1e-5\nbogus: 3\n")


def test_append_row_rejects_slash_run_id(tmp_path):
    with pytest.raises(ValueError):
        append_row(tmp_path, {"run_id": "a/b", "step": 0})
