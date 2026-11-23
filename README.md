# rl-razor: RL's Razor at small scale

Andrew Qiu. Reproduction of *RL's Razor* (Shenfeld, Pari, Agrawal, arXiv 2509.04259).

A small-scale reproduction of **RL's Razor: Why Online Reinforcement Learning Forgets Less**
(Shenfeld, Pari, Agrawal; [arXiv 2509.04259](https://arxiv.org/abs/2509.04259)).
Qwen2.5-0.5B-Instruct is fine-tuned on GSM8K with SFT (on answers from a stronger teacher, Qwen2.5-7B-Instruct)
and with GRPO. Forgetting is measured on the paper's six prior-task benchmarks (HellaSwag, TruthfulQA, MMLU, IFEval,
Winogrande, HumanEval), and forward KL to the base model is measured on GSM8K.

**Results:** [`writeup/writeup.pdf`](writeup/writeup.pdf) (source `writeup/writeup.tex`; short summary in
[`WRITEUP.md`](WRITEUP.md)). In short, at 0.5B the main result is only partly reproduced: SFT
forgets more than GRPO at matched accuracy, the paper's sign, but within noise, and neither method forgets in the
six-benchmark mean. The only consistent per-benchmark difference is IFEval. Forward KL does not
predict forgetting across methods. Of the three take-home questions, Q1 is answered on the mean, weakly (SFT on the model's
own correct answers forgets like GRPO in the mean, but the mean does not separate teacher SFT from GRPO either, and
iterative RFT still loses IFEval like teacher SFT). Q2 is half answered:
label or reward noise costs accuracy (GRPO loses more of it than SFT), not measurable forgetting. Q3 is not answered
in this setup: mixing general data into SFT made IFEval worse (confounded: the general rows carried the GSM8K system
prompt), and the mean-forgetting criterion could not rank the mixes. The reason for both is the learning rate: the three paper-range rates we ran diverged GRPO at 0.5B, and at the rates that did train SFT
barely forgets, so there is little SFT–RL gap to measure; the write-up says what would answer them (a pilot on the
paper's Qwen2.5-3B-Instruct at its learning rates, an expectation, not a result).

**Budget.** The study used about $83 of GPU time on Modal in total. Stage 1 (Q0, Q1) ran under a guard at $41.50 and cost $40.74. The total budget was then raised to $100; stage 2 (Q2, Q3, the rest of Q1, and additional runs not reported in the write-up) ran under a guard at $96.00 and cost $42.37. Total Modal billing: $83.11.

## Trained models

Final checkpoints (seed 0, best learning rate) on Hugging Face, each with a model card giving its recipe and results:

- GRPO (Dr. GRPO, β = 0), LR 1e-6: [crushsoccer/rlrazor-qwen2.5-0.5b-gsm8k-grpo](https://huggingface.co/crushsoccer/rlrazor-qwen2.5-0.5b-gsm8k-grpo)
- Teacher SFT (Qwen2.5-7B-Instruct answers), LR 3e-6: [crushsoccer/rlrazor-qwen2.5-0.5b-gsm8k-teacher-sft](https://huggingface.co/crushsoccer/rlrazor-qwen2.5-0.5b-gsm8k-teacher-sft)
- Iterative RFT (4 rounds), LR 3e-6: [crushsoccer/rlrazor-qwen2.5-0.5b-gsm8k-iterative-rft](https://huggingface.co/crushsoccer/rlrazor-qwen2.5-0.5b-gsm8k-iterative-rft)

Load one with `AutoModelForCausalLM.from_pretrained("crushsoccer/rlrazor-qwen2.5-0.5b-gsm8k-grpo")` (and `AutoTokenizer` on the same repo).

## Setup

Requirements: Python 3.11, [uv](https://docs.astral.sh/uv/), and a [Modal](https://modal.com) account for anything
that needs a GPU.

```bash
uv sync                      # local env: analysis, tests, config generation (CPU only)
uv run modal setup           # Modal auth: opens a browser, writes the token to ~/.modal.toml
                             # (or export MODAL_TOKEN_ID / MODAL_TOKEN_SECRET); never commit the token
```

GPU jobs run in a Modal image built from `requirements-modal.txt`. That image is the authoritative runtime:
torch 2.14.0, transformers 5.20.2, TRL 1.18.2, vLLM 0.33.0, lm-eval 0.4.13, Python 3.11. Training runs on
`RLRAZOR_GPU` (default A100-40GB), evaluation on `RLRAZOR_EVAL_GPU` (default L4). Results live on the Modal Volume
`rlrazor`.

## One command per stage

| Stage | Command |
|---|---|
| Tests (CPU, about 11 min) | `uv run pytest -q` |
| Smoke run (train + eval one tiny config on Modal) | `uv run python scripts/run.py configs/smoke/sft.yaml --smoke` |
| Base-model row (step 0) | `uv run modal run modal_app.py::base_eval` |
| Pilot and cost projection | `uv run python scripts/cost_report.py --fetch --spent <billing $>` |
| Generate a stage's configs | `uv run python scripts/make_configs.py --stage q0_sweep` (later stages take the best LRs / checkpoints; see the script's docstring) |
| Launch a stage under a budget guard | `mkdir -p outputs && uv run python scripts/launch.py configs/grid/q0_sweep/*.yaml --budget 41.5 --spent <billing $> \| tee outputs/launch-q0_sweep.log` |
| Train / evaluate one config | `uv run python scripts/run.py configs/grid/q0_lowlr/00_grpo_lr1e-06_s0.yaml` |
| Fetch results from the Volume | `uv run python scripts/cost_report.py --fetch && cp outputs/results.jsonl outputs/train_times.jsonl outputs/dataset_stats.jsonl results/` |
| Summarise launcher and training logs | `uv run python scripts/analyze.py --results results/results.jsonl --from-logs outputs` |
| Figures and tables | `uv run python scripts/analyze.py --results results/results.jsonl --out figures/` |
| Check the write-up's numbers | `uv run python scripts/analyze.py --check-writeup writeup/writeup.tex` |
| Build the PDF | `cd writeup && pdflatex writeup.tex && pdflatex writeup.tex` (TeX Live with booktabs, adjustbox, float, pdflscape, caption, enumitem, xspace, xltabular) |
| Actual spend | `uv run modal billing report --start 2026-11-20 --resolution h --json > results/billing_hourly.json` |

Stages in the order they were run: `q0_sweep` (teacher data, plus the paper-range LRs: GRPO diverged) →
`q0_lowlr` (lower LRs, forced because the paper's LRs diverged GRPO and did not help SFT at 0.5B) → `q0_seeds` → `q1`
(stage 1), then `q2` → `q3` → `q1` remainder (iterative RFT, self-rewrite seeds) → `q3_seeds` (f = 0.02, 0.05) →
`q4` (stage 2; additional runs not reported in the write-up).

## Reproduce the figures without a GPU

Everything in `figures/` and every number in the write-up comes from files committed under `results/`:

```bash
uv run python scripts/analyze.py --results results/results.jsonl --out figures/
uv run python scripts/analyze.py --check-writeup writeup/writeup.tex   # every decimal must be in the tables
```

`results/teacher_data_stats.json` and `results/q0_diag_samples.json` are committed outputs of one-off diagnostic
jobs whose scripts are not included in this repository.

`results/` holds the result rows (`results.jsonl`: one row per checkpoint, with per-sequence KL log-ratios), the
training times, the dataset statistics, the greedy diagnostic samples, the summaries of the launcher and training
logs, and the Modal billing record.

## Repository map

```
modal_app.py               Modal image, Volume and GPU functions (train, eval_ckpt, base_eval, teacher_data, wise)
src/rlrazor/
  config.py, results.py    RunConfig / run_id; one JSON row per (run, step)
  reward.py                \boxed{} answer parser: the GRPO reward and the GSM8K metric
  gen.py                   vLLM / HF sampling
  data/                    GSM8K, teacher answers, offline RFT, self-rewrite, label noise, general-data mixing
  train/                   SFT, GRPO (Dr. GRPO, beta 0), GKD distillation, checkpoint schedule
  eval/                    GSM8K, lm-eval forgetting suite, forward KL, WiSE-FT
  pipeline.py              train -> checkpoints -> eval on the Volume
  analysis/                Pareto frontier, exponential fit, seed bands, figures
scripts/
  run.py, launch.py        one config / many configs with a budget guard
  make_configs.py          grid configs per stage
  cost_report.py           unit costs and projections
  analyze.py               figures, tables (md + tex), numbers.json, check-writeup, KL / cost analysis
configs/                   smoke, pilot and grid YAMLs (configs/grid/BEST.md: chosen LRs)
results/                   committed study data (see above)
figures/                   generated figures and tables
writeup/                   writeup.tex and writeup.pdf
docs/paper-fidelity.md     every pipeline step against the paper, with quotes and deviations
docs/rulings.md            decision log: what each "R<n>" cited in code comments and docs decided, and why
reports/paper/             the RL's Razor PDF
tests/                     pytest suite (CPU; tiny models)
```

## Run time and cost

All GPU work ran on Modal. Billing total: **$83.11** (hourly rows through the 22:00–23:00 UTC hour of 2026-11-22).
Stage 1 cost $40.74: $6.69 for smoke tests, deterministic evaluation and the pilot, $16.45 for the teacher data and
the paper-LR sweep, and $17.60 for the low-LR runs, seeds and Q1. Stage 2 cost $42.37.
Per run: GRPO took 0.88 h on an A100-40GB on average; teacher SFT 0.13 h. In stage 1 a GRPO run cost
about $2.61 and a teacher-SFT run about $0.68. Generating the teacher data took 1.59 h on an L4. Each checkpoint
evaluation took about 639 GPU-seconds on an L4 (all 126 checkpoints) (`figures/tables.md`, "Cost, run time and runtime").
