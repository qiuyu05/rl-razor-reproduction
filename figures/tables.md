# Results tables

Formats: accuracy and benchmark scores in percent (1 decimal); forgetting and benchmark changes in percentage points, checkpoint − base (1 decimal, negative = forgetting); forward KL in nats (2 decimals); R² 2 decimals. Matched-accuracy targets are base + 3, + 6, ... accuracy points up to the largest accuracy every compared method reaches. Seed bands use the highest target every seed reaches; a band marked partial is missing seeds and has no std.
Note: ignoring rows from unknown arm(s): pilot.
Note: dropping 2 smoke row(s).
Note: excluded 3 diverged run(s) (9 checkpoint row(s)) from frontiers and fits; listed in the appendix.

## Reference constants

**Constants quoted in the write-up**

| Item | Value |
|---|---|
| Paper (RL's Razor), arXiv | 2509.04259 |
| Model | Qwen2.5-0.5B-Instruct |
| Paper: R² of forgetting vs forward KL (LLMs) | 0.71 |
| Paper: R² of forgetting vs forward KL (ParityMNIST toy, §4 p6) | 0.96 |
| Frontier tolerance (acc. points) | 2 |
| Matched-accuracy target step (acc. points) | 3 |
| Paper sections cited | §1 (p2), §3.1 (p4, p5), §4 (p6), §5.1 (p7, p8), §5.2 (p8), §7 (p9, p10) |
| Paper sections cited, continued | Abstract (p1), App. A (p18), App. B.1 (p18), Table 2 (p19), Fig. 9 (p22), Fig. 11 (p23) |

**Base model scores (%)**

| Metric | Score (%) |
|---|---|
| GSM8K | 48.1 |
| hellaswag | 51.2 |
| mmlu | 48.2 |
| truthfulqa | 41.5 |
| winogrande | 56.6 |
| ifeval | 26.4 |
| humaneval | 32.3 |

## Q0: SFT vs GRPO

**Headline: final checkpoints at each method's best setting (mean ± sd over seeds; forgetting in pp with 2 decimals; Welch t-test of mean forgetting vs GRPO, uncorrected)**

| Best setting (final ckpt) | Seeds | GSM8K (%) | GSM8K change (pp) | Forgetting (pp) | KL (nats/seq) | KL (nats/token) | Forgetting − GRPO (pp) | Welch t | p |
|---|---|---|---|---|---|---|---|---|---|
| grpo lr=1e-06 | 3 | 55.1 ± 1.4 | 7.1 | 0.25 ± 0.16 | 50.52 | 0.163 | n/a | n/a | n/a |
| rft_iter lr=3e-06 | 3 | 51.4 ± 2.2 | 3.3 | 0.32 ± 0.53 | 9.54 | 0.031 | 0.08 | 0.24 | 0.828 |
| rft_offline lr=3e-06 | 3 | 49.4 ± 0.5 | 1.3 | 0.55 ± 0.33 | 5.54 | 0.018 | 0.30 | 1.41 | 0.257 |
| rl_distill_sft lr=3e-06 | 1 | 51.6 | 3.5 | 0.13 | 50.57 | 0.163 | -0.12 | n/a | n/a |
| self_rewrite lr=3e-06 | 3 | 47.0 ± 0.8 | -1.1 | -0.48 ± 0.13 | 37.94 | 0.122 | -0.73 | -5.96 | 0.004 |
| teacher_sft lr=3e-06 | 3 | 51.2 ± 0.4 | 3.1 | -0.05 ± 0.19 | 10.43 | 0.034 | -0.30 | -2.06 | 0.110 |

The seeded comparison above is the primary result. The pooled-frontier tables below mix settings and seeds: a Pareto frontier is an upper envelope of noisy points and a method with more checkpoints gets a more optimistic envelope.

**Matched accuracy: forgetting along each pooled frontier (fit, and piecewise-linear interpolation of the observed frontier points)**

| Target acc. (%) | teacher_sft fit (pp) | teacher_sft interp. (pp) | grpo fit (pp) | grpo interp. (pp) | teacher_sft − grpo fit (pp) | teacher_sft − grpo interp. (pp) |
|---|---|---|---|---|---|---|
| 51.1 | -0.1 | 0.0 | 0.5 | 0.6 | -0.6 | -0.5 |

**Frontier fits (near-frontier points; exponential with either sign of b, or a line)**

| Method | Fit | Points fitted | Fit RMSE (pp) |
|---|---|---|---|
| teacher_sft | exponential | 5 | 0.1 |
| grpo | exponential | 5 | 0.0 |

Caveat: 51.1% lies below grpo's lowest frontier point (52.4% GSM8K, 0.6 pp, from grpo-lr3e-06-s0-30f0f4 step 234); its interp. value there is that one point held downward, not an interpolation, and its fit is anchored by it.

**Matched accuracy: seed variance at the highest target every seed reaches**

| Method | Best setting | Seeds used | Target acc. (%) | Forgetting mean (pp) | Std (pp) | Status |
|---|---|---|---|---|---|---|
| teacher_sft | teacher_sft lr=3e-06 | 2/3 | 51.1 | -0.1 | n/a | partial (2/3 seeds) |
| grpo | grpo lr=1e-06 | 3/3 | 51.1 | 0.0 | 0.1 | complete |

A seed band's target is the highest target every seed of the best setting reaches; it can be lower than the highest target above and can differ between methods. Near-frontier filtering is by accuracy distance, so dominated checkpoints of other seeds can enter a fit. The base model is a point of every method's set.

**Q0: per-benchmark change vs base (pp), mean over seeds**

| Setting (final ckpt) | Seeds | GSM8K (%) | hellaswag | mmlu | truthfulqa | winogrande | ifeval | humaneval | Mean forgetting (pp) | KL |
|---|---|---|---|---|---|---|---|---|---|---|
| grpo lr=1e-06 | 3 | 55.1 | -0.5 | -0.1 | -0.1 | 0.7 | 1.4 | 0.0 | 0.2 | 50.52 |
| grpo lr=3e-06 | 1 | 52.4 | -0.2 | 0.5 | -0.1 | 0.4 | 0.9 | 1.8 | 0.6 | 161.34 |
| teacher_sft lr=1e-05 | 1 | 47.5 | -0.6 | 2.1 | -0.4 | 2.4 | -4.4 | -1.8 | -0.5 | 23.78 |
| teacher_sft lr=3e-05 | 1 | 38.4 | -2.6 | -2.8 | -2.2 | -0.8 | -10.4 | -10.4 | -4.9 | 52.71 |
| teacher_sft lr=3e-06 | 3 | 51.2 | -0.3 | 0.5 | 0.0 | 1.6 | -3.5 | 1.4 | 0.0 | 10.43 |
| teacher_sft lr=7e-05 | 1 | 27.3 | -10.6 | -11.9 | -1.6 | -2.0 | -15.0 | -32.3 | -12.2 | 83.89 |

## Forgetting vs forward KL (all checkpoints)

**Quadratic fit of mean forgetting on forward KL, all non-diverged checkpoints (the headline Q0-only fit is in the next section)**

| Checkpoints | R² (this study) | R² (paper, LLMs) |
|---|---|---|
| 118 | 0.32 | 0.71 |

## Forward KL: subsets, per-token and per-sequence detail

**Quadratic fit of mean forgetting on forward KL, by subset (headline: Q0 only, the population of the paper's 0.71)**

| Subset | Checkpoints | R² | Spearman ρ | R² (paper, LLMs) |
|---|---|---|---|---|
| Q0: teacher SFT + GRPO + base, sequence KL (headline) | 25 | 0.30 | 0.24 | 0.71 |
| Q0, per-token KL | 25 | 0.30 | 0.24 |  |
| Q0, median per-sequence KL | 25 | 0.29 | 0.23 |  |
| Q0 + Q1, all methods + base, sequence KL | 57 | 0.36 | -0.26 |  |
| SFT-like (teacher SFT, RFT, self-rewrite, RL-distill), all LRs | 44 | 0.91 | -0.65 |  |
| SFT-like, LR ≤ 1e-5 | 36 | 0.21 | -0.39 |  |
| SFT-like, LR ≤ 3e-6 | 32 | 0.17 | -0.32 |  |
| All methods, LR ≤ 3e-6 | 44 | 0.11 | 0.05 |  |
| GRPO only | 12 | 0.79 | 0.94 |  |

**Per-checkpoint forward KL on the 256 cached base samples: mean and median per sequence (nats), per token (total log-ratio / total tokens) and median per token, share of the total from the top 10% of samples, Spearman ρ with response length**

| Setting | Step | GSM8K (%) | Forgetting (pp) | KL mean | KL median | KL/token | Median KL/token | Top-10% share | ρ(len) |
|---|---|---|---|---|---|---|---|---|---|
| distill lr=1e-05 s0 | 33 | 44.9 | 0.1 | 6.48 | 5.75 | 0.021 | 0.019 | 0.27 | 0.36 |
| distill lr=1e-05 s0 | 67 | 35.0 | 0.5 | 8.63 | 6.63 | 0.028 | 0.022 | 0.32 | 0.43 |
| distill lr=1e-05 s0 | 100 | 35.6 | -0.1 | 7.57 | 6.02 | 0.024 | 0.020 | 0.32 | 0.42 |
| grpo lr=1e-06 s0 | 78 | 50.7 | -0.6 | 6.65 | 5.52 | 0.021 | 0.017 | 0.28 | 0.06 |
| grpo lr=1e-06 p=0.2 s0 | 78 | 50.8 | -0.5 | 2.42 | 1.89 | 0.008 | 0.006 | 0.35 | -0.01 |
| grpo lr=1e-06 p=0.4 s0 | 78 | 47.5 | -0.5 | 0.77 | 0.61 | 0.002 | 0.002 | 0.46 | 0.03 |
| grpo lr=1e-06 s0 | 156 | 52.9 | -0.1 | 26.97 | 25.07 | 0.087 | 0.078 | 0.21 | 0.23 |
| grpo lr=1e-06 p=0.2 s0 | 156 | 52.2 | 0.2 | 12.83 | 11.73 | 0.041 | 0.036 | 0.23 | 0.13 |
| grpo lr=1e-06 p=0.4 s0 | 156 | 47.5 | -0.3 | 5.25 | 5.10 | 0.017 | 0.017 | 0.24 | 0.11 |
| grpo lr=1e-06 s0 | 234 | 53.8 | 0.4 | 36.05 | 34.50 | 0.116 | 0.110 | 0.18 | 0.27 |
| grpo lr=1e-06 p=0.2 s0 | 234 | 50.9 | 0.0 | 24.54 | 22.79 | 0.079 | 0.075 | 0.20 | 0.19 |
| grpo lr=1e-06 p=0.4 s0 | 234 | 46.1 | -0.3 | 8.32 | 7.95 | 0.027 | 0.027 | 0.21 | 0.09 |
| grpo lr=1e-06 s1 | 78 | 51.5 | -0.1 | 6.28 | 5.41 | 0.020 | 0.017 | 0.27 | 0.08 |
| grpo lr=1e-06 p=0.2 s1 | 78 | 50.0 | -0.3 | 3.52 | 2.87 | 0.011 | 0.009 | 0.30 | 0.06 |
| grpo lr=1e-06 p=0.4 s1 | 78 | 49.7 | -0.2 | 0.58 | 0.50 | 0.002 | 0.002 | 0.48 | 0.16 |
| grpo lr=1e-06 s1 | 156 | 54.3 | 0.0 | 29.31 | 27.71 | 0.094 | 0.092 | 0.20 | 0.17 |
| grpo lr=1e-06 p=0.2 s1 | 156 | 50.3 | -0.2 | 13.23 | 11.96 | 0.043 | 0.038 | 0.23 | 0.16 |
| grpo lr=1e-06 p=0.4 s1 | 156 | 50.1 | -0.3 | 3.66 | 2.95 | 0.012 | 0.009 | 0.30 | 0.12 |
| grpo lr=1e-06 s1 | 234 | 56.5 | 0.1 | 53.15 | 48.94 | 0.171 | 0.179 | 0.18 | 0.30 |
| grpo lr=1e-06 p=0.2 s1 | 234 | 53.1 | -0.5 | 21.20 | 19.86 | 0.068 | 0.063 | 0.20 | 0.26 |
| grpo lr=1e-06 p=0.4 s1 | 234 | 49.7 | 0.0 | 10.53 | 9.39 | 0.034 | 0.031 | 0.25 | 0.15 |
| grpo lr=1e-06 s2 | 78 | 52.3 | -0.4 | 6.04 | 5.01 | 0.019 | 0.016 | 0.27 | 0.06 |
| grpo lr=1e-06 p=0.2 s2 | 78 | 50.2 | -0.7 | 2.70 | 1.97 | 0.009 | 0.006 | 0.32 | 0.01 |
| grpo lr=1e-06 p=0.4 s2 | 78 | 50.0 | -0.7 | 0.78 | 0.58 | 0.003 | 0.002 | 0.46 | -0.01 |
| grpo lr=1e-06 s2 | 156 | 53.7 | 0.1 | 29.89 | 27.03 | 0.096 | 0.083 | 0.21 | 0.18 |
| grpo lr=1e-06 p=0.2 s2 | 156 | 50.6 | -0.2 | 8.45 | 7.52 | 0.027 | 0.026 | 0.23 | 0.10 |
| grpo lr=1e-06 p=0.4 s2 | 156 | 49.7 | 0.2 | 7.21 | 6.63 | 0.023 | 0.021 | 0.25 | 0.06 |
| grpo lr=1e-06 s2 | 234 | 55.2 | 0.3 | 62.36 | 57.75 | 0.201 | 0.189 | 0.19 | 0.28 |
| grpo lr=1e-06 p=0.2 s2 | 234 | 52.6 | 0.2 | 16.98 | 15.93 | 0.055 | 0.050 | 0.20 | 0.15 |
| grpo lr=1e-06 p=0.4 s2 | 234 | 48.1 | 0.5 | 16.94 | 16.20 | 0.055 | 0.053 | 0.20 | 0.13 |
| grpo lr=3e-06 s0 | 78 | 49.1 | 0.5 | 112.42 | 109.26 | 0.362 | 0.362 | 0.17 | 0.24 |
| grpo lr=3e-06 s0 | 156 | 50.4 | 0.5 | 143.63 | 140.59 | 0.463 | 0.488 | 0.17 | 0.25 |
| grpo lr=3e-06 s0 | 234 | 52.4 | 0.6 | 161.34 | 152.55 | 0.520 | 0.544 | 0.17 | 0.37 |
| rft_iter lr=3e-06 s0 | 1 | 48.7 | -0.1 | 2.22 | 1.83 | 0.007 | 0.007 | 0.31 | 0.23 |
| rft_iter lr=3e-06 s0 | 2 | 48.7 | -0.4 | 4.75 | 4.51 | 0.015 | 0.015 | 0.26 | 0.34 |
| rft_iter lr=3e-06 s0 | 3 | 50.1 | 0.1 | 7.53 | 6.79 | 0.024 | 0.023 | 0.24 | 0.38 |
| rft_iter lr=3e-06 s0 | 4 | 49.7 | -0.2 | 10.14 | 9.33 | 0.033 | 0.031 | 0.23 | 0.43 |
| rft_iter lr=3e-06 s1 | 1 | 49.4 | 0.1 | 2.09 | 1.95 | 0.007 | 0.006 | 0.31 | 0.22 |
| rft_iter lr=3e-06 s1 | 2 | 51.2 | 0.3 | 4.50 | 3.97 | 0.015 | 0.013 | 0.27 | 0.29 |
| rft_iter lr=3e-06 s1 | 3 | 50.0 | 0.8 | 6.93 | 6.16 | 0.022 | 0.021 | 0.25 | 0.32 |
| rft_iter lr=3e-06 s1 | 4 | 50.6 | 0.8 | 9.29 | 8.65 | 0.030 | 0.029 | 0.23 | 0.36 |
| rft_iter lr=3e-06 s2 | 1 | 49.1 | -0.2 | 2.18 | 2.16 | 0.007 | 0.007 | 0.30 | 0.24 |
| rft_iter lr=3e-06 s2 | 2 | 49.1 | 0.4 | 4.73 | 4.38 | 0.015 | 0.015 | 0.27 | 0.34 |
| rft_iter lr=3e-06 s2 | 3 | 50.9 | 0.3 | 7.10 | 6.59 | 0.023 | 0.023 | 0.24 | 0.41 |
| rft_iter lr=3e-06 s2 | 4 | 53.8 | 0.4 | 9.19 | 8.28 | 0.030 | 0.029 | 0.23 | 0.42 |
| rft_offline lr=3e-06 s0 | 192 | 48.8 | -0.3 | 2.55 | 2.26 | 0.008 | 0.008 | 0.30 | 0.18 |
| rft_offline lr=3e-06 s0 | 384 | 49.1 | 0.6 | 5.44 | 4.90 | 0.018 | 0.016 | 0.27 | 0.41 |
| rft_offline lr=3e-06 α=0.25 from rft_offline-lr3e-06-s0-863d2a s0 | 384 | 48.1 | -0.1 | 0.11 | 0.07 | 0.000 | 0.000 | 0.92 | 0.08 |
| rft_offline lr=3e-06 α=0.5 from rft_offline-lr3e-06-s0-863d2a s0 | 384 | 46.8 | -0.1 | 1.39 | 1.17 | 0.004 | 0.004 | 0.36 | 0.24 |
| rft_offline lr=3e-06 α=0.75 from rft_offline-lr3e-06-s0-863d2a s0 | 384 | 49.7 | 0.4 | 4.06 | 3.61 | 0.013 | 0.012 | 0.28 | 0.36 |
| rft_offline lr=3e-06 s1 | 191 | 49.4 | 0.0 | 2.25 | 1.94 | 0.007 | 0.007 | 0.34 | 0.25 |
| rft_offline lr=3e-06 s1 | 381 | 49.1 | 0.2 | 5.51 | 4.66 | 0.018 | 0.016 | 0.27 | 0.41 |
| rft_offline lr=3e-06 s2 | 192 | 50.0 | 0.2 | 2.81 | 2.43 | 0.009 | 0.008 | 0.29 | 0.28 |
| rft_offline lr=3e-06 s2 | 383 | 50.0 | 0.8 | 5.65 | 4.82 | 0.018 | 0.017 | 0.27 | 0.42 |
| rft_offline lr=1e-05 s0 | 192 | 42.5 | 0.4 | 11.52 | 10.54 | 0.037 | 0.036 | 0.24 | 0.50 |
| rft_offline lr=1e-05 s0 | 384 | 45.0 | -0.3 | 18.52 | 15.01 | 0.060 | 0.056 | 0.25 | 0.58 |
| rft_offline lr=3e-05 s0 | 192 | 34.6 | -3.1 | 32.55 | 27.42 | 0.105 | 0.096 | 0.24 | 0.64 |
| rft_offline lr=3e-05 s0 | 384 | 37.3 | -3.8 | 44.35 | 35.92 | 0.143 | 0.123 | 0.26 | 0.62 |
| rft_offline lr=7e-05 s0 | 192 | 19.5 | -9.4 | 64.39 | 50.92 | 0.207 | 0.183 | 0.25 | 0.66 |
| rft_offline lr=7e-05 s0 | 384 | 25.0 | -12.3 | 76.60 | 58.09 | 0.247 | 0.212 | 0.28 | 0.66 |
| rl_distill_sft lr=3e-06 s0 | 212 | 53.6 | 0.6 | 34.89 | 33.98 | 0.112 | 0.110 | 0.18 | 0.33 |
| rl_distill_sft lr=3e-06 s0 | 423 | 51.6 | 0.1 | 50.57 | 48.81 | 0.163 | 0.157 | 0.17 | 0.40 |
| self_rewrite lr=3e-06 s0 | 234 | 46.7 | -0.6 | 27.05 | 25.76 | 0.087 | 0.086 | 0.21 | 0.47 |
| self_rewrite lr=3e-06 s0 | 468 | 46.1 | -0.5 | 36.17 | 32.58 | 0.117 | 0.114 | 0.21 | 0.53 |
| self_rewrite lr=3e-06 s1 | 234 | 48.4 | -0.8 | 24.28 | 23.36 | 0.078 | 0.077 | 0.21 | 0.49 |
| self_rewrite lr=3e-06 s1 | 468 | 47.5 | -0.3 | 38.43 | 35.90 | 0.124 | 0.119 | 0.21 | 0.54 |
| self_rewrite lr=3e-06 s2 | 234 | 45.6 | -0.7 | 27.13 | 25.14 | 0.087 | 0.084 | 0.21 | 0.45 |
| self_rewrite lr=3e-06 s2 | 468 | 47.3 | -0.6 | 39.23 | 36.58 | 0.126 | 0.122 | 0.21 | 0.52 |
| teacher_sft lr=3e-06 s0 | 230 | 51.4 | -0.2 | 7.39 | 6.89 | 0.024 | 0.023 | 0.24 | 0.22 |
| teacher_sft lr=3e-06 p=0.2 s0 | 230 | 50.4 | -0.7 | 9.57 | 8.41 | 0.031 | 0.029 | 0.24 | 0.24 |
| teacher_sft lr=3e-06 p=0.4 s0 | 230 | 46.6 | -1.1 | 11.49 | 9.97 | 0.037 | 0.036 | 0.23 | 0.30 |
| teacher_sft lr=3e-06 f=0.02 s0 | 234 | 50.1 | -1.0 | 7.22 | 6.79 | 0.023 | 0.023 | 0.23 | 0.21 |
| teacher_sft lr=3e-06 f=0.05 s0 | 242 | 49.1 | -0.8 | 7.20 | 6.67 | 0.023 | 0.023 | 0.24 | 0.21 |
| teacher_sft lr=3e-06 f=0.1 s0 | 255 | 51.9 | -1.8 | 7.48 | 7.21 | 0.024 | 0.025 | 0.22 | 0.22 |
| teacher_sft lr=3e-06 f=0.25 s0 | 306 | 48.8 | -1.2 | 7.11 | 6.73 | 0.023 | 0.022 | 0.23 | 0.23 |
| teacher_sft lr=3e-06 s0 | 459 | 51.5 | -0.3 | 10.21 | 9.31 | 0.033 | 0.031 | 0.24 | 0.35 |
| teacher_sft lr=3e-06 f=0.5 s0 | 459 | 51.6 | -0.9 | 7.55 | 6.72 | 0.024 | 0.023 | 0.23 | 0.27 |
| teacher_sft lr=3e-06 p=0.2 s0 | 459 | 50.1 | -0.2 | 13.31 | 11.96 | 0.043 | 0.040 | 0.25 | 0.34 |
| teacher_sft lr=3e-06 p=0.4 s0 | 459 | 45.2 | -0.5 | 14.95 | 13.58 | 0.048 | 0.045 | 0.25 | 0.39 |
| teacher_sft lr=3e-06 α=0.25 from teacher_sft-lr3e-06-s0-72863e s0 | 459 | 48.3 | -0.5 | 0.45 | 0.34 | 0.001 | 0.001 | 0.57 | 0.07 |
| teacher_sft lr=3e-06 α=0.5 from teacher_sft-lr3e-06-s0-72863e s0 | 459 | 49.2 | 0.3 | 3.36 | 2.94 | 0.011 | 0.010 | 0.30 | 0.18 |
| teacher_sft lr=3e-06 α=0.75 from teacher_sft-lr3e-06-s0-72863e s0 | 459 | 50.8 | -0.1 | 7.88 | 7.08 | 0.025 | 0.024 | 0.25 | 0.30 |
| teacher_sft lr=3e-06 f=0.02 s0 | 468 | 51.0 | 0.2 | 10.13 | 9.23 | 0.033 | 0.032 | 0.24 | 0.38 |
| teacher_sft lr=3e-06 f=0.05 s0 | 483 | 53.1 | 0.0 | 10.21 | 9.22 | 0.033 | 0.032 | 0.23 | 0.38 |
| teacher_sft lr=3e-06 f=0.1 s0 | 510 | 50.8 | -1.1 | 9.87 | 8.80 | 0.032 | 0.031 | 0.24 | 0.35 |
| teacher_sft lr=3e-06 f=0.25 s0 | 611 | 49.4 | -0.8 | 9.93 | 8.91 | 0.032 | 0.030 | 0.24 | 0.35 |
| teacher_sft lr=3e-06 f=0.5 s0 | 917 | 50.7 | -0.7 | 10.67 | 9.23 | 0.034 | 0.032 | 0.25 | 0.41 |
| teacher_sft lr=3e-06 s1 | 230 | 49.9 | -0.1 | 7.27 | 6.60 | 0.023 | 0.022 | 0.24 | 0.20 |
| teacher_sft lr=3e-06 p=0.2 s1 | 230 | 50.1 | 0.0 | 9.16 | 8.27 | 0.030 | 0.026 | 0.24 | 0.24 |
| teacher_sft lr=3e-06 p=0.4 s1 | 230 | 45.1 | -0.5 | 11.41 | 10.04 | 0.037 | 0.034 | 0.24 | 0.27 |
| teacher_sft lr=3e-06 f=0.02 s1 | 234 | 50.0 | 0.0 | 7.44 | 6.96 | 0.024 | 0.024 | 0.23 | 0.29 |
| teacher_sft lr=3e-06 f=0.05 s1 | 242 | 50.6 | -0.8 | 7.07 | 6.52 | 0.023 | 0.022 | 0.24 | 0.22 |
| teacher_sft lr=3e-06 s1 | 459 | 51.3 | 0.0 | 10.52 | 9.55 | 0.034 | 0.031 | 0.24 | 0.33 |
| teacher_sft lr=3e-06 p=0.2 s1 | 459 | 49.1 | 0.2 | 12.78 | 11.10 | 0.041 | 0.038 | 0.24 | 0.36 |
| teacher_sft lr=3e-06 p=0.4 s1 | 459 | 46.6 | 0.1 | 15.84 | 13.49 | 0.051 | 0.046 | 0.25 | 0.41 |
| teacher_sft lr=3e-06 f=0.02 s1 | 468 | 51.9 | -0.4 | 10.50 | 9.20 | 0.034 | 0.032 | 0.24 | 0.37 |
| teacher_sft lr=3e-06 f=0.05 s1 | 483 | 52.1 | -0.1 | 10.52 | 9.24 | 0.034 | 0.031 | 0.24 | 0.38 |
| teacher_sft lr=3e-06 s2 | 230 | 48.7 | -0.3 | 7.47 | 6.62 | 0.024 | 0.022 | 0.25 | 0.21 |
| teacher_sft lr=3e-06 p=0.2 s2 | 230 | 49.4 | -0.3 | 9.12 | 8.56 | 0.029 | 0.028 | 0.23 | 0.29 |
| teacher_sft lr=3e-06 p=0.4 s2 | 230 | 46.3 | -0.6 | 11.41 | 10.16 | 0.037 | 0.034 | 0.23 | 0.28 |
| teacher_sft lr=3e-06 f=0.02 s2 | 234 | 49.1 | -0.7 | 7.76 | 7.12 | 0.025 | 0.024 | 0.24 | 0.26 |
| teacher_sft lr=3e-06 f=0.05 s2 | 242 | 50.4 | -1.2 | 7.12 | 6.79 | 0.023 | 0.023 | 0.25 | 0.22 |
| teacher_sft lr=3e-06 s2 | 459 | 50.7 | 0.1 | 10.55 | 9.21 | 0.034 | 0.033 | 0.24 | 0.36 |
| teacher_sft lr=3e-06 p=0.2 s2 | 459 | 49.3 | -0.1 | 12.66 | 11.07 | 0.041 | 0.036 | 0.24 | 0.37 |
| teacher_sft lr=3e-06 p=0.4 s2 | 459 | 47.7 | -0.3 | 14.68 | 12.58 | 0.047 | 0.044 | 0.24 | 0.36 |
| teacher_sft lr=3e-06 f=0.02 s2 | 468 | 49.7 | -0.7 | 10.23 | 8.91 | 0.033 | 0.032 | 0.24 | 0.31 |
| teacher_sft lr=3e-06 f=0.05 s2 | 483 | 50.9 | -0.2 | 9.82 | 9.21 | 0.032 | 0.029 | 0.24 | 0.36 |
| teacher_sft lr=1e-05 s0 | 230 | 45.8 | -0.9 | 16.87 | 14.85 | 0.054 | 0.050 | 0.24 | 0.46 |
| teacher_sft lr=1e-05 s0 | 459 | 47.5 | -0.5 | 23.78 | 20.74 | 0.077 | 0.069 | 0.24 | 0.51 |
| teacher_sft lr=3e-05 s0 | 230 | 24.9 | -3.7 | 41.23 | 35.61 | 0.133 | 0.127 | 0.23 | 0.60 |
| teacher_sft lr=3e-05 s0 | 459 | 38.4 | -4.9 | 52.71 | 42.56 | 0.170 | 0.150 | 0.27 | 0.60 |
| teacher_sft lr=7e-05 s0 | 230 | 19.5 | -10.7 | 73.16 | 60.38 | 0.236 | 0.213 | 0.25 | 0.63 |
| teacher_sft lr=7e-05 s0 | 459 | 27.3 | -12.2 | 83.89 | 60.88 | 0.270 | 0.226 | 0.28 | 0.64 |
| teacher_sft lr=0.0001 r=64 s0 | 230 | 48.5 | -0.2 | 9.72 | 8.81 | 0.031 | 0.031 | 0.22 | 0.36 |
| teacher_sft lr=0.0001 r=64 s0 | 459 | 49.9 | 0.3 | 14.65 | 12.94 | 0.047 | 0.044 | 0.23 | 0.47 |
| teacher_sft lr=0.0003 r=64 s0 | 230 | 44.7 | -1.2 | 19.81 | 17.33 | 0.064 | 0.063 | 0.22 | 0.47 |
| teacher_sft lr=0.0003 r=64 s0 | 459 | 41.7 | -0.8 | 32.16 | 27.87 | 0.104 | 0.096 | 0.25 | 0.55 |

**Forward KL at near-equal accuracy: checkpoints of each method's best setting within 1 point of 51.1% GSM8K**

| Method | Checkpoints | Mean GSM8K (%) | Mean forgetting (pp) | Mean KL (nats/seq) | Mean KL (nats/token) | Checkpoints used |
|---|---|---|---|---|---|---|
| teacher_sft | 4 | 51.2 | -0.09 | 9.67 | 0.031 | lr3e-06-s0@230, lr3e-06-s0@459, lr3e-06-s1@459, lr3e-06-s2@459 |
| grpo | 2 | 51.1 | -0.36 | 6.46 | 0.021 | lr1e-06-s0@78, lr1e-06-s1@78 |
| rft_iter | 4 | 50.7 | 0.37 | 7.11 | 0.023 | lr3e-06-s0@3, lr3e-06-s1@2, lr3e-06-s1@4, lr3e-06-s2@3 |
| rl_distill_sft | 1 | 51.6 | 0.13 | 50.57 | 0.163 | lr3e-06-s0@423 |

**Top-5 base samples by log-ratio at the seed-0 final checkpoints of the best settings (grpo lr=1e-06 vs teacher_sft lr=3e-06); the other method's value and rank (of 256) on the same sample**

| Ranked by | Sample | Tokens | Log-ratio (nats) | Per token | Other method (nats) | Other's rank |
|---|---|---|---|---|---|---|
| grpo | 230 | 250 | 82.70 | 0.331 | 6.96 | 170 |
| grpo | 239 | 485 | 80.23 | 0.165 | 24.99 | 9 |
| grpo | 192 | 214 | 73.76 | 0.345 | 9.54 | 126 |
| grpo | 189 | 372 | 73.39 | 0.197 | 3.54 | 221 |
| grpo | 61 | 395 | 72.48 | 0.183 | 16.80 | 37 |
| teacher_sft | 204 | 397 | 37.56 | 0.095 | 66.60 | 9 |
| teacher_sft | 180 | 371 | 28.86 | 0.078 | 21.42 | 211 |
| teacher_sft | 182 | 229 | 28.84 | 0.126 | 40.73 | 89 |
| teacher_sft | 12 | 512 | 26.40 | 0.052 | 53.81 | 35 |
| teacher_sft | 254 | 512 | 26.24 | 0.051 | 30.44 | 157 |

**Do checkpoints drift on the same samples? Spearman ρ of per-sample log-ratios (final checkpoints, best settings)**

| Checkpoint A | Checkpoint B | Spearman ρ |
|---|---|---|
| grpo s0 | teacher_sft s0 | 0.22 |
| grpo s0 | grpo s1 | 0.73 |
| grpo s0 | grpo s2 | 0.75 |
| teacher_sft s0 | teacher_sft s1 | 0.90 |
| teacher_sft s0 | teacher_sft s2 | 0.91 |
| grpo s0 | rl_distill_sft s0 | 0.88 |

**GRPO training dynamics: mean over the first and last 10 logged steps (policy entropy per token on its own rollouts, reward, completion length)**

| Run | Steps logged | Entropy first | Entropy last | Reward first | Reward last | Length first | Length last |
|---|---|---|---|---|---|---|---|
| grpo-lr1e-06-s0-6f9f01 | 234 | 0.322 | 0.127 | 0.413 | 0.653 | 310.9 | 235.1 |
| grpo-lr1e-06-s0-n0.2-110b38 | 234 | 0.326 | 0.134 | 0.440 | 0.557 | 313.0 | 281.0 |
| grpo-lr1e-06-s0-n0.4-6b80f2 | 234 | 0.328 | 0.186 | 0.476 | 0.518 | 310.4 | 282.1 |
| grpo-lr1e-06-s1-bc03b4 | 234 | 0.292 | 0.112 | 0.440 | 0.632 | 305.0 | 254.7 |
| grpo-lr1e-06-s1-n0.2-a19556 | 234 | 0.288 | 0.149 | 0.467 | 0.541 | 304.4 | 259.5 |
| grpo-lr1e-06-s1-n0.4-247ba6 | 234 | 0.297 | 0.196 | 0.497 | 0.514 | 305.6 | 293.9 |
| grpo-lr1e-06-s2-0a4e37 | 234 | 0.315 | 0.092 | 0.426 | 0.672 | 307.3 | 274.2 |
| grpo-lr1e-06-s2-n0.2-17ab18 | 234 | 0.307 | 0.154 | 0.445 | 0.581 | 306.9 | 280.4 |
| grpo-lr1e-06-s2-n0.4-548ac4 | 234 | 0.313 | 0.171 | 0.481 | 0.522 | 307.6 | 246.3 |
| grpo-lr3e-06-s0-30f0f4 | 234 | 0.323 | 0.111 | 0.425 | 0.584 | 309.8 | 223.7 |

**Greedy diagnostic on the first 200 GSM8K test prompts (results/q0_diag_samples.json)**

| Checkpoint | Accuracy (%) | Boxed (%) | Mean tokens | Hit 512 tokens (%) |
|---|---|---|---|---|
| base | 45.5 | 94.5 | 301.5 | 4.0 |
| teacher_sft-lr1e-05-s0-9bf416/step-459 | 49.5 | 98.0 | 294.1 | 2.0 |
| teacher_sft-lr3e-05-s0-2f6ed6/step-459 | 39.0 | 96.0 | 295.4 | 3.5 |
| rft_offline-lr1e-05-s0-64354f/step-384 | 44.0 | 95.5 | 301.5 | 4.5 |
| grpo-lr1e-05-s0-307b32/step-78 | 21.5 | 82.0 | 303.6 | 23.5 |
| grpo-lr1e-05-s0-307b32/step-234 | 0.0 | 0.0 | 512.0 | 98.0 |

## Q1: SFT on self-generated data

**Matched accuracy: forgetting along each pooled frontier**

| Method | Target acc. (%) | Fit (pp) | Interp. (pp) |
|---|---|---|---|
| teacher_sft | 51.1 | -0.1 | 0.0 |
| grpo | 51.1 | 0.5 | 0.6 |
| rft_offline | 51.1 | n/a | n/a |
| rft_iter | 51.1 | 0.7 | 0.7 |
| self_rewrite | 51.1 | n/a | n/a |
| rl_distill_sft | 51.1 | 0.6 | 0.6 |

**Frontier fits (near-frontier points; exponential with either sign of b, or a line)**

| Method | Fit | Points fitted | Fit RMSE (pp) |
|---|---|---|---|
| teacher_sft | exponential | 5 | 0.1 |
| grpo | exponential | 5 | 0.0 |
| rft_offline | exponential | 7 | 0.3 |
| rft_iter | interpolation | 3 | 0.0 |
| self_rewrite | exponential | 6 | 0.3 |
| rl_distill_sft | interpolation | 1 | 0.0 |

**Matched accuracy: seed variance at the highest target every seed reaches**

| Method | Best setting | Seeds used | Target acc. (%) | Forgetting mean (pp) | Std (pp) | Status |
|---|---|---|---|---|---|---|
| teacher_sft | teacher_sft lr=3e-06 | 2/3 | 51.1 | -0.1 | n/a | partial (2/3 seeds) |
| grpo | grpo lr=1e-06 | 3/3 | 51.1 | 0.0 | 0.1 | complete |
| rft_offline | rft_offline lr=3e-06 | 0/3 | n/a | n/a | n/a | no seed reaches a target |
| rft_iter | rft_iter lr=3e-06 | 2/3 | 51.1 | 0.5 | n/a | partial (2/3 seeds) |
| self_rewrite | self_rewrite lr=3e-06 | 0/3 | n/a | n/a | n/a | no seed reaches a target |
| rl_distill_sft | rl_distill_sft lr=3e-06 | 1/1 | 51.1 | 0.6 | n/a | complete |

**Q1: per-benchmark change vs base (pp), mean over seeds**

| Setting (final ckpt) | Seeds | GSM8K (%) | hellaswag | mmlu | truthfulqa | winogrande | ifeval | humaneval | Mean forgetting (pp) | KL |
|---|---|---|---|---|---|---|---|---|---|---|
| rft_iter lr=3e-06 | 3 | 51.4 | -0.9 | -0.1 | 0.0 | 1.7 | -2.6 | 3.9 | 0.3 | 9.54 |
| rft_offline lr=1e-05 | 1 | 45.0 | -0.2 | 2.5 | -1.2 | 1.2 | -4.8 | 0.6 | -0.3 | 18.52 |
| rft_offline lr=3e-05 | 1 | 37.3 | -1.8 | -0.2 | -1.9 | -2.4 | -7.4 | -9.1 | -3.8 | 44.35 |
| rft_offline lr=3e-06 | 3 | 49.4 | -0.4 | 1.1 | -0.1 | 2.2 | -0.9 | 1.4 | 0.5 | 5.54 |
| rft_offline lr=7e-05 | 1 | 25.0 | -7.8 | -11.9 | -2.7 | -3.2 | -18.1 | -29.9 | -12.3 | 76.60 |
| rl_distill_sft lr=3e-06 | 1 | 51.6 | 0.0 | 0.7 | 0.3 | 1.0 | -1.8 | 0.6 | 0.1 | 50.57 |
| self_rewrite lr=3e-06 | 3 | 47.0 | 0.5 | 0.7 | -0.9 | 1.2 | -3.3 | -1.0 | -0.5 | 37.94 |

## Per-benchmark change at the best settings (seeded)

**Per-benchmark change at each method's best setting (final checkpoint, change vs base in pp, mean ± sd over seeds)**

| Best setting (final ckpt) | Seeds | GSM8K (pp) | hellaswag | mmlu | truthfulqa | winogrande | ifeval | humaneval | Mean (pp) |
|---|---|---|---|---|---|---|---|---|---|
| grpo lr=1e-06 | 3 | 7.1 ± 1.4 | -0.5 ± 0.3 | -0.1 ± 0.4 | -0.1 ± 0.1 | 0.7 ± 0.5 | 1.4 ± 0.5 | 0.0 ± 1.2 | 0.2 ± 0.2 |
| rft_iter lr=3e-06 | 3 | 3.3 ± 2.2 | -0.9 ± 0.1 | -0.1 ± 0.5 | 0.0 ± 0.3 | 1.7 ± 0.6 | -2.6 ± 0.3 | 3.9 ± 2.5 | 0.3 ± 0.5 |
| rft_offline lr=3e-06 | 3 | 1.3 ± 0.5 | -0.4 ± 0.7 | 1.1 ± 1.3 | -0.1 ± 0.3 | 2.2 ± 0.3 | -0.9 ± 0.9 | 1.4 ± 1.3 | 0.5 ± 0.3 |
| rl_distill_sft lr=3e-06 | 1 | 3.5 | 0.0 | 0.7 | 0.3 | 1.0 | -1.8 | 0.6 | 0.1 |
| self_rewrite lr=3e-06 | 3 | -1.1 ± 0.8 | 0.5 ± 0.2 | 0.7 ± 0.3 | -0.9 ± 0.2 | 1.2 ± 0.5 | -3.3 ± 1.3 | -1.0 ± 1.4 | -0.5 ± 0.1 |
| teacher_sft lr=3e-06 | 3 | 3.1 ± 0.4 | -0.3 ± 0.2 | 0.5 ± 0.4 | 0.0 ± 0.1 | 1.6 ± 0.3 | -3.5 ± 1.2 | 1.4 ± 0.7 | 0.0 ± 0.2 |

**Welch t-test p vs GRPO at the best settings (post hoc, uncorrected; n = seeds)**

| Method | GSM8K | hellaswag | mmlu | truthfulqa | winogrande | ifeval | humaneval | Mean |
|---|---|---|---|---|---|---|---|---|
| rft_iter | 0.075 | 0.104 | 1.000 | 0.695 | 0.088 | 0.000 | 0.095 | 0.828 |
| rft_offline | 0.011 | 0.893 | 0.264 | 0.813 | 0.016 | 0.035 | 0.234 | 0.257 |
| self_rewrite | 0.002 | 0.016 | 0.065 | 0.005 | 0.275 | 0.017 | 0.399 | 0.004 |
| teacher_sft | 0.029 | 0.420 | 0.171 | 0.534 | 0.065 | 0.010 | 0.173 | 0.110 |
| rft_iter (Bonferroni, × 6 benchmarks) | n/a | 0.625 | 1.000 | 1.000 | 0.527 | 0.003 | 0.572 | n/a |
| rft_offline (Bonferroni, × 6 benchmarks) | n/a | 1.000 | 1.000 | 1.000 | 0.095 | 0.211 | 1.000 | n/a |
| self_rewrite (Bonferroni, × 6 benchmarks) | n/a | 0.094 | 0.389 | 0.031 | 1.000 | 0.101 | 1.000 | n/a |
| teacher_sft (Bonferroni, × 6 benchmarks) | n/a | 1.000 | 1.000 | 1.000 | 0.389 | 0.057 | 1.000 | n/a |

The six-benchmark mean can hide benchmark-specific changes; p-values are post hoc and uncorrected for the six benchmarks, with n = seeds per method.

## Q2: noise robustness

SFT corrupts a fraction p of labels; GRPO flips each reward with probability p. These are different kinds of corruption: the comparison is about practical robustness, not equal information content. p = 0 is the method's Q0 best setting at the same LR.

**Final checkpoint vs noise rate, mean and std over seeds (forgetting with 2 decimals)**

| Method | p | Seeds | GSM8K (%) | Std | Forgetting (pp) | Std | IFEval change (pp) | Std | KL |
|---|---|---|---|---|---|---|---|---|---|
| teacher_sft | 0 | 3 | 51.2 | 0.4 | -0.05 | 0.19 | -3.5 | 1.2 | 10.43 |
| teacher_sft | 0.2 | 3 | 49.5 | 0.6 | -0.04 | 0.20 | -2.3 | 1.2 | 12.92 |
| teacher_sft | 0.4 | 3 | 46.5 | 1.3 | -0.23 | 0.29 | -2.5 | 0.4 | 15.16 |
| grpo | 0 | 3 | 55.1 | 1.4 | 0.25 | 0.16 | 1.4 | 0.5 | 50.52 |
| grpo | 0.2 | 3 | 52.2 | 1.2 | -0.11 | 0.35 | 0.2 | 0.9 | 20.90 |
| grpo | 0.4 | 3 | 48.0 | 1.8 | 0.06 | 0.41 | 0.2 | 0.8 | 11.93 |

**GSM8K at each noise rate: gain over base and GRPO − teacher_sft (final checkpoints; Welch t-test over seeds, uncorrected)**

| p | GRPO gain (pp) | teacher_sft gain (pp) | GRPO change vs p = 0 (pp) | teacher_sft change vs p = 0 (pp) | GRPO − teacher_sft (pp) | Welch t | p-value |
|---|---|---|---|---|---|---|---|
| 0 | 7.1 | 3.1 | 0.0 | 0.0 | 4.0 | 4.83 | 0.029 |
| 0.2 | 4.1 | 1.4 | -2.9 | -1.7 | 2.7 | 3.60 | 0.040 |
| 0.4 | -0.1 | -1.6 | -7.2 | -4.7 | 1.5 | 1.19 | 0.309 |

**Q2: per-benchmark change vs base (pp), mean over seeds**

| Setting (final ckpt) | Seeds | GSM8K (%) | hellaswag | mmlu | truthfulqa | winogrande | ifeval | humaneval | Mean forgetting (pp) | KL |
|---|---|---|---|---|---|---|---|---|---|---|
| grpo lr=1e-06 p=0.2 | 3 | 52.2 | -0.4 | -0.2 | 0.0 | 0.8 | 0.2 | -1.0 | -0.1 | 20.90 |
| grpo lr=1e-06 p=0.4 | 3 | 48.0 | -0.5 | -0.2 | -0.1 | 0.9 | 0.2 | 0.0 | 0.1 | 11.93 |
| teacher_sft lr=3e-06 p=0.2 | 3 | 49.5 | 0.1 | 1.2 | -1.6 | 1.1 | -2.3 | 1.2 | 0.0 | 12.92 |
| teacher_sft lr=3e-06 p=0.4 | 3 | 46.5 | -0.3 | 0.6 | -2.4 | 2.3 | -2.5 | 1.0 | -0.2 | 15.16 |

## Q3: general-data mixing into SFT

**Forgetting at 51.1% GSM8K (the GRPO band's target), and final checkpoints (mean ± sd over seeds; f = 0 is the Q0 best setting)**

| f | Seeds used | Forgetting at target (pp) | Std | Status | Final GSM8K (%) | Std | Final forgetting (pp) | Std | Final IFEval change (pp) | Std |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2/3 | -0.1 | n/a | partial (2/3 seeds) | 51.2 | 0.4 | -0.05 | 0.19 | -3.5 | 1.2 |
| 0.02 | 1/3 | -0.2 | n/a | partial (1/3 seeds) | 50.9 | 1.1 | -0.29 | 0.45 | -5.5 | 1.2 |
| 0.05 | 2/3 | -0.3 | n/a | partial (2/3 seeds) | 52.0 | 1.1 | -0.07 | 0.10 | -7.0 | 0.7 |
| 0.1 | 1/1 | -1.2 | n/a | complete | 50.8 | n/a | -1.06 | n/a | -8.1 | n/a |
| 0.25 | 0/1 | n/a | n/a | partial (0/1 seeds) | 49.4 | n/a | -0.83 | n/a | -7.4 | n/a |
| 0.5 | 1/1 | -0.8 | n/a | complete | 50.7 | n/a | -0.67 | n/a | -7.8 | n/a |

**Primary criterion in detail: forgetting at 51.1% GSM8K per seed (pp, 3 decimals; each run's own checkpoints and the base)**

| f | Seed | Best GSM8K reached (%) | Forgetting at target (pp) |
|---|---|---|---|
| 0 | s0 | 51.5 | -0.196 |
| 0 | s1 | 51.3 | -0.022 |
| 0 | s2 | 50.7 | not reached |
| 0.02 | s0 | 51.0 | not reached |
| 0.02 | s1 | 51.9 | -0.187 |
| 0.02 | s2 | 49.7 | not reached |
| 0.05 | s0 | 53.1 | 0.030 |
| 0.05 | s1 | 52.1 | -0.610 |
| 0.05 | s2 | 50.9 | not reached |
| 0.1 | s0 | 51.9 | -1.236 |
| 0.25 | s0 | 49.4 | not reached |
| 0.5 | s0 | 51.6 | -0.771 |
| GRPO band | 3/3 seeds | n/a | 0.005 ± 0.100 |

**Primary criterion margin: mean forgetting at the target minus the threshold (GRPO mean − sd = 0.005 − 0.100 pp; GRPO band sd 0.10 pp); negative = misses**

| f | Seeds reaching | Mean (pp) | Threshold (pp) | Margin (pp) |
|---|---|---|---|---|
| 0 | 2/3 | -0.109 | -0.095 | -0.014 |
| 0.02 | 1/3 | -0.187 | -0.095 | -0.092 |
| 0.05 | 2/3 | -0.290 | -0.095 | -0.195 |
| 0.1 | 1/1 | -1.236 | -0.095 | -1.141 |
| 0.5 | 1/1 | -0.771 | -0.095 | -0.676 |

Crossover: no fully-seeded mix fraction reaches GRPO mean − std (-0.1 pp).

**Secondary criterion (pre-registered, R54): IFEval change at 51.1% GSM8K**

| f | Seeds used | IFEval change (pp) | Std | Status |
|---|---|---|---|---|
| 0 | 2/3 | -3.2 | n/a | partial (2/3 seeds) |
| 0.02 | 1/3 | -4.3 | n/a | partial (1/3 seeds) |
| 0.05 | 2/3 | -5.5 | n/a | partial (2/3 seeds) |
| 0.1 | 1/1 | -8.7 | n/a | complete |
| 0.25 | 0/1 | n/a | n/a | partial (0/1 seeds) |
| 0.5 | 1/1 | -7.9 | n/a | complete |
| GRPO (best LR, seeds) | 3/3 | 0.7 | 0.9 | complete |

IFEval crossover: no fully-seeded mix fraction reaches GRPO mean − std (-0.3 pp).

Excluded from the crossover (not every seed reaches the target): f = 0.02, 0.05, 0.25, 0.

Final GSM8K is below GRPO's range (mean − std = 53.8%) at f = 0, 0.02, 0.05, 0.1, 0.25, 0.5 (f = 0, the unmixed Q0 setting, included).

**Final IFEval change: each seeded mix vs f = 0 (Welch t-test over seeds, uncorrected)**

| f | Seeds | Difference vs f = 0 (pp) | Welch t | p-value |
|---|---|---|---|---|
| 0.02 | 3 | -2.0 | -2.16 | 0.097 |
| 0.05 | 3 | -3.5 | -4.35 | 0.017 |

**Q3: per-benchmark change vs base (pp), mean over seeds**

| Setting (final ckpt) | Seeds | GSM8K (%) | hellaswag | mmlu | truthfulqa | winogrande | ifeval | humaneval | Mean forgetting (pp) | KL |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher_sft lr=3e-06 f=0.02 | 3 | 50.9 | -0.3 | 0.8 | 0.2 | 1.6 | -5.5 | 1.4 | -0.3 | 10.29 |
| teacher_sft lr=3e-06 f=0.05 | 3 | 52.0 | -0.3 | 0.7 | 0.2 | 2.3 | -7.0 | 3.7 | -0.1 | 10.19 |
| teacher_sft lr=3e-06 f=0.1 | 1 | 50.8 | -1.0 | 0.4 | 0.6 | 1.8 | -8.1 | 0.0 | -1.1 | 9.87 |
| teacher_sft lr=3e-06 f=0.25 | 1 | 49.4 | -1.4 | 0.2 | 0.4 | 1.4 | -7.4 | 1.8 | -0.8 | 9.93 |
| teacher_sft lr=3e-06 f=0.5 | 1 | 50.7 | -0.6 | 0.7 | 1.0 | 0.8 | -7.8 | 1.8 | -0.7 | 10.67 |

## Q4: cheap mitigations

**WiSE-FT, 1 seed: θ = (1 − α)·θ₀ + α·θ_ft (α = 1 is the source run's final checkpoint)**

| Source run | α | GSM8K (%) | Forgetting (pp) | KL |
|---|---|---|---|---|
| rft_offline-lr3e-06-s0-863d2a | 1 (source) | 49.1 | 0.6 | 5.44 |
| rft_offline-lr3e-06-s0-863d2a | 0.25 | 48.1 | -0.1 | 0.11 |
| rft_offline-lr3e-06-s0-863d2a | 0.5 | 46.8 | -0.1 | 1.39 |
| rft_offline-lr3e-06-s0-863d2a | 0.75 | 49.7 | 0.4 | 4.06 |
| teacher_sft-lr3e-06-s0-72863e | 1 (source) | 51.5 | -0.3 | 10.21 |
| teacher_sft-lr3e-06-s0-72863e | 0.25 | 48.3 | -0.5 | 0.45 |
| teacher_sft-lr3e-06-s0-72863e | 0.5 | 49.2 | 0.3 | 3.36 |
| teacher_sft-lr3e-06-s0-72863e | 0.75 | 50.8 | -0.1 | 7.88 |

**LoRA SFT, 1 seed: final checkpoint and forgetting at 51.1% GSM8K**

| Setting | GSM8K (%) | Forgetting (pp) | KL | Forgetting at target (pp) |
|---|---|---|---|---|
| teacher_sft lr=0.0001 r=64 | 49.9 | 0.3 | 14.65 | n/a |
| teacher_sft lr=0.0003 r=64 | 41.7 | -0.8 | 32.16 | n/a |

**On-policy distillation from the base model (recovery), 1 seed**

| Start run | Start GSM8K (%) | Start forgetting (pp) | End GSM8K (%) | End forgetting (pp) | Forgetting recovered (pp) | GSM8K change (pp) |
|---|---|---|---|---|---|---|
| teacher_sft-lr3e-06-s0-72863e | 51.5 | -0.3 | 35.6 | -0.1 | 0.2 | -15.8 |

**Distillation checkpoints (forgetting with 2 decimals)**

| Run | Step | GSM8K (%) | Forgetting (pp) | KL |
|---|---|---|---|---|
| start (source final) | 0 | 51.5 | -0.26 | 10.21 |
| distill lr=1e-05 | 33 | 44.9 | 0.06 | 6.48 |
| distill lr=1e-05 | 67 | 35.0 | 0.48 | 8.63 |
| distill lr=1e-05 | 100 | 35.6 | -0.06 | 7.57 |

**Q4: per-benchmark change vs base (pp), mean over seeds**

| Setting (final ckpt) | Seeds | GSM8K (%) | hellaswag | mmlu | truthfulqa | winogrande | ifeval | humaneval | Mean forgetting (pp) | KL |
|---|---|---|---|---|---|---|---|---|---|---|
| distill lr=1e-05 | 1 | 35.6 | -0.2 | 2.1 | 0.1 | 3.2 | -1.3 | -4.3 | -0.1 | 7.57 |
| rft_offline lr=3e-06 α=0.25 from rft_offline-lr3e-06-s0-863d2a | 1 | 48.1 | 0.0 | 0.2 | -0.1 | 1.6 | -1.7 | -0.6 | -0.1 | 0.11 |
| rft_offline lr=3e-06 α=0.5 from rft_offline-lr3e-06-s0-863d2a | 1 | 46.8 | 0.4 | 0.0 | -0.3 | 1.2 | -0.7 | -1.2 | -0.1 | 1.39 |
| rft_offline lr=3e-06 α=0.75 from rft_offline-lr3e-06-s0-863d2a | 1 | 49.7 | -0.2 | 1.9 | -0.5 | 1.4 | -0.4 | 0.0 | 0.4 | 4.06 |
| teacher_sft lr=0.0001 r=64 | 1 | 49.9 | -0.2 | 0.5 | 1.0 | 1.8 | -1.8 | 0.6 | 0.3 | 14.65 |
| teacher_sft lr=0.0003 r=64 | 1 | 41.7 | -2.2 | 0.4 | -0.9 | 3.6 | -3.1 | -2.4 | -0.8 | 32.16 |
| teacher_sft lr=3e-06 α=0.25 from teacher_sft-lr3e-06-s0-72863e | 1 | 48.3 | 0.0 | -0.4 | 0.0 | 1.4 | -2.8 | -1.2 | -0.5 | 0.45 |
| teacher_sft lr=3e-06 α=0.5 from teacher_sft-lr3e-06-s0-72863e | 1 | 49.2 | -0.4 | 0.7 | 0.0 | 2.0 | -1.1 | 0.6 | 0.3 | 3.36 |
| teacher_sft lr=3e-06 α=0.75 from teacher_sft-lr3e-06-s0-72863e | 1 | 50.8 | -0.4 | 1.6 | -0.1 | 2.0 | -4.1 | 0.6 | -0.1 | 7.88 |

## Data and benchmark subsets

**Teacher SFT data (Qwen2.5-7B-Instruct, T = 1, up to 16 samples per prompt, keep 1 correct)**

| Item | Value |
|---|---|
| Prompts | 7473 |
| Rows kept | 7332 |
| Keep rate (%) | 98.1 |
| Solved in the first 4 samples (%) | 97.3 |
| Samples per prompt | 4.28 |
| Correct rate over all samples (%) | 87.3 |
| Boxed-answer rate (%) | 96.4 |

**Self-generated SFT data sets**

| Method | Sampled from | Rows | Prompts kept (%) | Correct per prompt | Gold fallback (%) |
|---|---|---|---|---|---|
| rft_offline s0 | Qwen2.5-0.5B-Instruct | 6134 | 82.1 | 3.44 | n/a |
| rft_offline s1 | Qwen2.5-0.5B-Instruct | 6088 | 81.5 | 3.41 | n/a |
| rft_offline s2 | Qwen2.5-0.5B-Instruct | 6113 | 81.8 | 3.39 | n/a |
| rl_distill_sft s0 | grpo-lr1e-06-s0-6f9f01/step-234 | 6768 | 90.6 | 5.36 | n/a |
| self_rewrite s0 | own rewrites of the gold answers | 7473 | n/a | n/a | 28.2 |
| self_rewrite s1 | own rewrites of the gold answers | 7473 | n/a | n/a | 28.6 |
| self_rewrite s2 | own rewrites of the gold answers | 7473 | n/a | n/a | 27.8 |

**Iterative RFT: prompts with a kept correct sample, per round (%)**

| Run | Round 1 | Round 2 | Round 3 | Round 4 |
|---|---|---|---|---|
| rft_iter s0 | 72.4 | 77.3 | 78.5 | 79.9 |
| rft_iter s1 | 73.1 | 77.8 | 79.4 | 80.9 |
| rft_iter s2 | 72.5 | 77.4 | 79.0 | 79.9 |

**Benchmark subsets and item-sampling SE at the base score**

| Benchmark | Items | Base (%) | SE (pp) | SE of an unpaired difference (pp) |
|---|---|---|---|---|
| hellaswag | 500 | 51.2 | 2.2 | 3.2 |
| mmlu | 570 | 48.2 | 2.1 | 3.0 |
| truthfulqa | 500 | 41.5 | 2.2 | 3.1 |
| winogrande | 500 | 56.6 | 2.2 | 3.1 |
| ifeval | 541 | 26.4 | 1.9 | 2.7 |
| humaneval | 164 | 32.3 | 3.7 | 5.2 |
| GSM8K (new task, full test set) | 1319 | 48.1 | 1.4 | 1.9 |

## Cost, run time and runtime

**Actual spend from Modal billing, by phase (USD)**

| Phase | Billing hours | USD |
|---|---|---|
| Engineering: smoke tests, deterministic evaluation, pilot runs | 11-21T00–11-22T05 UTC | 6.69 |
| Teacher data, Q0 sweep at the paper's LRs, diagnostics | 11-22T06–11-22T09 UTC | 16.45 |
| Q0 at lower LRs, extra seeds, Q1 arms | 11-22T10–11-22T14 UTC | 17.60 |
| Stage 1 subtotal |  | 40.74 |
| Stage 2: Q2, Q3, Q1 remainder, Q4 | 11-22T15–11-22T23 UTC | 42.37 |
| Total (Modal billing) |  | 83.11 |

**Actual spend from Modal billing, by app (USD)**

| App | USD |
|---|---|
| rlrazor (all training and evaluation) | 82.31 |
| one-off diagnostic apps (rlrazor-det, rlrazor-diag, rlrazor-diag-q0, rlrazor-diagkl, rlrazor-diagkl2, rlrazor-equiv) | 0.80 |

**Stage 1: launcher-measured cost per grid run (GPU-seconds × list price; USD; stage 2 is in the per-arm table below)**

| Method | Runs | Total | Mean per run |
|---|---|---|---|
| grpo | 7 | 18.25 | 2.61 |
| rft_offline | 6 | 7.89 | 1.32 |
| rl_distill_sft | 1 | 1.27 | 1.27 |
| self_rewrite | 1 | 0.79 | 0.79 |
| teacher_data | 1 | 1.37 | 1.37 |
| teacher_sft | 6 | 4.07 | 0.68 |
| all launcher-measured runs | 22 | 33.64 | n/a |

**Budget guard**

| Item | USD |
|---|---|
| Guard ceiling | 41.50 |
| Launcher committed at the end (conservative; billing baseline + measured + projections) | 41.11 |
| Stage 2 guard ceiling (R54; $100 total ceiling) | 96.00 |
| Stage 2: launcher committed at the end (failed runs charged their projection) | 96.00 |
| Modal billing, month total | 83.11 |

**Stage 2: measured cost per arm and method (GPU-seconds × list price; USD)**

| Arm | Method | Runs | USD |
|---|---|---|---|
| q1 | rft_iter | 3 | 6.22 |
| q1 | self_rewrite | 2 | 1.46 |
| q2 | grpo | 6 | 14.80 |
| q2 | teacher_sft | 6 | 4.09 |
| q3 | teacher_sft | 9 | 6.60 |
| q4 | distill | 1 | 7.80 |
| q4 | rft_offline WiSE-FT | 3 | 0.51 |
| q4 | teacher_sft LoRA | 2 | 1.62 |
| q4 | teacher_sft WiSE-FT | 3 | 0.46 |
| all stage-2 runs with results |  | 35 | 43.56 |

**Training wall-clock per run (A100-40GB; hours)**

| Method | Runs | Mean (h) | Total (h) |
|---|---|---|---|
| distill | 1 | 3.27 | 3.27 |
| grpo | 13 | 0.88 | 11.50 |
| rft_iter | 3 | 0.64 | 1.91 |
| rft_offline | 6 | 0.41 | 2.44 |
| rl_distill_sft | 1 | 0.36 | 0.36 |
| self_rewrite | 3 | 0.17 | 0.51 |
| teacher_data | 1 | 1.59 | 1.59 |
| teacher_sft | 23 | 0.13 | 3.09 |

**Evaluation per checkpoint (L4; GSM8K, 6 benchmarks, KL)**

| Checkpoints | Count | Mean GPU-seconds | Total GPU-hours |
|---|---|---|---|
| All evaluated grid checkpoints (incl. diverged runs) | 126 | 639 | 22.36 |
| Non-diverged checkpoints | 117 | 632 | 20.54 |

**Not run: projected cost of the remaining stages (launcher cost model; GSM8K-sized runs at the chosen LRs; 3 GRPO / 2 SFT-like checkpoints per run)**

| Stage | Configs | Projected USD |
|---|---|---|
| Optional: async staleness, GRPO k ∈ {4, 16, 64} (needs a lagged sampler) | 3 | 7.57 |
| Total of the configs above not yet run (without staleness) |  | 0.00 |

**Runtime on Modal (requirements-modal.txt; authoritative for every GPU result)**

| Component | Version |
|---|---|
| torch | 2.14.0 |
| transformers | 5.20.2 |
| trl | 1.18.2 |
| vllm | 0.33.0 |
| lm-eval | 0.4.13 |
| datasets | 5.2.0 |
| peft | 0.22.0 |
| accelerate | 1.16.0 |
| Python (Modal image) | 3.11 |
| Training GPU | A100-40GB |
| Evaluation GPU | L4 |

## Appendix: diverged runs (excluded from frontiers and fits)

**Checkpoints of diverged runs**

| Run | Step | GSM8K (%) | Forgetting (pp) | KL |
|---|---|---|---|---|
| grpo-lr1e-05-s0-307b32 | 78 | 21.5 | 0.8 | 451.32 |
| grpo-lr1e-05-s0-307b32 | 156 | 0.0 | -2.0 | 878.64 |
| grpo-lr1e-05-s0-307b32 | 234 | 0.0 | -2.1 | 878.37 |
| grpo-lr3e-05-s0-46ef4e | 78 | 1.7 | -7.0 | 2378.10 |
| grpo-lr3e-05-s0-46ef4e | 156 | 0.0 | -7.7 | 3395.74 |
| grpo-lr3e-05-s0-46ef4e | 234 | 0.0 | -7.4 | 3396.73 |
| grpo-lr5e-05-s0-4cad6f | 78 | 0.0 | -9.4 | 2238.83 |
| grpo-lr5e-05-s0-4cad6f | 156 | 0.0 | -9.5 | 2478.37 |
| grpo-lr5e-05-s0-4cad6f | 234 | 0.0 | -9.3 | 2478.03 |

**Quadratic fit of forgetting on forward KL including diverged runs**

| Checkpoints | R² (incl. diverged) |
|---|---|
| 127 | 0.38 |
