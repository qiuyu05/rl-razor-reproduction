# Decision log (rulings R1–R56)

Code comments, `docs/paper-fidelity.md` and `configs/grid/BEST.md` cite decisions as "R<n>". This file says what
each one decided and why. It is a cleaned record of the project's decisions. Where a decision was later replaced,
the replacing ruling is named. R52, R53 and R55 were process rulings about the project's own workflow and are not
listed. "The spec", "the design" and "the plan" in code comments and these documents mean the study's original
design document, which is not part of this repository; every decision that changed it is recorded here.

| # | Decision | Why |
|---|---|---|
| R1 | `RunConfig` gains `wise_alpha` (default 0); run ids get a `-w<alpha>` suffix when it is non-zero. | WiSE-FT evaluation rows need distinct run ids. |
| R2 | The Q0 sweep has 9 configs (SFT ×3, GRPO ×3, offline RFT ×3); the full grid has 46 training configs. | It resolves an ambiguous count in the plan. |
| R3 | Answer-parser false negatives and false positives (`\$`, `\%`, `\text{}`, trailing ".", unit tails, digit lists) are fixed before anything else. | The parser is both the GRPO reward and the evaluation metric, so a bias in it would affect every number. |
| R4 | Local virtual environment location (superseded by the cloud setup). | Environment detail only. |
| R5 | The results store is one JSON file per (run, step), written atomically, with `export_jsonl` producing `results.jsonl`. | Parallel evaluations on a shared volume lose rows when they append to one file. |
| R6 | YAML values are type-coerced (e.g. `lr: 3e-5` read as a string); a "/" in a run id raises `ValueError`. | Hand-written configs would otherwise crash or collide. |
| R7 | Base-model sampling is untruncated (top-k 0, top-p 1, repetition penalty 1); EOS is not scored, and this is documented. | KL(π₀‖π) and the self-generated data sets must be samples from π₀. |
| R8 | Development tooling choice. | Process only. |
| R9 | The general-data pool is cached on the volume, and every entrypoint ends with flush + `os._exit(0)`. | Streaming datasets can deadlock at interpreter exit and leave GPU jobs hanging. |
| R10 | The decision log is kept in the repository. | The cloud workspace is ephemeral. |
| R11 | The write-up is LaTeX, compiled to PDF. A separate explainer video is out of scope for this repository. | Format decision. |
| R12 | Add a proposal for a forgetting-aware retraining workflow (an internal proposal document, not included), grounded in measured numbers. | Extra deliverable. |
| R13 | Mixed precision: fp32 master weights with bf16 autocast; checkpoints are saved in bf16. | Pure-bf16 weights swallow small RL updates (the paper's §6 bf16 artefact), which would bias SFT vs GRPO. |
| R14 | GRPO and distillation use a constant learning rate, with no decay. | With decay, intermediate checkpoints (frontier points) would depend on the run's total length. |
| R15 | Keep TRL's `GKDTrainer` for distillation. | It is the trainer the design names, and one run is cheap. |
| R16 | The vLLM paths are verified on the first GPU smoke run. | They cannot be installed locally. |
| R17 | GRPO uses `loss_type="dr_grpo"` with `scale_rewards="none"`; distillation batch is 4 × 8 accumulation; checkpoint-step validation and config-arithmetic tests are added. | Dr. GRPO, as defined, drops both length normalisation and std scaling. The other changes prevent paid GPU failures. |
| R18 | Pre-approved spending gate: launch if the projection fits the cap, applying the design's cut order otherwise (later amended by R20 and R40). | Lets the grid run unattended. |
| R19 | `evaluate_checkpoint` takes `total_steps`, so `frac_trained` = step / total_steps. | The evaluator cannot otherwise know the run's horizon. |
| R20 | Budget ceiling raised from the design's $75 to $100. In practice the provider account allowed $30 of credits plus a $20 spend cap (R40). | Budget decision. |
| R21 | The base KL cache is built once before parallel evaluations; the vLLM engine is torn down after generation and leaves room for the KL scorer. | Avoids out-of-memory failures on the first paid evaluation. |
| R22 | lm-eval benchmarks keep their native generation limits (the 512-token cap applies only to our own generation). | Keeps scores comparable to published numbers; avoids truncating IFEval. |
| R23 | A real trimmed lm-eval output replaces the hand-written test fixture. | Test realism. |
| R24 | Evaluator hardening: stronger KL and GSM8K tests, error paths, full lm-eval version records, smoke flags, fp32 KL scoring, KL-cache spec validation. | The evaluator produces every number in the study. |
| R25 | Explainer-video sequencing (out of scope for this repository). | Process only. |
| R26 | The base KL cache is built on the same GPU class used for checkpoint scoring. | Avoids a constant device offset in KL. |
| R27 | Result rows record `wise_alpha` and `init_from`; `analyze.py` also writes `tables.tex`, and `--check-writeup` accepts `.tex` files. | Analysis needs to tell Q4 rows apart; the write-up is LaTeX. |
| R28 | Explainer-video plan (out of scope for this repository). | Process only. |
| R29 | Self-generated data sets write `dataset_stats.json` (kept fraction, correct per prompt, gold-fallback rate), exported with the results. | Disclosure of what each training set contains. |
| R30 | `arm` values are q0–q4; `init_from` is `<source run_id>/step-<n>`; Q0 selection also requires no noise, mixing, LoRA or WiSE. | Analysis links runs to their sources unambiguously. |
| R31 | `seed_band` returns a `SeedBand` (mean, std, k, n, target); `check-writeup` reads `numbers.json`. | The band must carry how many seeds reach the target. |
| R32 | The Modal image is the authoritative runtime (torch 2.14.0, transformers 5.20.2, TRL 1.18.2, vLLM 0.33.0), pinned in `requirements-modal.txt` and recorded per run. | vLLM forces those versions on GPU; the local CPU environment differs. |
| R33 | GRPO's vLLM sampling seed is keyed on the run seed. | Otherwise all seeds share rollout randomness and seed variance is understated. |
| R34 | lm-eval is consolidated into as few calls per checkpoint as possible; evaluation runs on a separate, cheaper GPU class (L4). | Evaluation cost dominated the budget. |
| R35 | Evaluator and launcher fixes: validate the KL cache and base row before paid evaluation, slice versions per benchmark, build the general pool in a dedicated step, skip already-done work client-side. | Prevents wasted GPU time. |
| R36 | The orchestrator builds the general pool and the base row once before starting parallel runs. | Avoids races and duplicate cost. |
| R37 | lm-eval stays at 3 calls per checkpoint (2 calls measured slower). Evaluation is made deterministic with vLLM batch-invariant mode; two identical runs matched on every benchmark. | Repeated runs had differed by over a point on IFEval and HumanEval, which would have hidden real differences. |
| R38 | Small follow-ups: KL sample generation runs under the evaluation vLLM settings, langdetect is seeded and recorded, GPU-seconds are recorded per row, and pre-deterministic smoke rows are archived. | Reproducibility. |
| R39 | Pilot finding: SFT on GSM8K gold solutions drops GSM8K by 20–23 points (terse gold style). The SFT arm becomes SFT on teacher answers (superseded in detail by R42). | Gold-only SFT cannot reach matched accuracy. |
| R40 | Stage 1 = Q0 + Q1 at 3 checkpoints per run, with a spend guard that stops launching at $41.50 committed. Q2–Q4 wait for budget. | The account allowed $30 of credits plus a $20 spend cap, and the cap was not raised; the guard keeps a margin below $50 for billing lag. |
| R41 | Launcher safety: dependency checks, concurrency cap, per-config locks, appended logs, budget guard using measured unit costs. | Unattended overnight launch. |
| R42 | Follow the paper's methods: SFT on stronger-model answers (up to 16 samples, keep 1 correct), the paper's LR ranges (SFT {1e-5, 3e-5, 7e-5}, GRPO {1e-5, 3e-5, 5e-5}), no SFT-gold arm. | Fidelity to the paper (App. B.1, Table 2). |
| R43 | The teacher is Qwen2.5-7B-Instruct rather than DeepSeek R1 or an R1-distilled 7B model. | R1 API is about $30 over budget; R1-distilled traces add about $12, are 1.5–2k tokens long and risk OOM and truncation for a 0.5B student. |
| R44 | Launcher accounting fixes: charge runs already in flight, survive Modal errors while polling, budget the teacher data job, per-run timeout. | Prevents overspend after a crash or relaunch. |
| R45 | Priority: Q0 (main result) > Q1 > Q2 > Q3; Q4 and the staleness experiment are extras. | The three follow-up questions come after the main result. |
| R46 | Paper fidelity before the GRPO launch: GRPO constant LR with 50 warmup steps, 1 epoch (234 steps); SFT-like runs evaluated at the end of epochs 1 and 2; 32 prompts × 8 kept and documented; per-sequence KL log-ratios and token counts stored. Q0-only R² is the KL headline. | Paper Table 2; the paper's 0.71 covers SFT and RL only. |
| R47 | Forced deviation: all three paper-range GRPO LRs diverged at 0.5B, so GRPO is re-swept at {1e-6, 3e-6}. Diverged runs are labelled and excluded from frontiers and the KL fit, and shown in an appendix (now in figures/tables.md; the PDF has no appendix). | The paper's 3B learning rates do not train a 0.5B full fine-tune. |
| R48 | No SFT LR below the paper range and no extra epochs at first; Q0 uses the paper's protocol (frontier, 2-point filter, exponential fit) plus matched accuracy where both methods have points (amended by R50). | Avoids inventing settings. |
| R49 | Spend order for the remaining budget: GRPO low LRs, GRPO seeds, SFT on GRPO outputs, teacher SFT and offline RFT seeds, iterative RFT, self-rewrite; stop when the guard refuses. | Priority under a fixed budget. |
| R50 | The forced LR deviation is applied to both methods: teacher SFT and offline RFT also get 3e-6. GRPO's larger forward KL is treated as a measured finding, not a bug. Dead-run early stop for GRPO; launcher rate-limit handling. | Fairness between methods. |
| R51 | Write-up scope: LaTeX PDF; Q2–Q4 stated as not run with a costed plan; KL analysis from stored fields only; proposal result slots filled from measured numbers (superseded by R54). | No further budget. |
| R54 | Stage 2 after the budget was raised to $100 in total: Q2 (label and reward noise), Q3 (general-data mixing), the rest of Q1 (iterative RFT, self-rewrite seeds) and Q4 (WiSE-FT, LoRA, distillation), launched in that order under a $96 spend guard. Arms use the best learning rates (teacher SFT 3e-6, GRPO 1e-6). Corrupted SFT labels are other problems' teacher answers whose final answer differs from the true one. Q3 adds a secondary criterion fixed before any Q3 result: the same crossover rule applied to IFEval alone. The teacher stays Qwen2.5-7B-Instruct. | More budget; the $4 margin covers billing lag. IFEval was the one benchmark on which teacher SFT forgot on every seed in Q0. |
| R56 | Stage-2 deviations, recorded for the write-up: (1) GKD distillation prompts are capped at 256 tokens (the 2,000 Tulu prompts are drawn from pool rows with prompts ≤ 256 tokens); (2) GKD ran at LR 1e-5, constant, never rescaled under R47/R50, on plain user prompts without the GSM8K system prompt, so its result is reported as inconclusive; (3) Q3 general (Tulu 3) rows are trained under the GSM8K system prompt, a confound for the Q3 IFEval result; (4) the Q3 seeds were placed at f = 0.02 and 0.05, the bracket of the 1-seed crossover; (5) each run evaluates its checkpoints serially on one GPU; (6) the async-staleness experiment was not run because there is no lagged sampler. | (1) Full-length prompts ran an A100-40GB out of memory; (2) `DISTILL_LR` predates the LR rescaling and one run cannot separate the method from its settings; (3) the spec's prompt applies to every row; (4) the measured crossover replaced the plan's placeholder; (5) the workspace's 10-GPU cap; (6) R54 allowed it only if a sampler existed. |
