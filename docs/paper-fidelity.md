# Paper-fidelity audit: Q0 (SFT vs GRPO) pipeline vs RL's Razor (arXiv 2509.04259)

First audited 2026-11-22 before the grid ran; updated to the final state of the branch after all runs (stage 1 and stage 2, §4)
and their analysis. File:line references were re-checked against the final code; items in §2 marked "fixed" describe the code
as it was when audited. Ruling numbers (R1–R56) refer to the decision log `docs/rulings.md`. Page numbers are PDF pages of `reports/paper/rl-razor-2509.04259.pdf`. Quotes are verbatim,
with line breaks and hyphenation from the PDF text layer removed.

Status key: **MATCH** · **DEV-forced** (forced by the assignment or the budget) · **DEV-unforced** · **SILENT** (the paper does not say).

Fixed by the assignment: Qwen2.5-0.5B-Instruct instead of 3B, GSM8K instead of Open-Reasoner-Zero, small budget.
Rulings that override the original study design: R13 (fp32 master weights), R14 (GRPO constant LR), R17 (Dr. GRPO without std
scaling), R22 (native lm-eval generation lengths), R40 (3 ckpts/run), R42/R43 (teacher SFT, paper LR ranges,
Qwen2.5-7B-Instruct teacher), R45 (Q0 first).

## 1. Traceability table

### Data

| Step | Our choice (file:line / config) | Paper quote (section, page) | Status |
|---|---|---|---|
| New-task data | GSM8K `openai/gsm8k` main, 7,473 train / 1,319 test (`data/gsm8k.py:27-39`) | "Qwen 2.5 3B-Instruct (Qwen et al., 2025) trained on math questions from the Open-Reasoner-Zero dataset (Hu et al., 2025)." (§3.1, p4) | DEV-forced (assignment) |
| Same prompts for both methods | GRPO uses all 7,473 train prompts (`pipeline.py:237,247`). SFT uses the teacher set, which is the same prompts minus those with no correct teacher answer (`data/teacher.py:58-65`) | "For each new task, we fine-tuned models using the same set of prompts." (§3.1, p4) | MATCH (the paper's SFT set also drops unannotated prompts: "valid annotations for 96% of the dataset") |
| Prompt format | Qwen chat template, system prompt "Please reason step by step, and put your final answer within \boxed{}." (`data/gsm8k.py:5,20-24`) | — | SILENT |
| SFT annotation source | Qwen2.5-7B-Instruct teacher (`data/teacher.py:22`; R43) | "we queried DeepSeek R1 (Guo et al., 2025a), sampling up to 16 responses per prompt and retaining a single response that matched the correct final answer." (App. B.1, p18) | DEV-forced (budget, R42/R43: the R1 API is about $30 over budget; R1-Distill-7B would add about $12) |
| SFT annotation procedure | Adaptive sampling, 4 per round up to 16 per prompt; keep 1 uniformly random correct answer; drop prompts with none (`data/teacher.py:23-24,32-65`; `data/rft.py:16-30`) | same quote; "This yielded valid annotations for 96% of the dataset." (App. B.1, p18) | MATCH in intent; DEV-unforced in effect: the four rounds used overlapping sampling seeds (round r used seeds r to r+3; `data/teacher.py:46`, `gen.py:132-133`; vLLM gives the n samples of a seeded request the seeds seed + index), so a prompt got up to 16 draws from 7 distinct seeds. This can only lower the keep rate (98.1 %). Which correct response to keep is SILENT; random is reasonable. |
| Teacher sampling | T=1.0, top-p 1, 512 new tokens (`data/teacher.py:25-26`; `gen.py` Sampler) | — | SILENT |
| GRPO reward | Binary: 1 if the last `\boxed{}` matches the gold number, else 0 (`reward.py:40-58`) | "In RL training, we used only a binary success indicator as the reward, without explicit KL regularization." (§3.1, p4) | MATCH |

### SFT hyperparameters

| Step | Our choice (file:line / config) | Paper quote (section, page) | Status |
|---|---|---|---|
| Optimizer | `optim="adamw_torch"` (`train/sft.py:33`) | Table 2, SFT column: "Optimizer adamw" (p19) | MATCH |
| LR set | {1e-5, 3e-5, 7e-5} (q0_sweep), plus 3e-6 (q0_lowlr, R50) (`scripts/make_configs.py`) | Table 2: "Learning Rate {1e-5, 3e-5, 5e-5, 7e-5, 9e-5}" (p19) | DEV-forced (budget: a 3-value subset of the paper's range, R42). 3e-6 is below the paper's range: DEV-forced (R50, applied symmetrically with GRPO's R47; see §3) |
| Scheduler | `constant_with_warmup` (`train/sft.py:31`) | Table 2: "LR Scheduler {constant w. warmup, cosine w. warmup}" (p19) | DEV-forced (budget: one value of the sweep) |
| Warmup | 50 steps (`train/sft.py:14,32`) | Table 2: "Warmup steps 50" (p19) | MATCH |
| Epochs | 2 epochs (`config.py:31`, `train/sft.py:23`), evaluated at exactly 2 checkpoints, the end of epoch 1 and of epoch 2 (`make_configs.py` `SFT_N_CKPTS=2`; R46) | Table 2: "Epochs {1,2}" (p19) | MATCH (fixed by R46: with a constant LR the epoch-1 checkpoint is the 1-epoch model; it lands at `round_half_up(total/2)`, at most one step from the exact boundary) |
| Batch | Effective 32 (8 × 4 accumulation) (`train/sft.py:12-13,28-29`) | Table 2: "Batch Size {16,32,64,128}" (p19) | DEV-forced (budget: one value of the sweep) |
| Grad clip | 1.0 (`train/sft.py:35`) | Table 2: "Max Grad Norm 1" (p19) | MATCH |
| Weight decay | 0.0 (`train/sft.py:34`) | Table 2: "Weight Decay 0" (p19) | MATCH |
| Precision | `bf16=True` AMP with fp32 master weights (TRL 1.18.2 loads in fp32 by default; `train/__init__.py:10-15`). Checkpoints saved in bf16 (`train/schedule.py` `save_full_model`) (R13) | Table 2: "bfloat16 True" (p19). §6 also implies bf16 *weights*: "the reason for the observed sparse updates was the use of bfloat16 for model training … Performing the same training with float32 resulted in models with identical performance but without any sparsity in their weight updates." (p9) | MATCH at the flag level. Weight dtype differs (DEV-unforced, deliberate, low impact: by the paper's own p9 statement fp32 gives "identical performance"). |
| Loss | Completion-only cross-entropy (`train/sft.py:36`); TRL default `max_length=1024` | "LSFT(π) = −E x∼D, y∼πβ [log π(y\|x)]" (§5.1, p7) | MATCH. Completion-only masking is SILENT. |

### GRPO hyperparameters

| Step | Our choice (file:line / config) | Paper quote (section, page) | Status |
|---|---|---|---|
| Algorithm / loss | TRL `GRPOTrainer`, `loss_type="dr_grpo"`, `scale_rewards="none"` (`train/grpo.py:57-58`; R17) | Table 2: "Loss type Dr. GRPO (Liu et al., 2025)" (p19). §5.1: "A(x, y) is an Advantage function, which is the reward of y normalized with respect to other rewards for the same x." (p7) | MATCH with Dr. GRPO as Liu et al. define it (no std scaling). **Ambiguous:** "normalized" on p7 could mean std-scaled (TRL's `dr_grpo` with its default `scale_rewards="group"` is what an off-the-shelf user would get). We follow the Dr. GRPO definition, which is the most plausible reading. |
| KL regularization | `beta=0.0` (`train/grpo.py:59`) | Table 2: "KL reg. 0" (p19). "without explicit KL regularization" (§3.1, p4) | MATCH |
| Group size / prompts per step | 8 completions per prompt, 32 prompts per step = 256 completions per optimizer step, one generation per step (`train/grpo.py:12-14,44-67`) | Table 2, RL column: "Batch Size See Below", "Group Size 64", "Prompts per generation 8" (p19) | **Ambiguous in the paper.** Reading A (literal GRPO terms): G=64 completions per prompt × 8 prompts = 512 per step. Then our group is 8× smaller: DEV-forced (budget, as noted in R42). Reading B (the labels map onto TRL fields: per-step batch of 64 completions, `num_generations`=8): 8 prompts × 8 completions. Then our group size matches and our prompts per step are 4× larger: DEV-unforced but benign (larger, lower-variance batch). Reading A is the literal one; B fits "Batch Size: See Below" and is cheaper per run. Neither is decisive. |
| μ (num_iterations) | 1 (`train/grpo.py:63`) | Table 2: "num iterations (µ) {1,2}" (p19) | DEV-forced (budget: one value of the sweep) |
| LR set | {1e-5, 3e-5, 5e-5} (q0_sweep: **all three diverged**), then {1e-6, 3e-6} (q0_lowlr, R47) (`make_configs.py`) | Table 2: "{1e-5, 2e-5, 3e-5, 4e-5, 5e-5}" (p19) | DEV-forced (R47: the paper's 3B range diverges for the 0.5B full fine-tune; see §3) |
| Scheduler / warmup | `lr_scheduler_type="constant_with_warmup"`, `warmup_steps=50` (`train/grpo.py`; R46) | Table 2, RL column: "LR Scheduler constant w. warmup", "Warmup steps 50" (p19) | MATCH (fixed by R46; R14 only rules out decay) |
| Steps / epochs | `grpo_steps=234` × 32 prompts = 1 epoch of 7,473 (`make_configs.py` `GRPO_STEPS`; R46), 3 evenly spaced checkpoints (78/156/234). Runs stop early once dead (reward 0 and clipped ratio > 0.9 for 20 consecutive steps; `train/grpo.py` `DeadRunStop`, R50) and are labelled diverged | Table 2, RL column: "Epochs 1" (p19) | MATCH (fixed by R46). The dead-run stop is SILENT in the paper; it only cuts runs that have already collapsed. |
| Optimizer / clip / WD | TRL/transformers defaults: AdamW (`adamw_torch_fused`), `max_grad_norm=1.0`, `weight_decay=0.0` (not set in `grpo.py`; checked in TRL 1.18.2 / transformers 5.24 sources) | Table 2: "adamw", "Max Grad Norm 1", "Weight Decay 0" (p19) | MATCH (through library defaults; setting them explicitly in `grpo.py` would protect against version drift) |
| Precision | Same as SFT (`train/__init__.py:15`, R13) | Table 2: "bfloat16 True" (p19) | Same as the SFT row |
| Rollout sampling | T=1.0, top-p 1, top-k 0, 512 max new tokens, vLLM colocate seeded per run (`train/grpo.py:48,61-62,75-76`; engine seed `vllm_engine_seed`, `train/grpo.py:83-106`). TRL 1.18.2 defaults also apply vLLM importance-sampling correction. | — | SILENT |

### Selection, evaluation, KL

| Step | Our choice (file:line / config) | Paper quote (section, page) | Status |
|---|---|---|---|
| Checkpoint selection / frontier points | Every saved checkpoint is a point: 3 evenly spaced for GRPO (78/156/234), 2 for SFT-like runs (end of epochs 1 and 2, R46), 4 rounds for iterative RFT (`make_configs.py` `N_CKPTS`/`SFT_N_CKPTS`, R40/R46; `train/schedule.py` `ckpt_steps`). The base model is added to every method's point set (`scripts/analyze.py:240`). | "each point corresponds to a trained model with a different set of hyperparameters" (§3.1, p4; Fig. 2 caption p4: "Each point corresponds to a model trained with a different set of hyperparameters"). Mid-training checkpoints are mentioned only for MNIST: "Including mid-training checkpoints, this produced approximately 500 runs per method." (App. B.3, p20) | DEV-forced (budget, R40). It is defensible: with a constant LR, an intermediate checkpoint is the final model of a shorter run. Adding the base point is SILENT. |
| New-task evaluation | GSM8K test, all 1,319, greedy, max 512 tokens, last-`\boxed{}` numeric match (`eval/gsm8k.py:9-15`, `reward.py:40-45`) | "We measured performance on the held-out test set of the newly introduced task" (§3.1, p4). "For Math and Science Q&A, accuracy was measured by comparing the model's final answer to the ground truth, ignoring intermediate reasoning chains." (App. B.1, p18) | MATCH. Decoding settings are SILENT. |
| Prior-task benchmarks | HellaSwag, TruthfulQA-MC2, MMLU, IFEval, Winogrande, HumanEval (instruct variant) through the lm-eval CLI on vLLM (`eval/forgetting.py:36-43,75-96`) | "For LLMs, we used Hellaswag (Zellers et al., 2019), TruthfulQA (Lin et al., 2021), MMLU (Hendrycks et al., 2020), IFEval (Zhou et al., 2023), Winogrande (Sakaguchi et al., 2021), and HumanEval (Chen et al., 2021)." (§3.1, p4). "using the Language Model Evaluation Harness (Gao et al., 2024)" (App. B.1, p18) | MATCH (same set, same harness) |
| Benchmark subsets | limit 500 for HellaSwag, TruthfulQA and Winogrande; 10 per MMLU subject (about 570); IFEval and HumanEval complete (`eval/forgetting.py:37-42`) | — (implies full sets) | DEV-forced (budget) |
| Metrics, shots, chat template, generation lengths | acc_norm (HellaSwag); acc (MMLU, TruthfulQA mc2, Winogrande); prompt-level strict (IFEval); pass@1 (HumanEval). Task-default shots. Chat template only for IFEval and HumanEval. Native generation lengths (R22). | — | SILENT |
| Forgetting statistic | Mean over the 6 benchmarks of (checkpoint − base) (`eval/forgetting.py:165-169`) | Fig. 2/4/11 y-axis: "Avg. Score on Previous Tasks" (p4, p7, p23) | MATCH (the same quantity shifted by the base average; frontiers, fits and R² are unchanged) |
| KL direction | KL(π0‖π): mean over samples of log π0(y) − log π(y), with y ~ π0 (`eval/kl.py:15-17`, `eval/run_eval.py:130-168`) | "is accurately predicted by Ex∼τ KL(π0‖π), the KL divergence between the fine-tuned and base policy evaluated on the new task" (§1, p2). §6: "Forward KL (Ex∼τ KL(π0‖π)), Reverse KL (Ex∼τ KL(π‖π0))" (p9) | MATCH |
| KL inputs x | 256 fixed GSM8K **test** prompts (`eval/kl.py:6-12`, seed 0) | "measured on the new task" (§2, p3); Fig. 11: "the KL divergence between the base and fine-tuned model on the new task" (p23) | MATCH in kind. Train vs test split and the number of prompts are SILENT. |
| KL responses y | 1 sample per prompt from π0 at **T=1.0**, max 512 tokens, cached once (`eval/run_eval.py:31-34,130-153`); all 256 prompts use the same sampling seed 0 (`gen.py:53`), so the samples share one random stream and are correlated across prompts | "noise from approximate KL" (§4, p6) | SILENT. This is the unbiased Monte Carlo estimator of forward KL. The paper does not describe its approximation. |
| KL granularity | **Sequence level**: per-token log-ratios summed over the response, then averaged over samples. The EOS token is not scored. fp32 scoring on the same GPU class (`eval/kl.py:25-62`, `run_eval.py:58-99`). | — | SILENT. **Scale warning:** the paper's LLM KL axes span 0–6 (Fig. 4 right, p7) and 0.5–3.0 (Fig. 11, p23). Our pilot gives 131–161 (SFT gold) and 162–300 (GRPO 3e-6, +2 pts) (`results/results.jsonl`, pilot rows). That is about 100× larger, and GRPO comes out above SFT. Possible causes: the paper may average per token; heavy-tailed T=1 samples from a 0.5B model; RL sharpening penalized by forward KL. Not scoring EOS is a small, consistent, unforced understatement. |
| Pareto frontier | Per method, non-dominated in (accuracy, avg score) (`analysis/stats.py:24-32`) | "we identify the Pareto frontier in the two-dimensional plane of new-task performance versus previous-task performance" (§3.1, p4) | MATCH |
| Pareto filtering | Keep points whose accuracy is within 0.02 of the most accurate frontier point that forgets no more than they do (`analysis/stats.py:35-45`) | "Pareto filtering. From the trained models, we retained only those lying within 2 accuracy points of the Pareto frontier." (App. B.1 step 4, p18) | MATCH (a literal reading: the distance is measured along accuracy) |
| Curve fitting | f = c + a·exp(b·(acc − acc_min)), with b of either sign. Falls back to a line if the line fits as well, and to piecewise-linear interpolation if there are fewer than 4 distinct accuracies (`analysis/stats.py:66-104`) | "Curve fitting. An exponential function was fit to the filtered points to produce the trade-off curves." (App. B.1 step 5, p18) | MATCH. The functional form is SILENT. The line and interpolation fallbacks are a minor DEV-unforced; report `Fit.kind`. |
| Matched-accuracy comparison, seed bands | Forgetting interpolated along each frontier at base+3, +6, … points; 3 seeds at the best LR (`analysis/stats.py:111-148,246-274`) | — | SILENT (our addition) |
| KL–forgetting statistic | Quadratic fit and R² (`analysis/stats.py:133-139`) over all non-diverged rows (`scripts/analyze.py`, "Forgetting vs forward KL" section); the headline is the Q0-only fit (`kl_fit_subsets`) | "The same correlation appears in our LLM experiments, with a quadratic fit achieving R2 = 0.71 (Figure 11)." (§4, p6). Fig. 11 legend: SFT, RL (p23) | Quadratic R² MATCH. **Fixed in the analysis:** the headline is now the Q0-only fit (teacher SFT + GRPO + base, R² 0.30 on 25 checkpoints); the pooled fit (0.32 on 118 checkpoints, `figures/tables.md`) and other subsets are secondary (`analyze.py` `kl_fit_subsets`). |

## 2. Unforced deviations, most consequential first

*Items 1–3 were fixed before any grid run by R46 (GRPO constant with 50 warmup steps; 234 steps = 1 epoch; SFT
evaluated at exactly epochs 1 and 2). Item 5's per-sequence log-ratios and token counts are now stored in every
result row (`kl_logratio_sums`, `kl_n_tokens`, `kl_fwd_per_token`). Kept below as the audit record.*

1. **(fixed, R46) GRPO had no warmup** (then `train/grpo.py:42-43`: `constant`, `warmup_steps=0`). Paper: constant with 50 warmup steps.
   R42 raised the GRPO LRs about 10× above the pilot's 3e-6 (to 1e-5…5e-5), so full-size Adam steps from step 1
   risk early collapse or a KL spike. That would change which GRPO points exist on the frontier.
   *Smallest fix:* `lr_scheduler_type="constant_with_warmup", warmup_steps=50`. R14's no-decay rationale still holds.
   *When:* a code change (2 lines plus a test) before the GRPO configs launch. GRPO runs that already trained must be re-run.
2. **(fixed, R46) GRPO trains for 1.71 epochs** (400 steps × 32 prompts over 7,473). Table 2 says RL "Epochs 1".
   *Smallest fix:* `grpo_steps=234` on the q0/q1/q2 GRPO configs in `make_configs.py` `_cfg`. Update
   `tests/test_make_configs.py:53`. Checkpoints then fall at 78/156/234, and GRPO cost drops by about 40%.
   *When:* configs only, before launch. After launch, keep 400 and document it instead: the step-267 checkpoint is about 1.14 epochs.
3. **(fixed, R46) The SFT checkpoints miss the paper's 1-epoch point.** Table 2 sweeps "Epochs {1,2}". With a constant LR and
   3 checkpoints we save about 0.67, 1.33 and 2 epochs.
   *Smallest fix:* `n_ckpts=4` for teacher_sft and rft_offline, giving 0.5/1/1.5/2 epochs (exact 1- and 2-epoch
   models), or `n_ckpts=2` to get exactly {1,2} with one fewer eval per run.
   *When:* configs only (now `make_configs.py:58-59`), before launch.
4. **(fixed in the analysis) The KL–forgetting R² pooled all arms** (then `scripts/analyze.py:342`; now `kl_fit_subsets`). The paper's 0.71 covers SFT and RL.
   *Fix:* report a Q0-only R² (q0 clean rows plus base) as the number compared with 0.71, and keep the pooled value as secondary.
   *When:* analysis only, at any time.
5. **GRPO prompts per step**, under Table 2 reading B (8 prompts × 8 completions per step). We use 32 prompts, 4×
   larger. This applies only if reading B is adopted. *Fix:* `PROMPTS_PER_STEP=8` (`train/grpo.py:13`) and
   `grpo_steps=934` for one epoch. The number of rollouts stays the same, but there are 4× more optimizer steps.
   A code change before launch. Optional, because the table is ambiguous.
6. **Minor, document only:** fp32 master weights (R13; the paper's p9 says fp32 gives "identical performance");
   EOS not scored in KL; the exponential fit falls back to a line or interpolation; GRPO optimizer, clip and WD
   come from library defaults rather than being set explicitly.

**Highest-risk SILENT item (not a deviation, but load-bearing for Q0's KL claim): KL scale and estimator.**
*Resolved in the analysis without re-scoring:* the grid runs already stored per-sequence log-ratios and token counts in
every result row, so no KL re-score was needed. Per-token KL, median and top samples give the same ordering and the
same R². GRPO's larger KL at the final checkpoints is diffuse (not driven by outliers or length). It co-occurs with
falling GRPO rollout entropy, which is *consistent with* sharpening; this is an untested interpretation (SFT's rollout
entropy was not logged, and no test links entropy to KL). At near-equal accuracy GRPO's KL is below SFT's (2 GRPO
checkpoints). See the write-up, Section 3. The absolute scale (up to about 160 nats per sequence against the paper's
axes of 0–6) is not comparable, because the paper does not describe its estimator.

## 3. Forced deviations (one line each, for the write-up)

- Base model Qwen2.5-0.5B-Instruct instead of Qwen2.5-3B-Instruct (assignment).
- New task GSM8K (7,473 train / 1,319 test) instead of Open-Reasoner-Zero math. Only the math task; no Science Q&A, Tool Use or robotics (assignment and scope).
- SFT annotations from Qwen2.5-7B-Instruct (T=1, up to 16 samples from 7 distinct sampling seeds, §1; keep 1 correct) instead of DeepSeek R1 (budget, R43).
- LR grids are 3 of the paper's 5 values: SFT {1e-5, 3e-5, 7e-5}, GRPO {1e-5, 3e-5, 5e-5} (budget, R42).
- SFT sweep fixed at constant-with-warmup, batch 32 and 2 epochs, instead of sweeping scheduler, batch and epochs (budget).
- GRPO μ=1 only, not {1,2} (budget).
- **RL learning rate below the paper's range (R47).** All three of the paper's GRPO LRs we ran (1e-5, 3e-5, 5e-5;
  Table 2 range {1e-5..5e-5}, tuned for Qwen2.5-3B) diverged for the 0.5B full fine-tune with warmup and 1 epoch:
  at 1e-5 the training reward rose (≈0.42 → 0.60), then declined from about step 50 and fell below 0.01 by about
  step 125 (3e-5 and 5e-5 collapsed earlier, during warmup);
  completions ran to the 512-token cap (clipped ratio ≈ 0.98), gradients vanished, and GSM8K test accuracy at the
  final checkpoint was 0.0% for all three (degenerate output, e.g. repeated `\(`). GRPO is therefore re-swept at
  {1e-6, 3e-6}; the pilot at 3e-6 was stable. Diverged runs are labelled and excluded from the frontiers, in the
  spirit of the paper's trade-off-curve filter ("we retained only those lying within 2 accuracy points of the Pareto
  frontier", App. B.1 step 4). Excluding them from the KL–forgetting fit is **our choice**: the paper does not say
  its KL fit used filtered models. Both are listed in `figures/tables.md` (section "Appendix: diverged runs"; the write-up PDF has no appendix) with the pooled R² including them (0.38, all checkpoints of both stages).
  R50 gave GRPO two low LRs (1e-6, 3e-6) and SFT-like methods one (3e-6), as ordered.
- **SFT learning rate below the paper's range (R50), for symmetry with R47.** teacher_sft and offline RFT also get
  3e-6 alongside the paper-range sweep, so both methods are compared with the same low-LR extension. At 0.5B,
  teacher SFT did not raise GSM8K above base at any paper LR (best 1e-5: 47.5% vs 48.1%), and higher LRs lowered
  it monotonically (3e-5: 38.4%, 7e-5: 27.3%).
- GRPO rollout batch of 8 completions × 32 prompts. Under the literal reading of Table 2 (group 64, 8 prompts) the group is 8× smaller (budget).
- Frontier points are every saved checkpoint (3 per GRPO run, 2 per SFT-like run) plus the base model, not only final models of distinct configs (budget, R40/R46).
- Prior-task benchmarks subsampled: HellaSwag, TruthfulQA and Winogrande at 500 each, MMLU at 10 per subject; IFEval and HumanEval complete (budget).
- Forward KL estimated from 256 GSM8K test prompts × 1 base sample at T=1, with a 512-token cap (budget; the paper does not specify its estimator).
- Sweep runs use 1 seed, with 2 extra seeds only at each method's best LR (budget; the paper reports no seeds). A seed changes the data order for every method, and the rollouts for GRPO and the RFT arms; all teacher-SFT seeds share one teacher data set (built with seed 0, `pipeline.py:194,264`), the self-rewrite seeds share one greedy rewrite set (`pipeline.py:285`, `data/rewrite.py:53`), and the three offline-RFT seeds share six of their eight samples per prompt (seed s samples with seeds s to s+7, `pipeline.py:279-280`), so SFT-side seed spreads leave out data-sampling variance.

## 4. Stage-2 deviations (Q1 remainder, Q2–Q4; R54/R56)

The paper's LLM experiments have no counterpart for Q2–Q4 (noise, data mixing, mitigations); these are the
assignment's follow-up questions, so "deviation" here means a departure from the study's own pre-registered plan
(the study design and R54). Recorded as R56 in `docs/rulings.md`.

- **Arms at the low LRs.** Q2–Q4 use the Q0 best settings (teacher SFT 3e-6, GRPO 1e-6; offline RFT 3e-6 as the second
  WiSE-FT source), not the paper's LRs (R47/R50/R54). At these LRs there is little mean forgetting for noise, mixing or
  a mitigation to change.
- **Q2 corrupted labels** are other problems' teacher answers whose final `\boxed{}` answer differs numerically from the
  true one (R54). GRPO flips each reward with probability p. The realised number of corrupted rows is not logged.
- **Q3 general rows under the GSM8K system prompt.** Every training row, Tulu 3 rows included, is built with the study design's
  prompt ("put your final answer within \boxed{}"), which the Tulu responses do not follow. This is a confound for the
  Q3 IFEval result. The realised number of mixed rows is not logged (the step counts match the intended sizes).
- **Q3 seeds at f = 0.02 and 0.05**, the bracket of the 1-seed crossover (f = 0.05 was the only qualifying fraction
  after the 1-seed sweep). The plan's placeholder bracket was (0.1, 0.25).
- **Q3 secondary criterion** (the crossover rule on IFEval alone) was added before any Q3 result (R54).
- **GKD restore pass:** LR 1e-5 constant without warmup, never rescaled when every other arm moved to 1e-6 / 3e-6
  (teacher SFT at 1e-5 already ends below base); plain user prompts without the GSM8K system prompt; the 2,000 Tulu
  prompts are drawn only from prompts of at most 256 tokens, after an A100-40GB out-of-memory failure at full length
  (full-vocabulary log-probs over prompt + 512 sampled tokens). One run. The result is reported as inconclusive.
- **Serial evaluation:** each run scores its checkpoints one after another on a single L4 (a workspace GPU cap). Results
  are unchanged; wall-clock is longer.
- **Async-staleness experiment not run:** the code has no lagged-policy sampler, and R54 allowed the experiment only if
  one existed.
- **Q4 is 1 seed** per setting; Q3 fractions f ≥ 0.1 are 1 seed.
