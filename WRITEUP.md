# Write-up

Andrew Qiu. Reproduction of *RL's Razor* (Shenfeld, Pari, Agrawal, arXiv 2509.04259).

The write-up is LaTeX: **[`writeup/writeup.pdf`](writeup/writeup.pdf)** (10 pages including a contents page; source
[`writeup/writeup.tex`](writeup/writeup.tex)). It holds only the setup and the answers to the questions; this file
mirrors its structure. Every decimal number in both is checked against `figures/tables.md` / `tables.tex` with
`uv run python scripts/analyze.py --check-writeup writeup/writeup.tex --out figures/`. Full per-run tables:
`figures/tables.md`. In the PDF every factual sentence carries a reference: the paper by section and page, our
results by table, figure or section and the `figures/tables.md` section, and `docs/paper-fidelity.md` for setup and
deviations; assumptions are labelled.

## The task

The take-home: reproduce RL's Razor's main result at small scale (Qwen2.5-0.5B-Instruct, GSM8K, SFT vs GRPO,
forgetting on unrelated benchmarks), then:

1. Does SFT on the model's own correct answers forget like SFT or like RL?
2. Which method holds up better when you add noise to the labels or rewards?
3. How much general data do you need to mix into SFT before it forgets as little as RL?

Section 1 summarises the paper's main idea and section 2 our setup. Section 3 is the main result (Q0) and sections
4–6 answer these three questions (Q1–Q3); each opens with whether it was answered. Section 7 explains why two of them
are not fully answered, and what would answer them. Limitations follow (section 8), and the last section, section 9,
is a short proposal for Belvedir.

## 1. The paper's main idea

Page references are to the paper's PDF ([1] = RL's Razor, arXiv 2509.04259).

- **Claim.** "Even when SFT and RL achieve the same performance on the new task, … SFT often achieves new-task gains
  by erasing prior knowledge, while RL better preserves old skills" (§1, p. 2; Pareto frontiers in §3.1, pp. 4–5).
- **An empirical forgetting law.** The degree of forgetting "is accurately predicted by E_{x~τ} KL(π₀‖π), the KL
  divergence between the fine-tuned and base policy evaluated on the new task" (§1, p. 2). A quadratic fit gives
  R² = 0.96 in the toy ParityMNIST setting and 0.71 in the LLM experiments (§4, p. 6). In ParityMNIST, SFT on an
  oracle, KL-minimal label distribution forgets even less than RL (§4, p. 6).
- **Intuition.** On-policy RL samples outputs the model already finds likely and reweights them by reward, so it
  moves to a nearby reweighted distribution; SFT pulls the model toward an external label distribution that can be
  arbitrarily far away (abstract, p. 1; §5.2, p. 8). The critical factor is on-policy data, not negative gradients (§5.1,
  p. 8).
- **Theory.** For a finite output set, a convex policy family and a binary reward, and under regularity conditions,
  policy gradient converges, among all optimal representable policies, to the one closest to the starting policy in
  KL(π‖π₀) (Theorem 5.2, §5.2, p. 8); the paper notes that neural-network policy families may not converge exactly
  (App. A, p. 18).
- **The paper's caveats.** It lacks a mechanistic account of why larger KL shifts disrupt prior knowledge, its
  behaviour at frontier scales is unknown, and it did not study off-policy algorithms (§7, pp. 9–10).

## 2. Setup

We repeat the paper's LLM math experiment (Qwen2.5-3B-Instruct, SFT vs GRPO, models kept near each method's
accuracy–forgetting Pareto frontier) at small scale.

- **Model, task, data:** Qwen2.5-0.5B-Instruct; GSM8K (7,473 train / 1,319 test prompts); answer in `\boxed{}`.
- **Teacher SFT:** answers from Qwen2.5-7B-Instruct (up to 16 samples per prompt from 7 distinct sampling seeds, see
  [`docs/paper-fidelity.md`](docs/paper-fidelity.md); one correct kept; 98.1% of prompts kept); cross-entropy on answer tokens, AdamW, batch 32, 2 epochs.
- **GRPO:** Dr. GRPO loss, binary reward, no KL penalty (β = 0), 234 steps of 32 prompts × 8 completions,
  checkpoints at steps 78, 156, 234.
- **Forced LR deviation:** all three paper-range GRPO LRs diverged at 0.5B, and teacher SFT at every paper-range LR
  stayed at or below base (best 47.5% at 1e-5). Best LRs used: teacher SFT 3e-6, GRPO 1e-6. GRPO got two low LRs and
  the SFT-like methods one; this asymmetry favours GRPO.
- **Forgetting metric:** HellaSwag, TruthfulQA, MMLU, IFEval, Winogrande, HumanEval via lm-eval on vLLM
  (batch-invariant). HellaSwag, TruthfulQA, Winogrande subsampled to 500 items, MMLU to 570; IFEval and HumanEval
  complete. Item SE 1.9 to 3.7 pp per benchmark. Shared matched-accuracy target: 51.1% GSM8K.
- **KL estimator:** forward KL(π₀‖π) on 256 GSM8K test prompts, one base sample each, 512-token cap; absolute values
  are not comparable to the paper's.
- **Seeds and compute:** 3 seeds at best LRs; Q3 at f ≥ 0.1 and SFT on GRPO outputs are single runs. A seed
  changes the data order for every method, and the rollouts for GRPO and the RFT arms; all teacher-SFT seeds share
  one teacher data set, the self-rewrite seeds one greedy rewrite set, and the three offline-RFT seeds share six of
  their eight samples per prompt, so SFT-side seed spreads leave out data-sampling variance
  ([`docs/paper-fidelity.md`](docs/paper-fidelity.md) §3). Training
  on Modal A100-40GB, evaluation on L4; total Modal spend $83.11, which also covers smoke tests, pilots, diagnostics and
  runs not reported here (Modal billing through the 22:00–23:00 UTC hour of
  2026-11-22).
- **Deviations:** listed compactly in the PDF; all of them in [`docs/paper-fidelity.md`](docs/paper-fidelity.md).
  Reproduction commands: [`README.md`](README.md).

## 3. Q0 baseline: SFT vs GRPO, and does KL predict forgetting?

**Main result (Q0): partly reproduced.** Same sign at matched accuracy: at the shared 51.1% target SFT is 0.1 to
0.6 pp worse than GRPO, within noise (final checkpoints: GRPO 0.25 ± 0.16, SFT −0.05 ± 0.19 pp, p = 0.110; GRPO
learns more, 55.1 vs 51.2% GSM8K, base 48.1%). The IFEval split, post hoc: SFT −3.5 ± 1.2 vs GRPO +1.4 ± 0.5 pp
(p = 0.010 uncorrected, 0.057 after Bonferroni). Forward-KL R² is 0.30 vs the paper's 0.71. Values are mean ± seed std over 3 seeds at each method's
best LR, final checkpoints, unless marked; forgetting is the mean change over the six benchmarks, checkpoint − base,
in pp (negative means forgetting).

At the 51.1% target, SFT − GRPO is −0.1 to −0.6 pp: the paper's sign, within noise. IFEval is the one consistent
difference (p = 0.010 uncorrected, 0.057 Bonferroni, post hoc). The clear forgetting is an LR effect: teacher SFT at
three of the paper's LRs forgets more as the LR rises (−0.5, −4.9, −12.2 pp); GRPO at its paper-range LRs (1e-5, 3e-5,
5e-5) diverged instead. The Q0-only KL fit gives R² = 0.30 (paper 0.71). GRPO's final KL
exceeds SFT's (50.52 vs 10.43 nats/seq) with no more forgetting; at near-equal accuracy the ordering flips (6.46 vs
9.67), but that rests on 2 GRPO checkpoints, which forget slightly more (−0.36 vs −0.09 pp).

## 4. Q1: SFT on the model's own correct answers

**Answered on the mean: like RL, but weakly; on IFEval (post hoc), iterative RFT forgets like SFT.** Offline and
iterative RFT ≈ GRPO on the six-benchmark mean (iterative 0.32 ± 0.53 pp, p = 0.828; offline 0.55 ± 0.33 pp,
p = 0.257), but the mean does not separate teacher SFT from GRPO either (p = 0.110), so "like RL" is weak: teacher
SFT barely forgot. On IFEval, the one benchmark that did separate them (post hoc; p = 0.010 uncorrected, 0.057 after
Bonferroni), iterative RFT still loses
2.6 ± 0.3 pp, like teacher SFT; offline RFT barely learns (49.4%).

Iterative RFT: 51.4 ± 2.2%
GSM8K, 0.32 ± 0.53 pp, but IFEval (post hoc) −2.6 ± 0.3 pp on every seed, like teacher SFT. At the paper's LRs, self-generated
data forgets like teacher data at the same LR (−12.3 vs −12.2 pp at 7e-5). Caveats: teacher SFT barely forgot
either, so the contrast is small; offline RFT barely learns; SFT on GRPO outputs is 1 seed; self-rewrite
(47.0 ± 0.8%, −0.48 ± 0.13 pp) has about 28% gold fallback rows.

## 5. Q2: label noise (SFT) vs reward noise (GRPO)

**Half answered.** *Forgetting: answered.* Noise causes no measurable forgetting for either method (all within
±0.5 pp). The finding is weak: neither method forgets much even without noise, so it cannot show which would hold
up better where SFT does forget (section 7). *Accuracy: mixed.* GRPO is ahead at p = 0 and 0.2 but loses more to noise (−2.9 vs −1.7 pp at p = 0.2;
−7.2 vs −4.7 pp at p = 0.4); the p = 0.4 gap is unresolved (p = 0.309). The data do not show that a noisy outcome
is safer as a reward than as a label.

GRPO loses more GSM8K to noise (−2.9 / −7.2 pp) than SFT (−1.7 / −4.7 pp) and stays ahead at p = 0.2 only because
its clean lead (4.0 pp) shrinks. At p = 0.4 the gap (48.0 ± 1.8 vs 46.5 ± 1.3%) is not resolved (p = 0.309).
Forgetting stays inside ±0.5 pp. No claim that reward noise is safer than label noise.

## 6. Q3: mixing general instruction data into SFT

**Not answered in this setup.** Plain SFT misses the threshold by only 0.014 pp (inside the GRPO band's sd of
0.10 pp), so there was no gap for mixing to close. The pre-registered IFEval criterion fails at every f, and that
result carries a confound: the general (Tulu 3) rows were trained under a system prompt that asks for a `\boxed{}`
answer.

Primary criterion does not discriminate (0.014 pp miss, GRPO sd 0.10 pp); no seeded f (≤ 0.05) is distinguishable
from GRPO or from unmixed SFT. f ≥ 0.1 is single-seed (final −0.67 to −1.06 pp, dominated by the IFEval drop).
IFEval criterion fails at every f; the 0 → 0.05 step is resolved (p = 0.017), the 0 → 0.02 step is not
(p = 0.097). Confound beside the result: general rows carried the GSM8K `\boxed{}` system prompt, so the IFEval
result is about this mixing recipe, not support for the paper.

## 7. Why two of the three questions are not fully answered, and what would answer them

- **It was not a budget choice.** The three GRPO learning rates we ran from the paper's range (1e-5, 3e-5, 5e-5, of
  the five in its Table 2) all diverged at 0.5B. At 1e-5, GSM8K fell to 0.0% by step 156 and stayed there at step
  234, with 512-token completions (`figures/tables.md`). Teacher SFT at the paper's 1e-5
  and 3e-5 ended at or below base accuracy (47.5% and 38.4% vs 48.1%), so there is no matched-accuracy point to
  compare. Hence the forced lower rates (R47 and R50 in `docs/rulings.md`).
- **At the rates that did train (GRPO 1e-6, SFT 3e-6), SFT barely forgets** on the six-benchmark mean
  (−0.05 ± 0.19 pp). That shrinks the SFT–RL gap that Q1–Q3 measure against to about seed noise (0.30 pp,
  p = 0.110). That is why Q2's forgetting result, though answered, is weak and Q3 cannot separate the methods, and why Q1's
  contrast is small. The open half of Q2 is accuracy, and it is open for a different reason: GRPO is ahead at p = 0
  and p = 0.2 but loses more to noise, and at p = 0.4 three seeds do not resolve the gap.
- **Where SFT does forget, it does not learn.** Mean SFT forgetting appears only at high LR (−0.5, −4.9, −12.2 pp
  at 1e-5, 3e-5, 7e-5; 1 seed each), and there it comes without a GSM8K gain (47.5, 38.4, 27.3% against base 48.1%).
- **What would answer them:** rerun on the paper's own model, Qwen2.5-3B-Instruct, at the paper's Table 2 learning
  rates. A small pilot comes first, one GRPO run and one SFT run, to confirm that GRPO trains stably there and that
  SFT forgets measurably while gaining accuracy; only then would the Q2 noise arms and Q3 mixing fractions be rerun
  against a real SFT–RL gap. This is an expectation, not a result: nothing was run at 3B.

## 8. Limitations

One small model and task at low LRs; one shared target; 3 seeds with item SE 1.9 to 3.7 pp; IFEval post hoc in Q0;
single-run arms; the Q3 confound; self-rewrite's gold fallback; the KL estimator.

## 9. Proposal for Belvedir

**Current workflow** (Belvedir's public website and job descriptions, belvedir.ai and careers.belvedir.ai). Four
steps: collect, train, evaluate, and deploy "at scale" while continuing "training as new data arrives". In detail:
pipelines for "collection, cleaning, filtering, and synthesis" of raw customer data; training by "SFT, distillation,
preference tuning, and RL", with asynchronous RL; "task-specific evaluations" that "drive decisions about data and
training recipes", and benchmark creation; and a serving stack on Kubernetes with "batching, caching, routing, and
autoscaling". Since training continues as new data arrives, every lap is another fine-tune (my inference). How a new
model is promoted, and whether serving is multi-tenant, are not stated publicly.

**The proposal** (table below) makes that loop forgetting-aware. It assumes a served model per customer,
promotion by a routing change, and retraining in discrete cycles **[A]**. Basis tags: **[P]** the paper [1], with section and page; **[D]** our data, with the section of
this write-up; **[A]** assumption or my inference.

| Stage | Change and basis |
|---|---|
| Collect | Use customer traces to define prompts and a verifier (the trace's final answer, tool result or schema check), not SFT targets. **[P]** on-policy training stays closer to the base in KL and forgets less (§3.1, p. 5; §5.1, p. 8). **[D]** at 0.5B only the sign was reproduced, within noise (p = 0.110), but GRPO learned more (55.1 vs 51.2% GSM8K; section 3). Keep each customer's earlier-cycle prompts, and only the prompts, for regression checks. **[A]** |
| Train | On-policy first: GRPO, with no KL penalty. **[P]** the paper's RL used no explicit KL regularization (§3.1, p. 4). If imitation is unavoidable, regenerate targets from the current model each round (iterative RFT). **[P]** on-policy data, not negative gradients, is the critical factor (§5.1, p. 8). **[D]** iterative RFT was indistinguishable from GRPO on the mean (0.32 ± 0.53 pp, p = 0.828) but lost 2.6 ± 0.3 pp of IFEval on every seed (section 4), so it still goes through the canary gate. Do not replay stale customer data as async RL rollouts; use old traces as prompts. **[A]** my inference; the paper did not study off-policy algorithms (§7, p. 10). No default general-data mix fraction is supported: if general data is mixed in, keep each row's own prompt format and gate the mix on the canaries. **[D]** mean forgetting could not rank the mixes, and IFEval worsened as the fraction grew from 0 to 0.05, under a confound: the general rows carried the task's system prompt (section 6). |
| Evaluate | A promotion gate on fixed, cached inputs (no training). *G1:* task gain over production, above seed noise. **[A]** the threshold comes from the customer. *G2:* forward KL to the base on cached base samples of new-task prompts, logged and *advisory only*. **[P]** KL predicts forgetting, R² = 0.71 on LLMs (§4, p. 6). **[D]** here R² = 0.30, and GRPO's KL exceeded SFT's (50.52 vs 10.43 nats/seq) with no more forgetting, so no KL threshold can be calibrated (section 3, Figure 1). *G3:* task accuracy on prior-cycle prompts against production. **[A]** untested: our study has one task. *G4, binding:* per-benchmark canaries; no single benchmark may fall below base by more than its noise band, not only the mean. **[D]** motivated by IFEval, where teacher SFT lost 3.5 ± 1.2 pp under a flat mean (−0.05 ± 0.19 pp), a split found post hoc (section 3); item SE is 1.9 to 3.7 pp per benchmark (section 2). Promote only if G1, G3 and G4 pass. |
| Deploy | Shadow or canary traffic slice first; keep the last few promoted models per customer; rollback is a router pointer change. **[A]** assumed standard practice, not tested here. |
| Retrain | Rerun the loop each cycle on the cumulative prompt pool. **[A]** If a candidate fails only on drift, repair it cheapest first: interpolate its weights back toward the base, then run a short on-policy restore pass toward the last-good model, re-gating after each step; if it still fails, keep production. **[A]** untested by our data. Re-sweep the LR whenever the model size or recipe changes. **[D]** the three paper-range GRPO LRs we ran diverged at 0.5B, and SFT forgetting grew with LR, to −12.2 pp at 7e-5 (sections 3 and 7). |

**What the proposal does not rest on.** The on-policy recommendation rests on the paper's larger-scale results; our
0.5B data did not reproduce the forgetting gap. KL is advisory, not a gate. The canary gate's motivating IFEval
result was found post hoc. The repair ladder is untested by our data, and no mix fraction is recommended.
