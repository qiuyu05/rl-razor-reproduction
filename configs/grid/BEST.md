# Best learning rates (Q0 sweep, updated after q0_lowlr, 2026-11-22)

Ruling numbers (R<n>) refer to the decision log `docs/rulings.md`.

Rule (study design): the best LR is the setting whose near-frontier points (within 2 GSM8K points of the
pooled frontier) dominate (strict for teacher_sft and RFT; a judgement for GRPO, see its row) in the (GSM8K accuracy, mean forgetting) plane. Source: `results/results.jsonl`
(q0_sweep + q0_lowlr, 1 seed each). Base: GSM8K 48.1%, forgetting 0 by definition. Diverged runs (R47/R50) are
not candidates.

| Method | Best LR | One-line justification |
|---|---|---|
| grpo | **1e-6** | Chosen for accuracy, not by strict dominance: 1e-6 (53.8% / +0.40 pp at step 234) and 3e-6 (52.4% / +0.56 pp) do not dominate each other on 1 seed. 1e-6 has the higher accuracy and 4.5× lower KL; 3e-6 has the slightly better forgetting score, so the choice does not favour the hypothesis. 1e-5/3e-5/5e-5 diverged. |
| teacher_sft | **3e-6** | Both points (51.4% / −0.2 pp, 51.5% / −0.3 pp) dominate every 1e-5 (≤47.5%), 3e-5 and 7e-5 point. |
| rft_offline | **3e-6** | Both points (48.8% / −0.3 pp, 49.1% / +0.6 pp) dominate every 1e-5 (≤45.0%), 3e-5 and 7e-5 point. |

3e-6 (SFT-like) and 1e-6 / 3e-6 (GRPO) are below the paper's Table 2 ranges: forced deviations R47/R50, documented
in `docs/paper-fidelity.md` §3. Part A's interim choices (teacher_sft 1e-5, offline RFT 1e-5) are superseded.
Seeds (q0_seeds) and Q1 (iterative RFT, self-rewrite, RL-distilled SFT from `grpo-lr1e-06-s0-6f9f01/step-234`)
use these LRs.
