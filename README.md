# MAVE: Marginal-Value-Aware Feedback Acquisition for Retrosynthetic Planning

Official implementation of **MAVE**, introduced in:

> **Escalate Only When Necessary: Marginal-Value-Aware Feedback Acquisition for Retrosynthetic Planning**

MAVE formulates feedback acquisition in retrosynthetic planning as a **sequential stop-or-escalate process**. Instead of fixing the amount of external feedback in advance, the planner progressively acquires stronger feedback and stops when the current evidence is sufficient.

## Method

For each planning state, MAVE follows an ordered feedback hierarchy:

```text
L0  No feedback
 ↓
L1  Structural feedback
 ↓
L2  Evaluative feedback
 ↓
L3  Comparative feedback
 ↓
L4  Strategic feedback
```

The escalation policy \(\mu\) repeatedly chooses between `STOP` and `ESCALATE`, after which the reaction policy \(\pi\) selects among candidate retrosynthetic reactions.

During training, \(K\) complete escalation rollouts from the same planning context produce reward–cost observations

\[
\{(c_k,r_k)\}_{k=1}^{K}.
\]

MAVE fits a local reward–cost mapping

\[
\phi(c)=a+b\tanh[\kappa(c-\tau)]
\]

and computes

\[
D_k^{(1)}=\phi'(c_k),
\qquad
D_k^{(2)}=\phi''(c_k).
\]

The cost-aware outcome is

\[
J_k=r_k-\lambda c_k,
\]

and the final advantage is

\[
A_k^{\mathrm{MAVE}}
=
\hat J_k
+
\beta_1\hat D_k^{(1)}
+
\beta_2\hat D_k^{(2)}.
\]

This rollout-level advantage is used to optimize the escalation policy with GRPO.

At inference time, the learned escalation and reaction policies are integrated into an **AND-OR MCTS planner**. Reward–cost fitting is only required during training.

## Feedback hierarchy

| Level | Feedback |
|---|---|
| L0 | None |
| L1 | Activity assessment, reaction class, reaction center, bond disconnection |
| L2 | Reaction feasibility, reaction yield, selectivity |
| L3 | Reaction comparison, route comparison |
| L4 | Goal suggestion, multi-step strategy |

Reaction yield is used as the cost reference with base cost 1.0. Query-specific costs vary with acquisition complexity.

## Main settings

| Hyperparameter | Value |
|---|---:|
| Backbone | Llama-3-8B-Instruct |
| Maximum feedback level | 4 |
| Rollout group size \(K\) | 4 |
| \(\lambda\) | 0.5 |
| \(\beta_1,\beta_2\) | 1.0, 1.0 |
| Curve regularization \(\eta\) | 0.001 |
| Reward–cost mapping | Tanh |
| Learning rate | \(3\times10^{-5}\) |
| GRPO clipping | 0.2 |
| KL coefficient | 0.001 |
| Discount factor \(\gamma\) | 0.95 |

The single-step retrosynthesis model retains the top-50 candidate reactions. Main experiments use a budget of 100 single-step model calls, with additional evaluations at 200, 500, and 1000 calls.

## Repository

```text
MAVE-retro/
├── configs/
├── data/
├── src/mave_repro/
│   ├── chemistry/
│   ├── environment/
│   ├── models/
│   ├── oracle/
│   ├── mave/
│   ├── planning/
│   ├── training/
│   └── evaluation/
├── scripts/
└── tests/
```

## Installation

```bash
conda create -n mave python=3.10 -y
conda activate mave
pip install -e .
```

## Training

```bash
python scripts/train.py \
    --config-dir configs \
    --oracle hierarchy \
    --runtime-factory runtime:build_runtime \
    --output-dir outputs/train
```

## Evaluation

Main 100-call setting:

```bash
python scripts/evaluate.py \
    --benchmark uspto190 \
    --oracle hierarchy \
    --runtime-factory runtime:build_runtime \
    --budget 100 \
    --output outputs/eval/uspto190.json
```

Budget scaling:

```bash
python scripts/eval_budget_scaling.py \
    --benchmark uspto190 \
    --oracle hierarchy \
    --runtime-factory runtime:build_runtime \
    --budgets 100 200 500 1000 \
    --output-dir outputs/budget_scaling
```

## Metrics

We report:

- **Success rate**: fraction of targets with a complete retrosynthetic route.
- **Route quality**: target-normalized route score based on per-reaction yield quality.
- **Query rate**: fraction of planning iterations that invoke external feedback.
- **Query cost**: cumulative feedback acquisition cost per planning episode.

## Citation

```bibtex
@inproceedings{mave2027,
  title={Escalate Only When Necessary: Marginal-Value-Aware Feedback Acquisition for Retrosynthetic Planning},
  author={Anonymous},
  booktitle={International Conference on Learning Representations},
  year={2027}
}
```