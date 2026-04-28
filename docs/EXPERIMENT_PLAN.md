# DENA Experiments v0

This experiment package tests the first working prototype of Distributed Expert Neural Architecture (DENA).

The goal is not to prove that DENA is better than all AI architectures yet.
The goal is to test whether the core architectural claims are measurable.

## Core claims tested

1. DENA can route different input types to different experts.
2. DENA can handle mixed-domain queries using more than one expert.
3. DENA can estimate reliability using expert confidence and verification.
4. DENA can be compared against simple baseline systems.
5. DENA can record traceable routing decisions for interpretability.

## Experimental systems

### DENA_top2

Uses the DENA router and selects the top 2 experts per subtask.

### DENA_top1

Uses the DENA router and selects only the top 1 expert per subtask.
This tests sparse expert selection.

### AllExperts_proxy

Selects all experts for every subtask.
This acts like a simple ensemble baseline.

### GeneralOnly_proxy

Uses only the general expert.
This acts as a small proxy for a dense/general model baseline.

## Metrics

### Route Hit

Whether at least one expected expert was selected.

\[
RouteHit = 1[\hat{S} \cap S^* \neq \emptyset]
\]

### Route Coverage

How many expected experts were selected.

\[
RouteCoverage = \frac{|\hat{S} \cap S^*|}{|S^*|}
\]

### Keyword Score

Fraction of expected keywords found in the final output.

\[
KeywordScore = \frac{1}{K}\sum_{k=1}^{K}1[keyword_k \in y]
\]

### Exact Score

Used mainly for math tasks.

\[
ExactScore = 1[\text{expected value appears in output}]
\]

### Accepted Rate

Fraction of answers accepted by the verifier.

\[
AcceptedRate = \frac{1}{N}\sum_{i=1}^{N} Accept(y_i)
\]

### Cost

Approximate total cost from selected expert profiles.

\[
Cost = \sum_{i \in S} \kappa_i
\]

### Parallel Latency

Approximate latency assuming selected experts for one subtask run in parallel.

\[
Latency = \sum_{j=1}^{m}\max_{i \in S_j}\lambda_i
\]

## How to run

```bash
python run_dena_experiments.py
```

Outputs are saved into:

```text
outputs/summary.csv
outputs/details.csv
outputs/details.json
```

## Next experiment upgrades

1. Use real benchmark datasets.
2. Replace rule-based experts with small trainable models.
3. Add a trainable router.
4. Add ablation experiments.
5. Measure expert addition/removal.
6. Measure feedback improvement across multiple rounds.
7. Compare against real dense, MoE-like, RAG, and ensemble systems.
