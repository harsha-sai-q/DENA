# DENA-Bench v0: Benchmark Protocol

## Purpose

DENA-Bench v0 is a controlled benchmark suite for the first prototype of **Distributed Expert Neural Architecture (DENA)**.

The benchmark does not claim that DENA beats real production LLMs, dense transformers, or true MoE systems yet. Its purpose is to test whether DENA's architectural properties can be measured.

## What DENA-Bench v0 measures

DENA-Bench v0 measures six dimensions:

1. **Routing correctness**  
   Whether DENA selects the correct expert or experts for a task.

2. **Expert coverage**  
   Whether DENA selects all experts needed for multi-domain tasks.

3. **Answer relevance**  
   Whether the output contains expected information.

4. **Exact answer correctness**  
   Whether exact-answer tasks, mainly arithmetic tasks, produce the expected result.

5. **Efficiency**  
   Approximate expert activation cost and parallel latency.

6. **Interpretability**  
   Whether selected experts and routing decisions can be inspected.

## Benchmark systems

### DENA_top1

DENA with Top-1 expert selection per subtask.

### DENA_top2

DENA with Top-2 expert selection per subtask.

### DENA_top3

DENA with Top-3 expert selection per subtask.

### AllExperts_proxy

An ensemble-style proxy where every expert is selected.

### GeneralOnly_proxy

A dense/general-model proxy where only a general expert is used.

### RandomRouter_proxy

A random-router baseline.

### OracleRouter_upper_bound

An upper-bound router that uses benchmark labels to select expected experts. This is not a fair deployable system, but it helps estimate the gap between the current router and ideal routing.

## Core metrics

### Routing Precision

\[
Precision = \frac{|\hat{S} \cap S^*|}{|\hat{S}|}
\]

### Routing Recall / Coverage

\[
Recall = \frac{|\hat{S} \cap S^*|}{|S^*|}
\]

### Routing F1

\[
F1 = \frac{2PR}{P+R}
\]

### Keyword Score

\[
KeywordScore = \frac{1}{K}\sum_{k=1}^{K}1[k \in y]
\]

### Exact Match

\[
ExactMatch = 1[\text{gold answer appears in output}]
\]

This is computed only for exact-answer examples.

### Answer Score

For exact-answer tasks:

\[
AnswerScore = 0.7 \cdot ExactMatch + 0.3 \cdot KeywordScore
\]

For non-exact tasks:

\[
AnswerScore = KeywordScore
\]

### Quality Score

\[
Quality = 0.45 \cdot RoutingRecall + 0.45 \cdot AnswerScore + 0.10 \cdot Accepted
\]

### Cost-Adjusted Score

\[
CostAdjusted = Quality - 0.10 \cdot Cost - 0.05 \cdot Latency
\]

The cost-adjusted score is intentionally simple for DENA-Bench v0. Future versions should use real FLOPs, token counts, wall-clock latency, memory usage, and money cost.

## Dataset categories

DENA-Bench v0 uses five toy benchmark categories:

```text
math_exact
code_debug
definition
architecture_research
mixed_domain
```

## Correct interpretation

DENA-Bench v0 can support this claim:

> DENA v0 is implementable, measurable, interpretable, and capable of cost-aware expert routing in a controlled benchmark.

It should not yet be used to claim:

> DENA beats real dense transformers or true MoE systems.

That claim requires neural experts, stronger public datasets, true baselines, and larger-scale evaluation.
