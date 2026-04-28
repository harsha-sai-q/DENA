# Reviewer Feedback and Response

## Reviewer Comment

A reviewer noted that most components of DENA already exist in current systems, such as:

- model orchestration (e.g., HuggingGPT)
- model registries and expert metadata
- verification and rerouting in agent pipelines
- feedback learning (MoE + RLHF)

The reviewer suggested that DENA is closer to a framework or system architecture rather than a fundamentally new neural architecture.

## Response

This feedback is valid and has been incorporated into the updated paper (v0.2).

### Clarification of Positioning

DENA is now explicitly framed as:

> A system-level architecture for distributed expert AI systems

It does **not** claim to introduce a new neural computation primitive.

### Updated Contributions

The contributions are clarified as:

1. Formalization of distributed expert pipelines
2. Unified system architecture for routing, aggregation, verification, and feedback
3. Modular expert abstraction
4. Benchmark framework (DENA-Bench v0)

### Key Distinction

DENA focuses on:

- structure
- system design
- evaluation

rather than inventing new neural operations.

### Outcome

This repositioning makes DENA:

- more accurate
- more defensible
- more aligned with real-world AI systems

## Next Steps

- strengthen experiments
- compare against stronger baselines
- move toward learned routing
- pursue arXiv submission
