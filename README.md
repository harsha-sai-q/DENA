# DENA: Distributed Expert Neural Architecture

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19871653.svg)](https://doi.org/10.5281/zenodo.19871653)

**DENA (Distributed Expert Neural Architecture)** is a preliminary **system-level architecture and research framework** for distributed expert AI systems, proposed by **Harsha Sai Reddy Eada**.

DENA studies how independently trained expert subsystems can be coordinated through:

- task decomposition
- dynamic expert routing
- weighted aggregation
- verification
- memory
- feedback-driven modular updates

This repository contains the public research package for DENA, including the research paper, prototype implementation, experiments, DENA-Bench v0 benchmark, diagrams, and benchmark results.

> **Research status:** DENA v0 is an early prototype. The current implementation uses toy/rule-based experts and controlled benchmark tasks. These results demonstrate implementability and measurable cost-aware expert routing, but they do **not** prove superiority over production dense models or true Mixture-of-Experts systems.

---

## Current Paper

Latest title:

```text
DENA: A System-Level Architecture for Distributed Expert AI Systems
```

Latest research-grade version:

```text
paper/DENA_paper_v0_3.pdf
```

Author:

```text
Harsha Sai Reddy Eada
Independent Researcher
Email: harshasaireddyind@gmail.com
GitHub: https://github.com/harsha-sai-q
```

---

## DOI

Latest version DOI:

```text
https://doi.org/10.5281/zenodo.19871653
```

Concept DOI for all versions:

```text
https://doi.org/10.5281/zenodo.19841838
```

---

## Repository Structure

```text
paper/
  DENA_paper_v0_3.pdf
  DENA_paper_v0_3_source.zip

src/
  dena_prototype.py
  sample_run.txt

experiments/
  run_dena_experiments.py
  dena_experiment_dataset_v0.jsonl
  experiment_summary.csv
  experiment_details.csv

benchmarks/
  run_dena_benchmarks.py
  dena_benchmark_dataset_v0.jsonl
  results/
    benchmark_report.md
    benchmark_summary.csv
    benchmark_by_category.csv
    benchmark_details.csv

figures/
  dena_system_architecture_framework.png
  quality_score_chart.png
  cost_chart.png
  routing_f1_chart.png

docs/
  EXPERIMENT_PLAN.md
  DENA_BENCHMARK_PROTOCOL.md
```

---

## Quick Start

Clone the repository:

```bash
git clone https://github.com/harsha-sai-q/DENA.git
cd DENA
```

Run the prototype:

```bash
python src/dena_prototype.py --demo
```

Run one query:

```bash
python src/dena_prototype.py --query "What is DENA and how is it different from MoE?"
```

Run experiments:

```bash
python experiments/run_dena_experiments.py
```

Run benchmarks:

```bash
python benchmarks/run_dena_benchmarks.py
```

---

## DENA v0 Components

The first prototype includes:

```text
Expert_Math
Expert_Code
Expert_Research
Expert_Definition
Expert_General
```

These are simple expert modules used to test the architecture.

The DENA v0 pipeline is:

```text
Input
  -> Preprocessor
  -> Orchestrator
  -> Router
  -> Expert Models
  -> Aggregator
  -> Verifier
  -> Memory / Feedback
```

---

## DENA-Bench v0

DENA-Bench v0 evaluates the prototype on controlled toy tasks across:

```text
math_exact
code_debug
definition
architecture_research
mixed_domain
```

Systems compared:

```text
DENA_top1
DENA_top2
DENA_top3
AllExperts_proxy
GeneralOnly_proxy
RandomRouter_proxy
OracleRouter_upper_bound
```

Main metrics include:

```text
routing precision
routing recall
routing F1
answer score
quality score
cost-adjusted score
estimated expert activation cost
estimated latency
```

---

## Main Preliminary Benchmark Result

In DENA-Bench v0, **DENA_top2** nearly matched the all-expert proxy quality while using lower estimated expert activation cost.

This supports the early claim that DENA can perform cost-aware expert selection in a controlled prototype setting.

Again, this is a preliminary result using toy experts and should not be interpreted as proof that DENA outperforms real dense or MoE systems.

---

## Citation

If you use or discuss this work, please cite:

```bibtex
@misc{eada2026dena,
  title={DENA: A System-Level Architecture for Distributed Expert AI Systems},
  author={Eada, Harsha Sai Reddy},
  year={2026},
  doi={10.5281/zenodo.19871653},
  url={https://doi.org/10.5281/zenodo.19871653},
  note={Version 0.3}
}
```

---

## License

Code is released under the MIT License.

Paper, documentation, diagrams, benchmark descriptions, and non-code research materials are released under CC BY 4.0.

See:

```text
LICENSE
CONTENT_LICENSE.md
```

---

## Roadmap

Current next goals:

- prepare arXiv submission under cs.AI or cs.LG
- replace toy experts with neural experts
- train a real router
- improve the aggregator and verifier
- expand DENA-Bench
- compare against stronger dense, MoE, ensemble, RAG, and agent baselines
- submit to AI/ML workshops for feedback
