# DENA: Distributed Expert Neural Architecture

**DENA (Distributed Expert Neural Architecture)** is a preliminary modular AI architecture proposed by **Harsha Sai Reddy Eada**.

DENA is designed around the idea that AI systems can be built from independently trained expert subsystems coordinated through:

- task decomposition
- dynamic expert routing
- weighted aggregation
- verification
- memory
- feedback-driven modular updates

This repository contains the first public research package for DENA, including the research paper, prototype implementation, experiments, DENA-Bench v0 benchmark, diagrams, and benchmark results.

> **Important note:** DENA v0 is an early research prototype. The current implementation uses toy/rule-based experts and controlled benchmark tasks. These results demonstrate implementability and measurable cost-aware expert routing, but they do **not** prove superiority over production dense models or true Mixture-of-Experts systems.

---

## Paper

The paper is available in:

```text
paper/DENA_paper_Harsha_Sai_Reddy_Eada_v1.pdf
```

Suggested title:

```text
DENA: A Preliminary Distributed Expert Neural Architecture for Modular Expert Routing and Feedback-Driven Learning
```

Author:

```text
Harsha Sai Reddy Eada
Independent Researcher
Email: harshasaireddyind@gmail.com
GitHub: https://github.com/harsha-sai-q
```

---

## Repository Structure

```text
paper/
  DENA_paper_Harsha_Sai_Reddy_Eada_v1.pdf
  DENA_paper_source_package_v1.zip

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
  dena_architecture_diagram.png
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
git clone https://github.com/harsha-sai-q/DENA-Distributed-Expert-Neural-Architecture.git
cd DENA-Distributed-Expert-Neural-Architecture
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

This supports the early architectural claim that DENA can perform cost-aware expert selection in a controlled prototype setting.

Again, this is a preliminary result using toy experts and should not be interpreted as proof that DENA outperforms real dense or MoE systems.

---

## Citation

If you use or discuss this work, please cite:

```bibtex
@misc{eada2026dena,
  title={DENA: A Preliminary Distributed Expert Neural Architecture for Modular Expert Routing and Feedback-Driven Learning},
  author={Eada, Harsha Sai Reddy},
  year={2026},
  note={Preprint}
}
```

A DOI will be added after Zenodo/OSF publication.

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

## Status

This project is currently in the **preliminary research/prototype** stage.

Next goals:

- replace toy experts with neural experts
- train a real router
- improve the aggregator and verifier
- expand DENA-Bench
- compare against stronger dense, MoE, ensemble, RAG, and agent baselines
- submit to preprint/workshop platforms for feedback
