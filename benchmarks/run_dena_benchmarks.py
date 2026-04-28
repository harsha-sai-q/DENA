"""
Run DENA-Bench v0.

Usage:
    python run_dena_benchmarks.py

Outputs:
    outputs/benchmark_summary.csv
    outputs/benchmark_by_category.csv
    outputs/benchmark_details.csv
    outputs/benchmark_details.json
    outputs/benchmark_report.md
    outputs/quality_score_chart.png
    outputs/cost_chart.png
    outputs/routing_f1_chart.png
"""

from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple

from dena_prototype import (
    DENAEngine,
    MathExpert,
    CodeExpert,
    ResearchExpert,
    DefinitionExpert,
    GeneralExpert,
    RoutedExpert,
)


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "dena_benchmark_dataset_v0.jsonl"
OUTPUT_DIR = ROOT / "outputs"


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def clean_text(text: str) -> str:
    return " ".join(text.lower().split())


def keyword_score(answer: str, expected_terms: List[str]) -> float:
    if not expected_terms:
        return 0.0
    answer_l = clean_text(answer)
    hits = sum(1 for term in expected_terms if clean_text(term) in answer_l)
    return hits / len(expected_terms)


def exact_match(answer: str, exact_answer: str | None) -> float | None:
    if exact_answer is None:
        return None

    # Match integer/float tokens while avoiding false positives like "5" inside "0.50".
    escaped = re.escape(str(exact_answer))
    pattern = rf"(?<![\d.]){escaped}(?:\.0+)?(?![\d.])"
    return 1.0 if re.search(pattern, answer) else 0.0


def selected_experts_from_result(result) -> List[str]:
    names = []
    for sub_result in result.subtask_results:
        for routed in sub_result.selected_experts:
            names.append(routed.expert.profile.name)
    return names


def route_metrics(selected: List[str], expected: List[str]) -> Tuple[float, float, float, float]:
    selected_set = set(selected)
    expected_set = set(expected)

    if not selected_set:
        precision = 0.0
    else:
        precision = len(selected_set & expected_set) / len(selected_set)

    if not expected_set:
        recall = 0.0
    else:
        recall = len(selected_set & expected_set) / len(expected_set)

    hit = 1.0 if selected_set & expected_set else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return hit, precision, recall, f1


def estimated_cost_and_latency(result) -> Tuple[float, float, int, int]:
    total_cost = 0.0
    parallel_latency = 0.0
    activation_count = 0
    unique_experts = set()

    for sub_result in result.subtask_results:
        latencies = []
        for routed in sub_result.selected_experts:
            profile = routed.expert.profile
            total_cost += profile.cost
            latencies.append(profile.latency)
            activation_count += 1
            unique_experts.add(profile.name)
        if latencies:
            parallel_latency += max(latencies)

    return total_cost, parallel_latency, activation_count, len(unique_experts)


class RandomRouter:
    def __init__(self, experts, seed: int = 42) -> None:
        self.experts = experts
        self.rng = random.Random(seed)

    def route(self, subtask, rep, top_k: int = 2):
        k = max(1, min(top_k, len(self.experts)))
        selected = self.rng.sample(self.experts, k=k)
        return [RoutedExpert(expert=e, probability=1.0 / k, raw_score=0.0) for e in selected]


class OracleRouter:
    """
    Upper-bound router using benchmark labels.

    This is not a deployable baseline because it uses the expected experts from the benchmark.
    It is useful only to estimate the maximum score possible with ideal routing.
    """

    def __init__(self, experts, query_to_expected: Dict[str, List[str]]) -> None:
        self.experts = experts
        self.query_to_expected = query_to_expected
        self.name_to_expert = {e.profile.name: e for e in experts}

    def route(self, subtask, rep, top_k: int = 2):
        expected_names = self.query_to_expected.get(subtask.text, [])
        selected = [self.name_to_expert[n] for n in expected_names if n in self.name_to_expert]

        if not selected:
            selected = [self.experts[0]]

        # Fill if top_k asks for more experts than expected labels.
        if len(selected) < top_k:
            for expert in self.experts:
                if expert not in selected:
                    selected.append(expert)
                if len(selected) >= top_k:
                    break

        selected = selected[: max(1, min(top_k, len(selected)))]
        return [RoutedExpert(expert=e, probability=1.0 / len(selected), raw_score=1.0) for e in selected]


def build_engine(system: str, memory_path: Path, dataset: List[Dict[str, Any]]) -> DENAEngine:
    all_experts = [MathExpert(), CodeExpert(), ResearchExpert(), DefinitionExpert(), GeneralExpert()]

    if system == "DENA_top1":
        return DENAEngine(experts=all_experts, memory_path=str(memory_path), top_k=1)

    if system == "DENA_top2":
        return DENAEngine(experts=all_experts, memory_path=str(memory_path), top_k=2)

    if system == "DENA_top3":
        return DENAEngine(experts=all_experts, memory_path=str(memory_path), top_k=3)

    if system == "AllExperts_proxy":
        return DENAEngine(experts=all_experts, memory_path=str(memory_path), top_k=len(all_experts))

    if system == "GeneralOnly_proxy":
        return DENAEngine(experts=[GeneralExpert()], memory_path=str(memory_path), top_k=1)

    if system == "RandomRouter_proxy":
        engine = DENAEngine(experts=all_experts, memory_path=str(memory_path), top_k=2)
        engine.router = RandomRouter(all_experts, seed=123)
        return engine

    if system == "OracleRouter_upper_bound":
        query_to_expected = {row["query"]: row.get("expected_experts", []) for row in dataset}
        engine = DENAEngine(experts=all_experts, memory_path=str(memory_path), top_k=2)
        engine.router = OracleRouter(all_experts, query_to_expected)
        return engine

    raise ValueError(f"Unknown system: {system}")


def evaluate_system(system: str, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    memory_path = ROOT / f"memory_{system}.jsonl"
    if memory_path.exists():
        memory_path.unlink()

    engine = build_engine(system, memory_path, dataset)

    rows = []
    for item in dataset:
        result = engine.process(item["query"])
        selected = selected_experts_from_result(result)
        hit, precision, recall, f1 = route_metrics(selected, item.get("expected_experts", []))
        cost, latency, activation_count, unique_expert_count = estimated_cost_and_latency(result)

        k_score = keyword_score(result.final_answer, item.get("expected_terms", []))
        em = exact_match(result.final_answer, item.get("exact_answer"))

        if em is None:
            answer_score = k_score
            exact_applicable = 0
            exact_value = ""
        else:
            answer_score = 0.7 * em + 0.3 * k_score
            exact_applicable = 1
            exact_value = em

        accepted = 1.0 if result.accepted else 0.0
        quality_score = 0.45 * recall + 0.45 * answer_score + 0.10 * accepted
        cost_adjusted_score = quality_score - 0.10 * cost - 0.05 * latency

        rows.append(
            {
                "system": system,
                "id": item["id"],
                "category": item["category"],
                "difficulty": item["difficulty"],
                "query": item["query"],
                "expected_experts": "|".join(item.get("expected_experts", [])),
                "selected_experts": "|".join(selected),
                "route_hit": hit,
                "route_precision": precision,
                "route_recall": recall,
                "route_f1": f1,
                "keyword_score": k_score,
                "exact_applicable": exact_applicable,
                "exact_match": exact_value,
                "answer_score": answer_score,
                "accepted": accepted,
                "verification_score": result.verification_score,
                "estimated_cost": cost,
                "estimated_parallel_latency": latency,
                "activation_count": activation_count,
                "unique_expert_count": unique_expert_count,
                "quality_score": quality_score,
                "cost_adjusted_score": cost_adjusted_score,
                "final_answer": result.final_answer.replace("\n", "\\n"),
            }
        )

    return rows


def mean_of(rows: List[Dict[str, Any]], key: str) -> float:
    values = [float(r[key]) for r in rows if r[key] != ""]
    return mean(values) if values else 0.0


def summarize(details: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    systems = sorted(set(row["system"] for row in details))
    summary = []

    for system in systems:
        rows = [r for r in details if r["system"] == system]
        exact_rows = [r for r in rows if int(r["exact_applicable"]) == 1]
        summary.append(
            {
                "system": system,
                "n": len(rows),
                "route_hit": mean_of(rows, "route_hit"),
                "route_precision": mean_of(rows, "route_precision"),
                "route_recall": mean_of(rows, "route_recall"),
                "route_f1": mean_of(rows, "route_f1"),
                "keyword_score": mean_of(rows, "keyword_score"),
                "exact_match_on_exact_tasks": mean_of(exact_rows, "exact_match") if exact_rows else 0.0,
                "answer_score": mean_of(rows, "answer_score"),
                "accepted_rate": mean_of(rows, "accepted"),
                "avg_verification_score": mean_of(rows, "verification_score"),
                "avg_estimated_cost": mean_of(rows, "estimated_cost"),
                "avg_parallel_latency": mean_of(rows, "estimated_parallel_latency"),
                "avg_activation_count": mean_of(rows, "activation_count"),
                "avg_unique_expert_count": mean_of(rows, "unique_expert_count"),
                "quality_score": mean_of(rows, "quality_score"),
                "cost_adjusted_score": mean_of(rows, "cost_adjusted_score"),
            }
        )

    summary.sort(key=lambda r: r["cost_adjusted_score"], reverse=True)
    return summary


def summarize_by_category(details: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    systems = sorted(set(row["system"] for row in details))
    categories = sorted(set(row["category"] for row in details))
    rows_out = []

    for system in systems:
        for category in categories:
            rows = [r for r in details if r["system"] == system and r["category"] == category]
            if not rows:
                continue
            exact_rows = [r for r in rows if int(r["exact_applicable"]) == 1]
            rows_out.append(
                {
                    "system": system,
                    "category": category,
                    "n": len(rows),
                    "route_f1": mean_of(rows, "route_f1"),
                    "route_recall": mean_of(rows, "route_recall"),
                    "keyword_score": mean_of(rows, "keyword_score"),
                    "exact_match_on_exact_tasks": mean_of(exact_rows, "exact_match") if exact_rows else "",
                    "answer_score": mean_of(rows, "answer_score"),
                    "quality_score": mean_of(rows, "quality_score"),
                    "cost_adjusted_score": mean_of(rows, "cost_adjusted_score"),
                    "avg_estimated_cost": mean_of(rows, "estimated_cost"),
                    "avg_parallel_latency": mean_of(rows, "estimated_parallel_latency"),
                }
            )

    return rows_out


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_markdown_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        vals = []
        for col in columns:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.3f}")
            else:
                vals.append(str(val))
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + body)


def write_report(summary: List[Dict[str, Any]], by_category: List[Dict[str, Any]], details: List[Dict[str, Any]]) -> None:
    top = summary[0]
    dena_top2 = next((r for r in summary if r["system"] == "DENA_top2"), None)
    all_experts = next((r for r in summary if r["system"] == "AllExperts_proxy"), None)
    general = next((r for r in summary if r["system"] == "GeneralOnly_proxy"), None)

    summary_cols = [
        "system", "route_f1", "route_recall", "answer_score",
        "quality_score", "cost_adjusted_score", "avg_estimated_cost",
        "avg_parallel_latency", "avg_activation_count"
    ]

    category_cols = [
        "system", "category", "route_f1", "answer_score",
        "quality_score", "cost_adjusted_score", "avg_estimated_cost"
    ]

    report = []
    report.append("# DENA-Bench v0 Report\n")
    report.append("## Main result\n")
    report.append(f"Best cost-adjusted system in this toy benchmark: **{top['system']}**.\n")

    if dena_top2 and all_experts:
        cost_reduction = 1 - (dena_top2["avg_estimated_cost"] / all_experts["avg_estimated_cost"])
        report.append(
            f"DENA_top2 achieved quality score **{dena_top2['quality_score']:.3f}** "
            f"with average estimated cost **{dena_top2['avg_estimated_cost']:.3f}**. "
            f"AllExperts_proxy achieved quality score **{all_experts['quality_score']:.3f}** "
            f"with average estimated cost **{all_experts['avg_estimated_cost']:.3f}**. "
            f"That is an estimated cost reduction of **{cost_reduction * 100:.1f}%** for DENA_top2 compared with AllExperts_proxy.\n"
        )

    if general and dena_top2:
        report.append(
            f"Compared with GeneralOnly_proxy, DENA_top2 improved route F1 from "
            f"**{general['route_f1']:.3f}** to **{dena_top2['route_f1']:.3f}**.\n"
        )

    report.append("## Summary table\n")
    report.append(make_markdown_table(summary, summary_cols))
    report.append("\n## Category table\n")
    report.append(make_markdown_table(by_category, category_cols))
    report.append("\n## Correct interpretation\n")
    report.append(
        "DENA-Bench v0 supports the claim that DENA can be implemented, benchmarked, "
        "and compared using routing, answer quality, and efficiency metrics in a controlled toy setting. "
        "It does not yet prove that DENA outperforms real dense transformers or true MoE models.\n"
    )
    report.append("## Next benchmark upgrades\n")
    report.append(
        "1. Replace rule-based experts with neural experts.\n"
        "2. Add a trainable router.\n"
        "3. Use public task datasets.\n"
        "4. Add wall-clock latency and token/FLOP cost.\n"
        "5. Add ablation tests for memory, verifier, aggregation, and feedback.\n"
        "6. Compare against real dense, MoE-like, ensemble, and RAG baselines.\n"
    )

    (OUTPUT_DIR / "benchmark_report.md").write_text("\n".join(report), encoding="utf-8")


def make_charts(summary: List[Dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    systems = [r["system"] for r in summary]

    def chart(metric: str, filename: str, title: str, ylabel: str) -> None:
        values = [float(r[metric]) for r in summary]
        plt.figure(figsize=(10, 5))
        plt.bar(systems, values)
        plt.xticks(rotation=35, ha="right")
        plt.title(title)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / filename, dpi=160)
        plt.close()

    chart("quality_score", "quality_score_chart.png", "DENA-Bench v0 Quality Score", "Quality score")
    chart("avg_estimated_cost", "cost_chart.png", "DENA-Bench v0 Estimated Cost", "Average estimated cost")
    chart("route_f1", "routing_f1_chart.png", "DENA-Bench v0 Routing F1", "Routing F1")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    dataset = load_dataset(DATASET_PATH)
    systems = [
        "DENA_top1",
        "DENA_top2",
        "DENA_top3",
        "AllExperts_proxy",
        "GeneralOnly_proxy",
        "RandomRouter_proxy",
        "OracleRouter_upper_bound",
    ]

    details = []
    for system in systems:
        print(f"Running {system}...")
        details.extend(evaluate_system(system, dataset))

    summary = summarize(details)
    by_category = summarize_by_category(details)

    write_csv(OUTPUT_DIR / "benchmark_details.csv", details)
    write_csv(OUTPUT_DIR / "benchmark_summary.csv", summary)
    write_csv(OUTPUT_DIR / "benchmark_by_category.csv", by_category)

    with (OUTPUT_DIR / "benchmark_details.json").open("w", encoding="utf-8") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)

    write_report(summary, by_category, details)
    make_charts(summary)

    print("\nSummary:")
    for row in summary:
        print(
            f"{row['system']}: "
            f"quality={row['quality_score']:.3f}, "
            f"cost_adjusted={row['cost_adjusted_score']:.3f}, "
            f"route_f1={row['route_f1']:.3f}, "
            f"answer={row['answer_score']:.3f}, "
            f"cost={row['avg_estimated_cost']:.3f}, "
            f"latency={row['avg_parallel_latency']:.3f}"
        )

    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
