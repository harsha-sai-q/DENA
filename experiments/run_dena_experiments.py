"""
Run DENA v0 experiments.

Usage:
    python run_dena_experiments.py

This script evaluates:
- DENA_top2
- DENA_top1
- AllExperts_proxy
- GeneralOnly_proxy

Outputs:
- outputs/summary.csv
- outputs/details.csv
- outputs/details.json
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from dena_prototype import (
    DENAEngine,
    MathExpert,
    CodeExpert,
    ResearchExpert,
    DefinitionExpert,
    GeneralExpert,
)


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "dena_experiment_dataset_v0.jsonl"
OUTPUT_DIR = ROOT / "outputs"


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def clean_text(text: str) -> str:
    return " ".join(text.lower().split())


def keyword_score(answer: str, keywords: List[str]) -> float:
    if not keywords:
        return 0.0
    answer_l = clean_text(answer)
    hits = 0
    for kw in keywords:
        if clean_text(kw) in answer_l:
            hits += 1
    return hits / len(keywords)


def exact_score(answer: str, expected_value: str | None) -> float:
    if not expected_value:
        return 0.0
    return 1.0 if str(expected_value).lower() in answer.lower() else 0.0


def selected_experts_from_result(result) -> List[str]:
    names = []
    for sub_result in result.subtask_results:
        for routed in sub_result.selected_experts:
            names.append(routed.expert.profile.name)
    return names


def route_hit(selected: List[str], expected: List[str]) -> float:
    return 1.0 if set(selected) & set(expected) else 0.0


def route_coverage(selected: List[str], expected: List[str]) -> float:
    if not expected:
        return 0.0
    return len(set(selected) & set(expected)) / len(set(expected))


def estimated_cost_and_latency(result) -> tuple[float, float]:
    total_cost = 0.0
    total_parallel_latency = 0.0

    for sub_result in result.subtask_results:
        sub_latencies = []
        for routed in sub_result.selected_experts:
            total_cost += routed.expert.profile.cost
            sub_latencies.append(routed.expert.profile.latency)
        if sub_latencies:
            total_parallel_latency += max(sub_latencies)

    return total_cost, total_parallel_latency


def build_engine(system_name: str, memory_path: Path) -> DENAEngine:
    if system_name == "DENA_top2":
        return DENAEngine(memory_path=str(memory_path), top_k=2)

    if system_name == "DENA_top1":
        return DENAEngine(memory_path=str(memory_path), top_k=1)

    if system_name == "AllExperts_proxy":
        return DENAEngine(memory_path=str(memory_path), top_k=5)

    if system_name == "GeneralOnly_proxy":
        return DENAEngine(
            experts=[GeneralExpert()],
            memory_path=str(memory_path),
            top_k=1,
        )

    raise ValueError(f"Unknown system: {system_name}")


def evaluate_system(system_name: str, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    memory_path = ROOT / f"memory_{system_name}.jsonl"
    if memory_path.exists():
        memory_path.unlink()

    engine = build_engine(system_name, memory_path)

    details = []
    for row in dataset:
        result = engine.process(row["query"])

        selected = selected_experts_from_result(result)
        cost, latency = estimated_cost_and_latency(result)

        k_score = keyword_score(result.final_answer, row.get("expected_keywords", []))
        e_score = exact_score(result.final_answer, row.get("expected_value"))

        details.append(
            {
                "system": system_name,
                "id": row["id"],
                "query": row["query"],
                "expected_experts": "|".join(row.get("expected_experts", [])),
                "selected_experts": "|".join(selected),
                "route_hit": route_hit(selected, row.get("expected_experts", [])),
                "route_coverage": route_coverage(selected, row.get("expected_experts", [])),
                "keyword_score": k_score,
                "exact_score": e_score,
                "accepted": 1.0 if result.accepted else 0.0,
                "verification_score": result.verification_score,
                "estimated_cost": cost,
                "estimated_parallel_latency": latency,
                "final_answer": result.final_answer.replace("\n", "\\n"),
            }
        )

    return details


def summarize(details: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    systems = sorted(set(d["system"] for d in details))
    summary = []

    for system in systems:
        rows = [d for d in details if d["system"] == system]
        summary.append(
            {
                "system": system,
                "n": len(rows),
                "route_hit": mean(float(r["route_hit"]) for r in rows),
                "route_coverage": mean(float(r["route_coverage"]) for r in rows),
                "keyword_score": mean(float(r["keyword_score"]) for r in rows),
                "exact_score": mean(float(r["exact_score"]) for r in rows),
                "accepted_rate": mean(float(r["accepted"]) for r in rows),
                "avg_verification_score": mean(float(r["verification_score"]) for r in rows),
                "avg_estimated_cost": mean(float(r["estimated_cost"]) for r in rows),
                "avg_parallel_latency": mean(float(r["estimated_parallel_latency"]) for r in rows),
            }
        )

    return summary


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    dataset = load_dataset(DATASET_PATH)

    systems = [
        "DENA_top2",
        "DENA_top1",
        "AllExperts_proxy",
        "GeneralOnly_proxy",
    ]

    all_details = []
    for system in systems:
        print(f"Running {system}...")
        all_details.extend(evaluate_system(system, dataset))

    summary = summarize(all_details)

    write_csv(OUTPUT_DIR / "details.csv", all_details)
    write_csv(OUTPUT_DIR / "summary.csv", summary)

    with (OUTPUT_DIR / "details.json").open("w", encoding="utf-8") as f:
        json.dump(all_details, f, indent=2, ensure_ascii=False)

    print("\nSummary:")
    for row in summary:
        print(
            f"{row['system']}: "
            f"route_hit={row['route_hit']:.2f}, "
            f"coverage={row['route_coverage']:.2f}, "
            f"keyword={row['keyword_score']:.2f}, "
            f"exact={row['exact_score']:.2f}, "
            f"accepted={row['accepted_rate']:.2f}, "
            f"cost={row['avg_estimated_cost']:.2f}, "
            f"latency={row['avg_parallel_latency']:.2f}"
        )

    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
