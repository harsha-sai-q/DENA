"""
DENA v0 Prototype
Distributed Expert Neural Architecture

This is a small, dependency-free prototype that demonstrates the core DENA flow:

Input -> Preprocessor -> Orchestrator -> Router -> Expert Models -> Aggregator
      -> Verifier -> Memory/Feedback

It is intentionally simple. The goal is not to build a powerful AI model yet.
The goal is to prove the architecture works as a modular system.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import operator
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class TaskRepresentation:
    raw_input: str
    tokens: List[str]
    memory_hits: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SubTask:
    id: str
    text: str
    domain_hint: str = "general"


@dataclass
class ExpertProfile:
    name: str
    domain: str
    keywords: List[str]
    quality_score: float = 0.70
    cost: float = 0.10
    latency: float = 0.10
    available: bool = True


@dataclass
class ExpertOutput:
    expert_name: str
    subtask_id: str
    text: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutedExpert:
    expert: "BaseExpert"
    probability: float
    raw_score: float


@dataclass
class SubTaskResult:
    subtask: SubTask
    selected_experts: List[RoutedExpert]
    expert_outputs: List[ExpertOutput]
    aggregated_text: str
    reliability: float


@dataclass
class DENAResult:
    query: str
    subtasks: List[SubTask]
    subtask_results: List[SubTaskResult]
    final_answer: str
    verification_score: float
    accepted: bool
    memory_id: str


# -----------------------------
# Utility functions
# -----------------------------

STOPWORDS = {
    "a", "an", "the", "is", "are", "am", "to", "of", "in", "on", "for", "and",
    "or", "with", "by", "this", "that", "it", "as", "at", "be", "from", "can",
    "you", "me", "my", "i", "we", "our", "your", "how", "what", "why", "when",
}


def tokenize(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    return [w for w in words if w not in STOPWORDS]


def softmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    max_v = max(values)
    exps = [math.exp(v - max_v) for v in values]
    total = sum(exps)
    if total == 0:
        return [1.0 / len(values)] * len(values)
    return [e / total for e in exps]


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# -----------------------------
# Memory module
# -----------------------------

class MemoryStore:
    """
    Simple JSONL memory store.

    In a larger DENA system, this could become a vector database.
    For this prototype, we use token overlap for retrieval.
    """

    def __init__(self, path: str = "dena_memory.jsonl") -> None:
        self.path = Path(path)
        self.path.touch(exist_ok=True)

    def read_all(self) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return items

    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        q_tokens = tokenize(query)
        scored = []
        for item in self.read_all():
            item_tokens = item.get("tokens", [])
            score = jaccard(q_tokens, item_tokens)
            if score > 0:
                scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:k]]

    def append(self, record: Dict[str, Any]) -> str:
        memory_id = f"mem_{int(time.time() * 1000)}"
        record["memory_id"] = memory_id
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return memory_id


# -----------------------------
# Preprocessor
# -----------------------------

class Preprocessor:
    def __init__(self, memory: MemoryStore) -> None:
        self.memory = memory

    def transform(self, text: str) -> TaskRepresentation:
        return TaskRepresentation(
            raw_input=text,
            tokens=tokenize(text),
            memory_hits=self.memory.retrieve(text, k=3),
        )


# -----------------------------
# Orchestrator
# -----------------------------

class Orchestrator:
    """
    Splits a query into subtasks and gives each subtask a rough domain hint.
    """

    DOMAIN_KEYWORDS: Dict[str, List[str]] = {
        "math": [
            "calculate", "solve", "equation", "plus", "minus", "multiply",
            "divide", "integral", "derivative", "sum", "number", "math",
            "arithmetic", "probability", "statistics"
        ],
        "code": [
            "code", "python", "javascript", "java", "bug", "debug", "function",
            "class", "api", "error", "program", "algorithm", "compile"
        ],
        "research": [
            "architecture", "benchmark", "experiment", "research", "paper",
            "model", "router", "aggregator", "training", "expert", "dataset",
            "compare", "moe", "dense", "dena"
        ],
        "definition": [
            "define", "definition", "what", "explain", "meaning", "describe"
        ],
    }

    def decompose(self, rep: TaskRepresentation) -> List[SubTask]:
        text = rep.raw_input.strip()
        lowered = text.lower()

        detected_domains: List[str] = []
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            if any(k in lowered for k in keywords):
                detected_domains.append(domain)

        if not detected_domains:
            detected_domains = ["general"]

        # If multiple domains are present, create one subtask per domain.
        # This demonstrates DENA's decomposition idea.
        subtasks: List[SubTask] = []
        for idx, domain in enumerate(detected_domains, start=1):
            subtasks.append(
                SubTask(
                    id=f"task_{idx}",
                    text=text,
                    domain_hint=domain,
                )
            )
        return subtasks


# -----------------------------
# Expert models
# -----------------------------

class BaseExpert:
    def __init__(self, profile: ExpertProfile) -> None:
        self.profile = profile

    def keyword_relevance(self, subtask: SubTask) -> float:
        tokens = set(tokenize(subtask.text))
        kws = set(self.profile.keywords)
        if not tokens or not kws:
            return 0.0
        return len(tokens & kws) / max(1, len(kws))

    def run(self, subtask: SubTask, context: Optional[Dict[str, Any]] = None) -> ExpertOutput:
        raise NotImplementedError


class MathExpert(BaseExpert):
    def __init__(self) -> None:
        super().__init__(
            ExpertProfile(
                name="Expert_Math",
                domain="math",
                keywords=[
                    "calculate", "solve", "math", "plus", "minus", "multiply",
                    "divide", "sum", "number", "equation", "arithmetic"
                ],
                quality_score=0.82,
                cost=0.05,
                latency=0.05,
            )
        )

    def run(self, subtask: SubTask, context: Optional[Dict[str, Any]] = None) -> ExpertOutput:
        expression = extract_math_expression(subtask.text)
        if expression:
            try:
                value = safe_eval(expression)
                return ExpertOutput(
                    expert_name=self.profile.name,
                    subtask_id=subtask.id,
                    text=f"Mathematical result: {expression} = {value}",
                    confidence=0.92,
                    metadata={"expression": expression, "value": value},
                )
            except Exception as exc:
                return ExpertOutput(
                    expert_name=self.profile.name,
                    subtask_id=subtask.id,
                    text=f"I detected a math expression, but could not safely evaluate it: {expression}. Reason: {exc}",
                    confidence=0.35,
                    metadata={"expression": expression},
                )

        return ExpertOutput(
            expert_name=self.profile.name,
            subtask_id=subtask.id,
            text="I detected a math-related request, but no clear arithmetic expression was found.",
            confidence=0.45,
        )


class CodeExpert(BaseExpert):
    def __init__(self) -> None:
        super().__init__(
            ExpertProfile(
                name="Expert_Code",
                domain="code",
                keywords=[
                    "code", "python", "javascript", "java", "bug", "debug",
                    "function", "class", "api", "error", "algorithm"
                ],
                quality_score=0.78,
                cost=0.15,
                latency=0.15,
            )
        )

    def run(self, subtask: SubTask, context: Optional[Dict[str, Any]] = None) -> ExpertOutput:
        text = subtask.text.lower()

        if "debug" in text or "error" in text or "bug" in text:
            answer = (
                "Code analysis: start by isolating the failing input, reading the exact error message, "
                "checking variable types, and creating a minimal reproducible example. Then test one fix at a time."
            )
            confidence = 0.78
        elif "function" in text or "python" in text:
            answer = (
                "Code guidance: define the input, expected output, edge cases, and then write a small function with tests. "
                "For Python, keep the function pure when possible and validate with sample inputs."
            )
            confidence = 0.74
        else:
            answer = (
                "Code guidance: the request appears programming-related. Break it into requirements, algorithm, implementation, and tests."
            )
            confidence = 0.62

        return ExpertOutput(
            expert_name=self.profile.name,
            subtask_id=subtask.id,
            text=answer,
            confidence=confidence,
        )


class ResearchExpert(BaseExpert):
    def __init__(self) -> None:
        super().__init__(
            ExpertProfile(
                name="Expert_Research",
                domain="research",
                keywords=[
                    "architecture", "benchmark", "experiment", "research",
                    "paper", "model", "router", "aggregator", "training",
                    "expert", "dataset", "compare", "moe", "dense", "dena"
                ],
                quality_score=0.80,
                cost=0.20,
                latency=0.20,
            )
        )

    def run(self, subtask: SubTask, context: Optional[Dict[str, Any]] = None) -> ExpertOutput:
        answer = (
            "Research analysis: treat the idea as a modular AI-system architecture. "
            "Define components formally, implement a minimal prototype, then evaluate routing accuracy, "
            "final-task accuracy, latency, cost, modular update ability, and comparison against dense, MoE-like, ensemble, and RAG baselines."
        )
        return ExpertOutput(
            expert_name=self.profile.name,
            subtask_id=subtask.id,
            text=answer,
            confidence=0.82,
        )


class DefinitionExpert(BaseExpert):
    DEFINITIONS = {
        "dena": (
            "DENA means Distributed Expert Neural Architecture: a proposed architecture where independently trained expert systems "
            "are connected through an orchestrator, router, aggregator, verifier, memory, and feedback loop."
        ),
        "moe": (
            "MoE means Mixture of Experts: a neural architecture where a router activates a subset of internal expert networks for an input."
        ),
        "dense": (
            "A dense model is a model where most or all parameters are active for most inputs, unlike sparse expert-based models."
        ),
        "router": (
            "A router is the component that selects which expert or experts should handle a task."
        ),
        "aggregator": (
            "An aggregator combines outputs from selected experts into one final answer."
        ),
    }

    def __init__(self) -> None:
        super().__init__(
            ExpertProfile(
                name="Expert_Definition",
                domain="definition",
                keywords=["define", "definition", "what", "explain", "meaning", "describe", "dena", "moe", "dense"],
                quality_score=0.76,
                cost=0.06,
                latency=0.06,
            )
        )

    def run(self, subtask: SubTask, context: Optional[Dict[str, Any]] = None) -> ExpertOutput:
        text = subtask.text.lower()
        for key, definition in self.DEFINITIONS.items():
            if key in text:
                return ExpertOutput(
                    expert_name=self.profile.name,
                    subtask_id=subtask.id,
                    text=f"Definition: {definition}",
                    confidence=0.86,
                )

        return ExpertOutput(
            expert_name=self.profile.name,
            subtask_id=subtask.id,
            text="Definition-style answer: the request asks for explanation, but the exact term is not in this prototype's dictionary.",
            confidence=0.48,
        )


class GeneralExpert(BaseExpert):
    def __init__(self) -> None:
        super().__init__(
            ExpertProfile(
                name="Expert_General",
                domain="general",
                keywords=["general", "help", "answer", "question", "idea", "plan"],
                quality_score=0.64,
                cost=0.04,
                latency=0.04,
            )
        )

    def run(self, subtask: SubTask, context: Optional[Dict[str, Any]] = None) -> ExpertOutput:
        return ExpertOutput(
            expert_name=self.profile.name,
            subtask_id=subtask.id,
            text=(
                "General response: I can help structure this request. "
                "A strong answer should identify the task, choose the right expert knowledge, produce a clear response, and verify it."
            ),
            confidence=0.58,
        )


# -----------------------------
# Safe math evaluation
# -----------------------------

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def extract_math_expression(text: str) -> Optional[str]:
    normalized = (
        text.lower()
        .replace("plus", "+")
        .replace("minus", "-")
        .replace("multiplied by", "*")
        .replace("times", "*")
        .replace("x", "*")
        .replace("divided by", "/")
        .replace("over", "/")
        .replace("^", "**")
    )
    candidates = re.findall(r"[-+*/().\d\s*]+", normalized)
    candidates = [c.strip() for c in candidates if any(ch.isdigit() for ch in c)]
    if not candidates:
        return None
    expression = max(candidates, key=len)
    expression = re.sub(r"\s+", "", expression)
    if len(expression) < 1:
        return None
    return expression


def safe_eval(expr: str) -> float:
    tree = ast.parse(expr, mode="eval")
    return _eval_ast(tree.body)


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        return _ALLOWED_OPERATORS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        operand = _eval_ast(node.operand)
        return _ALLOWED_OPERATORS[type(node.op)](operand)

    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


# -----------------------------
# Router
# -----------------------------

class Router:
    """
    Scores experts for each subtask.

    Score formula:
        relevance + domain match + expert quality - cost penalty - latency penalty + memory bonus
    """

    def __init__(
        self,
        experts: List[BaseExpert],
        cost_penalty: float = 0.15,
        latency_penalty: float = 0.10,
        quality_reward: float = 0.80,
    ) -> None:
        self.experts = experts
        self.cost_penalty = cost_penalty
        self.latency_penalty = latency_penalty
        self.quality_reward = quality_reward

    def route(
        self,
        subtask: SubTask,
        rep: TaskRepresentation,
        top_k: int = 2,
    ) -> List[RoutedExpert]:
        scores: List[float] = []

        for expert in self.experts:
            profile = expert.profile
            if not profile.available:
                scores.append(-1e9)
                continue

            relevance = expert.keyword_relevance(subtask)
            domain_match = 1.0 if subtask.domain_hint == profile.domain else 0.0

            memory_bonus = self._memory_bonus(profile.name, rep.memory_hits)

            raw_score = (
                2.0 * domain_match
                + 1.5 * relevance
                + self.quality_reward * profile.quality_score
                - self.cost_penalty * profile.cost
                - self.latency_penalty * profile.latency
                + memory_bonus
            )
            scores.append(raw_score)

        probabilities = softmax(scores)

        routed = [
            RoutedExpert(expert=e, probability=p, raw_score=s)
            for e, p, s in zip(self.experts, probabilities, scores)
        ]
        routed.sort(key=lambda r: r.probability, reverse=True)
        return routed[: max(1, min(top_k, len(routed)))]

    def _memory_bonus(self, expert_name: str, memory_hits: List[Dict[str, Any]]) -> float:
        # If similar previous queries used this expert and were accepted, give a small boost.
        bonus = 0.0
        for item in memory_hits:
            if expert_name in item.get("selected_experts", []):
                if item.get("accepted", False):
                    bonus += 0.05
        return min(bonus, 0.15)


# -----------------------------
# Aggregator
# -----------------------------

class Aggregator:
    def __init__(
        self,
        eta_router: float = 1.0,
        eta_confidence: float = 1.2,
        eta_quality: float = 0.8,
        eta_cost: float = 0.2,
        eta_latency: float = 0.2,
    ) -> None:
        self.eta_router = eta_router
        self.eta_confidence = eta_confidence
        self.eta_quality = eta_quality
        self.eta_cost = eta_cost
        self.eta_latency = eta_latency

    def aggregate_subtask(
        self,
        subtask: SubTask,
        routed: List[RoutedExpert],
        outputs: List[ExpertOutput],
    ) -> Tuple[str, float]:
        if not outputs:
            return "No expert output was produced.", 0.0

        output_by_name = {o.expert_name: o for o in outputs}
        scores = []

        for r in routed:
            out = output_by_name.get(r.expert.profile.name)
            if not out:
                scores.append(-1e9)
                continue

            p = r.probability
            u = out.confidence
            q = r.expert.profile.quality_score
            c = r.expert.profile.cost
            l = r.expert.profile.latency

            score = (
                self.eta_router * p
                + self.eta_confidence * u
                + self.eta_quality * q
                - self.eta_cost * c
                - self.eta_latency * l
            )
            scores.append(score)

        weights = softmax(scores)
        weighted_outputs = list(zip(weights, outputs))
        weighted_outputs.sort(key=lambda item: item[0], reverse=True)

        primary_weight, primary_output = weighted_outputs[0]

        supporting = []
        for weight, output in weighted_outputs[1:]:
            if weight > 0.20 and output.confidence > 0.50:
                supporting.append(
                    f"- Supporting note from {output.expert_name} "
                    f"(weight={weight:.2f}, confidence={output.confidence:.2f}): {output.text}"
                )

        result = (
            f"Subtask `{subtask.domain_hint}` handled mainly by {primary_output.expert_name} "
            f"(weight={primary_weight:.2f}, confidence={primary_output.confidence:.2f}).\n"
            f"{primary_output.text}"
        )

        if supporting:
            result += "\n" + "\n".join(supporting)

        reliability = sum(w * o.confidence for w, o in weighted_outputs)
        return result, clamp(reliability)

    def aggregate_global(self, query: str, subtask_results: List[SubTaskResult]) -> str:
        if not subtask_results:
            return "DENA could not create a result."

        if len(subtask_results) == 1:
            body = subtask_results[0].aggregated_text
        else:
            sections = []
            for idx, result in enumerate(subtask_results, start=1):
                sections.append(f"Step {idx}: {result.aggregated_text}")
            body = "\n\n".join(sections)

        avg_reliability = sum(r.reliability for r in subtask_results) / len(subtask_results)
        return (
            "DENA Prototype Output\n"
            "---------------------\n"
            f"Query: {query}\n\n"
            f"{body}\n\n"
            f"Estimated reliability: {avg_reliability:.2f}"
        )


# -----------------------------
# Verifier
# -----------------------------

class Verifier:
    def __init__(self, threshold: float = 0.45) -> None:
        self.threshold = threshold

    def verify(self, final_answer: str, subtask_results: List[SubTaskResult]) -> Tuple[float, bool]:
        if not subtask_results:
            return 0.0, False

        avg_reliability = sum(r.reliability for r in subtask_results) / len(subtask_results)

        penalty = 0.0
        risky_phrases = [
            "could not safely evaluate",
            "no expert output",
            "not in this prototype",
            "no clear",
        ]
        lowered = final_answer.lower()
        for phrase in risky_phrases:
            if phrase in lowered:
                penalty += 0.08

        score = clamp(avg_reliability - penalty)
        return score, score >= self.threshold


# -----------------------------
# DENA engine
# -----------------------------

class DENAEngine:
    def __init__(
        self,
        experts: Optional[List[BaseExpert]] = None,
        memory_path: str = "dena_memory.jsonl",
        top_k: int = 2,
    ) -> None:
        self.memory = MemoryStore(memory_path)
        self.preprocessor = Preprocessor(self.memory)
        self.orchestrator = Orchestrator()
        self.experts = experts or [
            MathExpert(),
            CodeExpert(),
            ResearchExpert(),
            DefinitionExpert(),
            GeneralExpert(),
        ]
        self.router = Router(self.experts)
        self.aggregator = Aggregator()
        self.verifier = Verifier()
        self.top_k = top_k

    def process(self, query: str) -> DENAResult:
        rep = self.preprocessor.transform(query)
        subtasks = self.orchestrator.decompose(rep)

        subtask_results: List[SubTaskResult] = []

        for subtask in subtasks:
            selected = self.router.route(subtask, rep, top_k=self.top_k)

            outputs: List[ExpertOutput] = []
            for routed in selected:
                output = routed.expert.run(
                    subtask,
                    context={
                        "memory_hits": rep.memory_hits,
                        "tokens": rep.tokens,
                    },
                )
                outputs.append(output)

            aggregated_text, reliability = self.aggregator.aggregate_subtask(
                subtask=subtask,
                routed=selected,
                outputs=outputs,
            )

            subtask_results.append(
                SubTaskResult(
                    subtask=subtask,
                    selected_experts=selected,
                    expert_outputs=outputs,
                    aggregated_text=aggregated_text,
                    reliability=reliability,
                )
            )

        final_answer = self.aggregator.aggregate_global(query, subtask_results)
        verification_score, accepted = self.verifier.verify(final_answer, subtask_results)

        selected_expert_names = []
        for result in subtask_results:
            selected_expert_names.extend([r.expert.profile.name for r in result.selected_experts])

        memory_id = self.memory.append(
            {
                "timestamp": time.time(),
                "query": query,
                "tokens": rep.tokens,
                "subtasks": [
                    {"id": s.id, "text": s.text, "domain_hint": s.domain_hint}
                    for s in subtasks
                ],
                "selected_experts": selected_expert_names,
                "final_answer": final_answer,
                "verification_score": verification_score,
                "accepted": accepted,
            }
        )

        return DENAResult(
            query=query,
            subtasks=subtasks,
            subtask_results=subtask_results,
            final_answer=final_answer,
            verification_score=verification_score,
            accepted=accepted,
            memory_id=memory_id,
        )


# -----------------------------
# Feedback helper
# -----------------------------

def save_feedback(memory_path: str, memory_id: str, rating: float, comment: str = "") -> None:
    """
    Simple feedback file.

    In a real DENA system, feedback would update:
    - expert quality scores
    - router labels
    - aggregator preferences
    - verifier calibration
    """
    feedback_path = Path(memory_path).with_name("dena_feedback.jsonl")
    record = {
        "timestamp": time.time(),
        "memory_id": memory_id,
        "rating": clamp(rating),
        "comment": comment,
    }
    with feedback_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# -----------------------------
# CLI
# -----------------------------

def run_query(query: str, memory_path: str, top_k: int, feedback: bool = False) -> None:
    engine = DENAEngine(memory_path=memory_path, top_k=top_k)
    result = engine.process(query)

    print(result.final_answer)
    print()
    print(f"Verification score: {result.verification_score:.2f}")
    print(f"Accepted: {result.accepted}")
    print(f"Memory ID: {result.memory_id}")

    print("\nRouting trace:")
    for sub_result in result.subtask_results:
        print(f"- {sub_result.subtask.id} [{sub_result.subtask.domain_hint}]")
        for routed in sub_result.selected_experts:
            print(
                f"  {routed.expert.profile.name}: "
                f"probability={routed.probability:.2f}, raw_score={routed.raw_score:.2f}"
            )

    if feedback:
        try:
            value = float(input("\nRate this answer from 0 to 1: ").strip())
            comment = input("Optional comment: ").strip()
            save_feedback(memory_path, result.memory_id, value, comment)
            print("Feedback saved.")
        except Exception as exc:
            print(f"Feedback was not saved: {exc}")


def run_demo(memory_path: str, top_k: int) -> None:
    examples = [
        "What is DENA and how is it different from MoE?",
        "Calculate 24 * 7 + 13",
        "How should I debug a Python function error?",
        "Design an experiment to compare DENA with dense and MoE models.",
    ]

    for idx, query in enumerate(examples, start=1):
        print("=" * 80)
        print(f"Demo {idx}: {query}")
        print("=" * 80)
        run_query(query, memory_path=memory_path, top_k=top_k, feedback=False)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="DENA v0 Prototype")
    parser.add_argument("--query", type=str, help="Query to process with DENA")
    parser.add_argument("--demo", action="store_true", help="Run demo queries")
    parser.add_argument("--interactive", action="store_true", help="Start interactive mode")
    parser.add_argument("--memory", type=str, default="dena_memory.jsonl", help="Memory JSONL path")
    parser.add_argument("--top-k", type=int, default=2, help="Number of experts selected per subtask")
    parser.add_argument("--feedback", action="store_true", help="Ask for feedback after response")
    args = parser.parse_args()

    if args.demo:
        run_demo(memory_path=args.memory, top_k=args.top_k)
        return

    if args.interactive:
        print("DENA interactive mode. Type 'exit' to stop.")
        while True:
            query = input("\nQuery> ").strip()
            if query.lower() in {"exit", "quit"}:
                break
            if query:
                run_query(query, memory_path=args.memory, top_k=args.top_k, feedback=args.feedback)
        return

    if args.query:
        run_query(args.query, memory_path=args.memory, top_k=args.top_k, feedback=args.feedback)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
