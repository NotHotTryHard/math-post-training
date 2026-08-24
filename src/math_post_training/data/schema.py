"""Canonical representation of a mathematical training example."""

from dataclasses import dataclass


@dataclass
class MathExample:
    problem: str
    solution: str | None
    answer: str | None
    source: str
    difficulty: float | None = None
    topic: str | None = None
