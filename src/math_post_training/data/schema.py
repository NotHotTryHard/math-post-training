"""Canonical representation of a mathematical training example."""

from dataclasses import dataclass


@dataclass
class MathExample:
    problem: str
    solution: str | None
    answer: str | None
    source: str
