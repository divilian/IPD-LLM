from typing import Tuple

from mesa import Model, Agent

from .base import IPDAgent, register_agent

@register_agent("Sucker")
class SuckerAgent(IPDAgent):
    """Always cooperates."""

    def __init__(self, model: Model, node: int):
        super().__init__(model, node)

    def decide_against(
        self,
        other: "IPDAgent",
        payoff_matrix: dict[tuple[str, str], tuple[str, str]],
    ) -> tuple[str, str]:
        log = f"I'm node {self.node} (Sucker), interacting with {other.node}. "
        log += "(C'ing as always.)"
        return "C", log

    def shape(self) -> str:
        return "o"   # Circle = "soft/friendly/harmless" vibe


@register_agent("Random")
class RandomAgent(IPDAgent):
    """Chooses randomly."""

    def __init__(self, model: Model, node: int):
        super().__init__(model, node)

    def decide_against(
        self,
        other: "IPDAgent",
        payoff_matrix: dict[tuple[str, str], tuple[str, str]],
    ) -> tuple[str, str]:
        move = self.model.random.choice(['C','D'])
        log = f"I'm node {self.node} (Random), interacting with {other.node}. "
        log += f"Choosing to {move} this time."
        return move, log

    def shape(self) -> str:
        return "d"   # Diamond


@register_agent("Mean")
class MeanAgent(IPDAgent):
    """Always defects."""

    def __init__(self, model: Model, node: int):
        super().__init__(model, node)

    def decide_against(
        self,
        other: "IPDAgent",
        payoff_matrix: dict[tuple[str, str], tuple[str, str]],
    ) -> tuple[str, str]:
        log = f"I'm node {self.node} (Mean), interacting with {other.node}. "
        log += "(D'ing as always.)"
        return "D", log

    def shape(self) -> str:
        return "v"   # Down triangle = "mean/aggressive"


@register_agent("TFT")
class TitForTatAgent(IPDAgent):
    """Classic per-neighbor tit-for-tat (with optional noise)."""

    def __init__(self, model: Model, node: int, noise: float = 0.10):
        super().__init__(model, node)
        self.noise = noise

    def decide_against(
        self,
        other: "IPDAgent",
        payoff_matrix: dict[tuple[str, str], tuple[str, str]],
    ) -> tuple[str, str]:
        log = f"I'm node {self.node} (TFT), interacting with {other.node}. "
        h = self.history[other.node]
        if not h:
            choice = self.model.random.choice(["C", "D"])
            log += f"It's my first time! ({choice})."
            return choice, log

        if self.model.random.random() < self.noise:
            choice = self.model.random.choice(["C", "D"])
            log += f"I'm going random ({choice})."
            return choice, log

        log += (
            f"\n  Last time node {other.node} {h[-1]['other_action']}'d "
            + f"against me. So I'm {h[-1]['other_action']}'ing them this time."
        )
        return h[-1]["other_action"], log

    def shape(self) -> str:
        return "s"   # Square = "rule-based/fair/predictable"

