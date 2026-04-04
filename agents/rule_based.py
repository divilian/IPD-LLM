from typing import Tuple

from mesa import Model
from mesa.discrete_space import CellAgent, Cell

from .base import IPDAgent, register_agent

@register_agent("Sucker")
class SuckerAgent(IPDAgent):
    """Always cooperates."""

    def __init__(self, model: Model, cell: Cell):
        super().__init__(model, cell)

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

    def __init__(self, model: Model, cell: Cell):
        super().__init__(model, cell)

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

    def __init__(self, model: Model, cell: Cell):
        super().__init__(model, cell)

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


def tft_algorithm(
    h: dict[int, list[dict[str, int | str]]] | None,
    other_node: int,
    prelude: str,
    random: "Random",
    noise: float=0.0,
) -> tuple[str, str]:
    """
    Perform the prototypical TitForTat algorithm against an opponent with the
    given history against them.
    """
    log = prelude
    if not h:
        choice = random.choice(["C", "D"])
        log += f"It's my first time! ({choice})."
        return choice, log

    if random.random() < noise:
        choice = random.choice(["C", "D"])
        log += f"I'm going random ({choice})."
        return choice, log

    log += (
        f"\n  Last time node {other_node} {h[-1]['other_action']}'d "
        + f"against me. So I'm {h[-1]['other_action']}'ing them this time."
    )
    return h[-1]["other_action"], log

@register_agent("TFT")
class TitForTatAgent(IPDAgent):
    """Classic per-neighbor tit-for-tat (with optional noise)."""

    def __init__(self, model: Model, cell: Cell, noise: float = 0.10):
        super().__init__(model, cell)
        self.noise = noise

    def decide_against(
        self,
        other: "IPDAgent",
        payoff_matrix: dict[tuple[str, str], tuple[str, str]],
    ) -> tuple[str, str]:
        prelude = f"I'm node {self.node} (TFT), interacting with {other.node}. "
        h = self.history[other.node]
        return tft_algorithm(h, other.node, prelude, self.random, self.noise)

    def shape(self) -> str:
        return "s"   # Square = "rule-based/fair/predictable"


@register_agent("Browser")
class BrowserAgent(IPDAgent):
    """
    Tit-for-Tats, and breaks contact with uncooperative opponents, replacing
    them with random FOAFs.
    """

    def __init__(
        self,
        model: Model,
        cell: Cell,
        tft_noise: float,
        patience: int,
    ):
        """
        tft_noise: passed on to the Tit-for-Tat algorithm.
        patience: the number of consecutive D's by the opponent that this
        agent will tolerate before severing the connection and going shopping.
        """
        super().__init__(model, cell)
        self.tft_noise = tft_noise
        self.patience = patience

    def decide_against(
        self,
        other: "IPDAgent",
        payoff_matrix: dict[tuple[str, str], tuple[str, str]],
    ) -> tuple[str, str]:
        prelude = (
            f"I'm node {self.node} (Browser), interacting with {other.node}. "
        )
        h = self.history[other.node]
        return tft_algorithm(
            h,
            other.node,
            prelude,
            self.random,
            self.tft_noise,
        )
        return h[-1]["other_action"], log
        
    def shape(self) -> str:
        return "<"   # Triangle sideways = "shopping around"
