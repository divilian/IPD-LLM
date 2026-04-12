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
        give_rationale: bool,
    ) -> tuple[str, str]:
        if give_rationale:
            log = (
                f"I'm node {self.node} (Sucker), interacting with "
                "{other.node}. (C'ing as always.)"
            )
        else:
            log = ""
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
        give_rationale: bool,
    ) -> tuple[str, str]:
        move = self.model.random.choice(['C','D'])
        if give_rationale:
            log = (
                f"I'm node {self.node} (Random), interacting with "
                "{other.node}. Choosing to {move} this time."
            )
        else:
            log = ""
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
        give_rationale: bool,
    ) -> tuple[str, str]:
        if give_rationale:
            log = (
                f"I'm node {self.node} (Mean), interacting with "
                "{other.node}. (D'ing as always.)"
            )
        else:
            log = ""
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
        f"\n  Last time node {other_node} {h[-1]['other_move']}'d "
        + f"against me. So I'm {h[-1]['other_move']}'ing them this time."
    )
    return h[-1]["other_move"], log

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
        give_rationale: bool,
    ) -> tuple[str, str]:
        prelude = f"I'm node {self.node} (TFT), interacting with {other.node}. "
        h = self.history[other.node]
        output = tft_algorithm(
            h,
            other.node,
            prelude,
            self.random,
            self.noise,
        )
        if give_rationale:
            return output
        else:
            return output[0], ""

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
        give_rationale: bool,
    ) -> tuple[str, str]:
        prelude = (
            f"I'm node {self.node} (Browser), interacting with {other.node}. "
        )
        h = self.history[other.node]
        output = tft_algorithm(
            h,
            other.node,
            prelude,
            self.random,
            self.tft_noise,
        )
        if give_rationale:
            return output
        else:
            return output[0], ""
        
    def choose_neighbors_to_sever(
        self,
        payoff_matrix: dict[tuple[str, str], tuple[str, str]],
        starting_neighbors: set[int],
        my_foafs: set[int],
    ) -> list[int]:
        """
        BrowserAgent policy: anyone who's defected a lot recently gets the axe.
        """
        sever_targets: list[int] = []

        for node, history in self.history.items():
            if (
                node in starting_neighbors
                and self._has_defected_too_often(history)
            ):
                sever_targets.append(node)

        return sever_targets

    def _has_defected_too_often(self, other_history):
        """
        Return True iff the other agent has defected in each of the most recent
        `self.patience` interactions. If there are fewer than `self.patience`
        interactions in the history, return False.
        """
        if len(other_history) < self.patience:
            if self.model.debug:
                print(f"Not enough history to know whether to ditch this friend.")
            return False

        recent = other_history[-self.patience :]
        answer = all(move["other_move"] == "D" for move in recent)
        if answer:
            if self.model.debug:
                print(f"Ditch this guy!")
        else:
            if self.model.debug:
                print(f"Stay friends with this guy.")
        return all(move["other_move"] == "D" for move in recent)

    def choose_new_neighbors(
        self,
        payoff_matrix: dict[tuple[str, str], tuple[str, str]],
        eligible_foafs: set[int],
        num_needed_replacements: int,
        severed_nodes: set[int],
        starting_neighbors: set[int],
    ) -> list[int]:
        """
        BrowserAgent policy: pick random eligible FOAFs as replacements.
        """
        if not eligible_foafs or num_needed_replacements <= 0:
            return []

        candidates = list(eligible_foafs)
        self.model.random.shuffle(candidates)
        return candidates[:num_needed_replacements]

    def shape(self) -> str:
        return "<"   # Triangle sideways = "shopping around"
