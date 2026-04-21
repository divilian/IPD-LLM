from typing import Tuple

from mesa import Model
from mesa.discrete_space import CellAgent, Cell

from .base import IPDAgent
from .factory import register_agent

@register_agent("Sucker")
class SuckerAgent(IPDAgent):
    """Always cooperates."""

    def __init__(self, model: Model, cell: Cell):
        super().__init__(model, cell)

    def decide_against(
        self,
        other: int,
        give_rationale: bool,
    ) -> tuple[str, str]:
        if give_rationale:
            log = (
                f"I'm node {self.node} (Sucker), interacting with "
                "{other}. (C'ing as always.)"
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
        other: int,
        give_rationale: bool,
    ) -> tuple[str, str]:
        move = self.model.random.choice(['C','D'])
        if give_rationale:
            log = (
                f"I'm node {self.node} (Random), interacting with "
                "{other}. Choosing to {move} this time."
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
        other: int,
        give_rationale: bool,
    ) -> tuple[str, str]:
        if give_rationale:
            log = (
                f"I'm node {self.node} (Mean), interacting with "
                "{other}. (D'ing as always.)"
            )
        else:
            log = ""
        return "D", log

    def shape(self) -> str:
        return "v"   # Down triangle = "mean/aggressive"


@register_agent("TFT")
class TitForTatAgent(IPDAgent):
    """Classic per-neighbor tit-for-tat (with optional noise)."""

    def __init__(self, model: Model, cell: Cell, noise: float = 0.10):
        super().__init__(model, cell)
        self.noise = noise

    def decide_against(
        self,
        other: int,
        give_rationale: bool,
    ) -> tuple[str, str]:
        if other not in self.history:
            choice = self.model.random.choice(["C", "D"])
            return choice, f"It's my first time! ({choice})."

        if random.random() < noise:
            choice = self.model.random.choice(["C", "D"])
            return choice, f"I'm going random ({choice})."

        their_last_move = self.history[other][-1]['other_move']

        if give_rationale:
            agent_type = self.__class__.__name__.replace("Agent", "")
            return (
                their_last_move,
                (
                    f"I'm node {self.node} ({agent_type}), "
                    f"interacting with {other}.\n"
                    f"Last time node {other} {their_last_move}'d against "
                    f"me. So I'm {their_last_move}'ing against them now."
                ),
            )
        else:
            return their_last_move, ""

    def shape(self) -> str:
        return "s"   # Square = "rule-based/fair/predictable"


@register_agent("Browser")
class BrowserAgent(TitForTatAgent):
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
        super().__init__(model, cell, tft_noise)
        self.patience = patience
        self.known_non_neighbors = []

    def request_rewire(
        self,
        max_rewires: int,
    ) -> dict[str, list[int]]:
        """
        BrowserAgent policy: anyone who's defected a lot recently gets the axe.
        Pick random eligible FOAFs as replacements.
        """
        sever_targets = []

        for node, history in self.history.items():
            if (
                node in self.model.network.G.neighbors(self.node)
                and self._has_defected_too_often(history)
                and len(sever_targets) < max_rewires
            ):
                sever_targets.append(node)

        new_targets = self._select_up_to_n_new_targets(len(sever_targets))
        sever_targets = sever_targets[:len(new_targets)]

        return { 'nodes_to_sever': sever_targets, 'nodes_to_add': new_targets }

    def _select_up_to_n_new_targets(self, n: int):

        neighbors_left_to_ask = self.neighbors()
        new_targets = []
        # First, let's add nodes that I know about but which are known not to
        # be my neighbors. That's free.
        while (
            len(new_targets) < n and
            len(self.known_non_neighbors) >= 1
        ):
            new_targets.append(self.known_non_neighbors.pop(
                self.random.randrange(len(self.known_non_neighbors))
            ))

        # Now if that resulted in enough additions, great; yeet 'em outta here.
        if len(new_targets) == n:
            return new_targets

        # Otherwise, I guess we ought to ask around.
        neighbors_left_to_ask = self.neighbors()
        while (
            len(new_targets) < n and
            len(neighbors_left_to_ask) >= 1
        ):
            possible_targets = \
                [ k for k in self.model.request_foaf_info_from(self,
                    neighbors_left_to_ask.pop(
                        self.random.randrange(len(neighbors_left_to_ask))
                    )
                )
            ]
            new_targets += possible_targets[:n-len(new_targets)]

        # Whether or not we actually have enough, that's all we got.
        return new_targets

    def _has_defected_too_often(self, other_history):
        """
        Return True iff the other agent has defected in each of the most recent
        `self.patience` interactions. If there are fewer than `self.patience`
        interactions in the history, return False.
        """
        if len(other_history) < self.patience:
            return False

        recent = other_history[-self.patience :]
        return all(move["other_move"] == "D" for move in recent)

    def shape(self) -> str:
        return "<"   # Triangle sideways = "shopping around"
