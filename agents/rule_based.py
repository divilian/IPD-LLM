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
        
    def is_adjacent_to_node(self, x):
        return any(
            (cell.coordinate == x for cell in self.cell.neighborhood.cells)
        )

    def rewire_as_desired(
        self,
        payoff_matrix: dict[tuple[str, str], tuple[str, str]]):
        """
        Anyone who has defected a lot recently gets the axe.
        """
        if self.model.debug:
            print(f"***************************************\n"
                f"{self.unique_id} (node {self.node}) is considering rewiring")

        my_foafs = self._get_foaf_nodes()

        # Keep track of who my neighbors are at the start of rewiring.
        # We will exclude all of them from replacement candidates, so that if I
        # sever a connection with someone during this rewiring step, I do not
        # immediately reconnect to that same node.
        starting_neighbors = self._get_current_neighbors()

        # Keep track of how many connections I've severed, so that I know how
        # many new connections to make when I'm done severing. (So as to
        # preserve this node's degree to the extent possible.)
        self.num_needed_replacements = 0
        severed_nodes = set()

        for node, history in self.history.items():
            if (
                # Even if we've previously severed connections with this agent,
                # we still have its history from before our breakup. So we need
                # to check whether the node we think we want to ditch is
                # actually still a neighbor!
                self.is_adjacent_to_node(node) and
                self._has_defected_too_often(history)
            ):
                if self.model.debug:
                    print(f"You're dead to me, {node}.")
                # This should work vvvvv but does not. See discussion #3694.
                #self.cell.disconnect(self.model.node_to_agent[node].cell)
                self.model.network.remove_connection(
                    self.cell,
                    self.model.node_to_agent[node].cell,
                )
                self.num_needed_replacements += 1
                severed_nodes.add(node)

        if self.num_needed_replacements:
            if self.model.debug:
                print(f"I've terminated {self.num_needed_replacements} "
                    "connections, and will now try to make that many more.")
            pass
        else:
            if self.model.debug:
                print("Nobody terminated; no need to make more connections.")
            pass

        eligible_foafs = self._get_eligible_rewiring_candidates(
            my_foafs,
            starting_neighbors,
            severed_nodes,
        )

        for _ in range(self.num_needed_replacements):
            if eligible_foafs:
                # In this model, people don't have to approve friend requests.
                new_friend_node = self.model.random.choice(list(eligible_foafs))
                # This should work vvvvv but does not. See discussion #3694.
                #self.cell.connect(self.model.node_to_agent[new_friend_node])
                self.model.network.add_connection(
                    self.cell,
                    self.model.node_to_agent[new_friend_node].cell,
                )
                my_foafs -= {new_friend_node}
                if self.model.debug:
                    print(f"I'm now friends with {new_friend_node}!")
            else:
                if self.model.debug:
                    print(f"Yikes! Nobody to replace severed connection with.")
                return

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

    def shape(self) -> str:
        return "<"   # Triangle sideways = "shopping around"
