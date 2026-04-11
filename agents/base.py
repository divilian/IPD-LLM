from collections.abc import Callable
from typing import TypeVar
from collections import defaultdict

from mesa import Model
from mesa.discrete_space import CellAgent, Cell


T = TypeVar("T")

AGENT_REGISTRY: dict[str, type] = {}


def register_agent(name: str) -> Callable[[T], T]:
    """
    Decorator for new Agent IPDsubclasses.
    """
    def decorator(cls: T) -> T:
        AGENT_REGISTRY[name] = cls
        return cls
    return decorator


class IPDAgent(CellAgent):
    """
    Base class for all IPD agents. Note: there is no ".step()" method, here or
    in subclasses, as is traditional in Mesa. We instead require
    ".decide_against()" for each subclass.
    """

    def __init__(self, model: Model, cell: Cell):
        super().__init__(model)
        self.cell = cell
        self.current_iter_payment = 0

        # history[other_node] = list of {step, self_action, other_action}
        self.history = defaultdict(list)

        # current_decisions[other_node] = action for THIS step only
        self.current_decisions = {}

        self.wealth = 0.0

    def record_interaction(
        self,
        other_node: int,
        self_action: str,
        other_action: str,
    ) -> None:
        self.history[other_node].append(
            {
                "step": self.model.steps,
                "self_action": self_action,
                "other_action": other_action,
            }
        )

    def decide_against(
        self,
        other: "IPDAgent",
        payoff_matrix: dict[tuple[str, str], tuple[str, str]],
        give_rationale: bool,
    ) -> tuple[str, str]:
        """
        Make a decision against another agent. Return your decision ("C" or
        "D") and, if give_rationale is True, a description of the interaction
        (for logging). (If give_rationale is False, agents must still return an
        empty string as the second element of the returned tuple, for
        simplicity.)
        """
        raise NotImplementedError

    def rewire_as_desired(
        self,
        payoff_matrix: dict[tuple[str, str], tuple[str, str]]):
        """
        Optionally do either or both of the following:
        (1) Sever graph connections to agents you no longer want to play with.
        (2) Create graph connections to other agents, which you can discover by
        asking your neighbors for information via `.inform_foaf()`.
        """
        pass

    def inform_foaf(
        self,
        inquirer: "IPDAgent",
    ) -> dict[int, list[dict[str, int | str]]] | None:
        """
        Return a reported interaction history keyed by opponent node number.
        Each value is a list of dicts with keys 'step' (the round number),
        'self_action', and 'other_action'.

        Note that it is perfectly permissible to lie about this history, if you
        deem that advantageous. You can also return None to give the inquirer
        the hand.
        """
        return self.history

    def shape(self) -> str:
        """
        Return the shape your node should be in the graph. See:
        https://matplotlib.org/stable/api/markers_api.html.
        """
        raise 's'

    def size(self) -> str:
        """
        Return the size your node should be in the graph.
        """
        return 300

    @property
    def node(self) -> int:
        return self.cell.coordinate

    def __str__(self) -> str:
        return (
            f"Node {self.node} (agent id {self.unique_id}) "
            f"{self.__class__.__name__} "
            f"with ${int(self.wealth)}"
        )
