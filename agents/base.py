from collections.abc import Callable
from typing import TypeVar
from collections import defaultdict

from mesa import Agent, Model


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


class IPDAgent(Agent):

    def __init__(self, model: Model, node: int):
        super().__init__(model)
        self.node = node
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
    ) -> tuple[str, str]:
        """
        Make a decision against another agent. Return your decision ("C" or
        "D") and a description of the interaction (for logging).
        """
        raise NotImplementedError

    def shape(self) -> str:
        """
        Return the shape your node should be in the graph. See:
        https://matplotlib.org/stable/api/markers_api.html.
        """
        raise NotImplementedError

    def step(self) -> None:
        """
        Decide your actions for all neighbors (per-neighbor decision making).
        (Actual payoff resolution is done in the Model.step().)

        Note: this method is never called. It's now accomplished in the
        backend, in batch for many agents.
        """
        self.current_decisions.clear()
        for nbr in self.model.graph.neighbors(self.node):
            other = self.model.node_to_agent[nbr]
            self.current_decisions[nbr], desc = self.decide_against(
                other,
                self.model.payoff_matrix
            )
            logging.info(desc)

    def decision_context(self) -> dict:
        """
        Return a dict to send to the LLM representing this agent. It should
        include only things relevant to it making its decision.
        Note: this default behavior is intended to be overridden by subclasses
        who need (say) history information.
        """
        return {
            "id": self.node,   # use node, not unique_id
            "persona": self.persona,
        }

    def __str__(self) -> str:
        return (
            f"Node {self.node} (agent id {self.unique_id}) "
            f"{self.__class__.__name__} "
            f"with ${int(self.wealth)}"
        )
