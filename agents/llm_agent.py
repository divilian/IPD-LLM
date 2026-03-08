from typing import List, Tuple

from mesa import Model, Agent

from .base import IPDAgent
from .personas import PERSONAS, HistoryType

class LLMAgent(IPDAgent):
    """LLM-driven per-neighbor decisions."""

    def __init__(
        self,
        model: Model,
        node: int,
        persona: str,
    ):
        super().__init__(model, node)
        if persona not in PERSONAS:
            persona_names = ", ".join(PERSONAS.keys())
            raise ValueError(f"{persona} not one of {persona_names}.")
        self.persona = persona

    def decide_against(
        self,
        other: "IPDAgent",
        payoff_matrix: List[Tuple],
    ) -> tuple[str, str]:

        decision = self.decisions[other.node]
        log = f"I'm node {self.node} (LLM), interacting with {other.node}. "
        log += f"I'm {decision}'ing."
        return decision, log

    def decision_context(self) -> dict:
        persona = PERSONAS[self.persona]
        ctx = {
            "id": self.node,
            "persona": self.persona,
            "opponents": {},
        }

        for n in self.model.graph.neighbors(self.node):
            if persona.history == HistoryType.NONE:
                ctx["opponents"][n] = {}
            elif persona.history == HistoryType.LAST:
                ctx["opponents"][n] = {
                    "history": self.history[n][-1:]
                }
            elif persona.history == HistoryType.FULL:
                ctx["opponents"][n] = {
                    "history": self.history[n]
                }
        return ctx

    def shape(self) -> str:
        return "h"   # Hexagon = "tech/engineered/complex"
