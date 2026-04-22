from __future__ import annotations
import textwrap

from mesa import Model
from mesa.discrete_space import Cell

from .base import IPDAgent
from .factory import register_agent
from llm.ollama_backend import OllamaBackend


# JSON schemas used to enforce LLM compliance to outputs of certain shapes.
MOVE_PLUS_RATIONALE_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "move": {"enum": ["C", "D"]},
        "reason": {"type": "string"},
    },
    "required": ["move", "reason"],
    "additionalProperties": False,
}

MOVE_ONLY_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "move": {"enum": ["C", "D"]},
    },
    "required": ["move"],
    "additionalProperties": False,
}

REWIRING_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "nodes_to_drop": {
            "type": "array",
            "items": {"type": "string"},
        },
        "nodes_to_add": {
            "type": "array",
            "items": {"type": "string"},
        },
        "reason": {"type": "string"},
    },
    "required": ["nodes_to_drop", "nodes_to_add"],
    "additionalProperties": False,
}


@register_agent("LLMAgent")
class LLMAgent(IPDAgent):
    """
    Generic LLM-driven IPD agent.

    This is intended to be a student's starting point:
    - generic prompt structure
    - generic rewiring workflow

    Subclasses may override:
    - system_prompt()
    - build_decision_prompt()
    - build_rewiring_prompt()
    """

    def __init__(
        self,
        model: Model,
        cell: Cell,
        backend: OllamaBackend | None = None,
    ):
        super().__init__(model, cell)
        self._pending_rewiring_plan = None

        if backend is not None:
            self.backend = backend
        else:
            self.backend = OllamaBackend(self.model.ollama_model)


    """
    USEFUL METHODS YOU CAN CALL
    """
    def serialize_payoffs(self):
        """
        Return a string that encodes the payoff matrix for the LLM in a fairly
        compact way.
        """
        pm = self.model.payoff_matrix
        return textwrap.dedent(f"""\
            Payoffs per round:
            CC -> you {pm[('C','C')][0]}, other {pm[('C','C')][1]}
            CD -> you {pm[('C','D')][0]}, other {pm[('C','D')][1]}
            DC -> you {pm[('D','C')][0]}, other {pm[('D','C')][1]}
            DD -> you {pm[('D','D')][0]}, other {pm[('D','D')][1]}
        """).strip()

    def serialize_history(self, other, history=None, player="me"):
        """
        Return a string that encodes the play history of you against a specific
        opponent, or another player against a specific opponent.

        other: the node number of the opponent in the history.
        history: if None (or not specified), the history will be automatically
            drawn from this node's (self's) game history, as dutifully
            maintained by the simulator in the self.history object. If not
            None, then it should be a history object of the same shape (i.e.,
            a dict whose keys are node numbers, and whose values are lists of
            dicts, each of which has a 'step', 'self_move', and 'other_move'
            key/value pair.
        player: if "me," then the text is stylized to express the agent's
            personal history against the opponent. If anything else, it's
            stylized the express some *other* agent's history against the 
            opponent. (In future, it might make sense to encode *which* other
            agent was playing against this opponent, for use in advanced
            strategies.)
        """
        if not history:
            history = self.history
        possessive = 'your' if player=="me" else f"other node's"
        subject = 'you' if player=="me" else 'other node'
        hist = history.get(other, [])
        if not hist:
            return f"(This is {possessive} first move against {other}.)"

        lines = [f"{possessive} history vs {other}:"]
        lines.extend(
            f"round {h['step']} -- {subject}: {h['self_move']}, "
            f"node {other}: {h['other_move']}"
            for h in hist
        )
        return "\n".join(lines)


    """
    *** STOP READING HERE ***
    CLASS INNARDS THAT YOU NEEDN'T WORRY YOUR PRETTY LITTLE HEAD ABOUT
    """

    def system_prompt(self) -> str | None:
        raise NotImplementedError

    def build_decision_prompt(
        self,
        other: "IPDAgent",
    ) -> str:
        raise NotImplementedError

    def build_rewiring_prompt(
        self,
        starting_neighbors: set[int],
        new_neighbor_candidates: set[int],
        max_rewires: int,
        give_rationale: bool,
    ) -> str:
        raise NotImplementedError

    def decide_against(
        self,
        other: int,
        give_rationale: bool,
    ) -> tuple[str, str]:
        prompt = self.build_decision_prompt(other)

        if give_rationale:
            num_predict = 256
            schema = MOVE_PLUS_RATIONALE_DECISION_SCHEMA
        else:
            num_predict = 16
            schema = MOVE_ONLY_DECISION_SCHEMA

        resp = self.backend.generate_json(
            prompt=prompt,
            num_predict=num_predict,
            schema=schema,
            system=self.system_prompt(),
        )

        decision = resp["move"].strip().upper()
        rationale = resp.get("reason", "(none)")

        with open(self.model.llm_out_file, "a", encoding="utf-8") as f:
            print(
                "---------------------------------------------------\n"
                + self.serialize_history(other), file=f,
            )
            print(f"DECISION: {decision}. RATIONALE: {rationale}", file=f)

        return decision, rationale

    def request_rewire(
        self,
        max_rewires: int,
    ) -> dict[str, list[int]]:
        prompt = self.build_rewiring_prompt(max_rewires)

        num_predict = 100
        schema = REWIRING_DECISION_SCHEMA

        resp = self.backend.generate_json(
            prompt=prompt,
            num_predict=num_predict,
            schema=schema,
            system=self.system_prompt(),
        )
        return resp

    def shape(self) -> str:
        return "h"  # Hex

    def size(self) -> int:
        return 2000

