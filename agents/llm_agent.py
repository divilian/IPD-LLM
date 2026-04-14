from __future__ import annotations

from mesa import Model
from mesa.discrete_space import Cell

from .base import IPDAgent
from .factory import register_agent
from llm.ollama_backend import OllamaBackend


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
        "drop_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "add_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "reason": {"type": "string"},
    },
    "required": ["drop_ids", "add_ids"],
    "additionalProperties": False,
}


@register_agent("LLMAgent")
class LLMAgent(IPDAgent):
    """
    Generic LLM-driven IPD agent.

    This is intended to be a student's starting point:
    - generic prompt structure
    - generic rewiring workflow
    - no especially opinionated strategy beyond "make a choice given the game state"

    Subclasses may override:
    - system_prompt()
    - build_decision_prompt()
    - build_rewiring_prompt()
    """

    def __init__(
        self,
        model: Model,
        cell: Cell,
        rewiring_aware: bool = False,
        backend: OllamaBackend | None = None,
    ):
        super().__init__(model, cell)
        self.rewiring_aware = rewiring_aware
        self._pending_rewiring_plan = None

        if backend is not None:
            self.backend = backend
        else:
            self.backend = OllamaBackend(self.model.ollama_model)

    def system_prompt(self) -> str | None:
        return (
            "You are an agent in an iterated prisoner's dilemma simulation. "
            "Read the game state carefully and return a valid answer that matches the requested format."
        )

    def build_decision_prompt(
        self,
        other: "IPDAgent",
        payoff_matrix: dict[tuple[str, str], tuple[str, str]],
    ) -> str:
        prompt = f"""You are playing an iterated prisoner's dilemma.

Payoffs to you:
CC -> {payoff_matrix['C', 'C'][0]}
DD -> {payoff_matrix['D', 'D'][0]}
DC -> {payoff_matrix['D', 'C'][0]}
CD -> {payoff_matrix['C', 'D'][0]}

History against this opponent:
{self.serialize_history(self.history, other.unique_id)}

Turns remaining including this one: {self.model.num_iter - self.model.steps + 1}

Choose your next move.
"""

        if self.rewiring_aware:
            prompt += (
                "\nAfter this round, you may have an opportunity to sever connections "
                "with current opponents and have them replaced with new opponents drawn "
                "from your friends-of-friends.\n"
            )

        return prompt

    def build_rewiring_prompt(
        self,
        starting_neighbors: set[int],
        new_neighbor_candidates: set[int],
        max_rewires: int,
        give_rationale: bool,
    ) -> str:
        current_neighbor_lines = []
        for node in sorted(starting_neighbors):
            hist = self.serialize_history(self.history, node + 1)
            current_neighbor_lines.append(f"- node={node}\n{hist}")

        current_neighbor_text = (
            "\n".join(current_neighbor_lines) if current_neighbor_lines else "(none)"
        )

        candidate_lines = []
        for node in sorted(new_neighbor_candidates):
            mutual_contacts = sorted(starting_neighbors & self._get_neighbors_of_node(node))
            candidate_lines.append(
                f"- node={node}, mutual_contacts={mutual_contacts}"
            )

        candidate_text = "\n".join(candidate_lines) if candidate_lines else "(none)"

        prompt = f"""You are deciding how to rewire your social network in a networked iterated prisoner's dilemma.

You may sever up to {max_rewires} current neighbors and add up to {max_rewires} new neighbors.

Current neighbors and your history against each:
{current_neighbor_text}

Available candidate new neighbors:
{candidate_text}
"""

        if give_rationale:
            prompt += "\nInclude a brief reason."
        else:
            prompt += "\nDo not include a reason."

        return prompt

    def decide_against(
        self,
        other: "IPDAgent",
        payoff_matrix: dict[tuple[str, str], tuple[str, str]],
        give_rationale: bool,
    ) -> tuple[str, str]:
        prompt = self.build_decision_prompt(other, payoff_matrix)

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
                + self.serialize_history(self.history, other.unique_id),
                file=f,
            )
            print(f"DECISION: {decision}. RATIONALE: {rationale}", file=f)

        return decision, rationale

    def shape(self) -> str:
        return "h"  # Hex

    def size(self) -> int:
        return 2000


    def _plan_rewiring_once(
        self,
        starting_neighbors: set[int],
        new_neighbor_candidates: set[int],
        max_rewires: int,
        give_rationale: bool = False,
    ) -> dict:
        prompt = self.build_rewiring_prompt(
            starting_neighbors=starting_neighbors,
            new_neighbor_candidates=new_neighbor_candidates,
            max_rewires=max_rewires,
            give_rationale=give_rationale,
        )

        resp = self.backend.generate_json(
            prompt=prompt,
            num_predict=192 if give_rationale else 96,
            schema=REWIRING_DECISION_SCHEMA,
            system=self.system_prompt(),
        )

        drop_ids = resp.get("drop_ids", [])
        add_ids = resp.get("add_ids", [])
        reason = resp.get("reason", "(none)")

        if not isinstance(drop_ids, list) or not all(isinstance(x, str) for x in drop_ids):
            raise ValueError(f"Invalid drop_ids returned by LLM: {drop_ids!r}")

        if not isinstance(add_ids, list) or not all(isinstance(x, str) for x in add_ids):
            raise ValueError(f"Invalid add_ids returned by LLM: {add_ids!r}")

        allowed_drop_ids = {str(node) for node in starting_neighbors}
        allowed_add_ids = {str(node) for node in new_neighbor_candidates}

        drop_ids = [x for x in drop_ids if x in allowed_drop_ids]
        add_ids = [x for x in add_ids if x in allowed_add_ids]

        drop_ids = list(dict.fromkeys(drop_ids))[:max_rewires]
        add_ids = list(dict.fromkeys(add_ids))[:max_rewires]

        plan = {
            "drop_ids": [int(x) for x in drop_ids],
            "add_ids": [int(x) for x in add_ids],
            "reason": reason,
        }

        with open(self.model.llm_out_file, "a", encoding="utf-8") as f:
            print("---------------------------------------------------", file=f)
            print("REWIRING DECISION", file=f)
            print(f"DROP: {plan['drop_ids']}", file=f)
            print(f"ADD: {plan['add_ids']}", file=f)
            print(f"RATIONALE: {reason}", file=f)

        return plan

    def choose_neighbors_to_sever(
        self,
        payoff_matrix: dict[tuple[str, str], tuple[str, str]],
        starting_neighbors: set[int],
        new_neighbor_candidates: set[int],
        max_rewires: int,
    ) -> list[int]:
        if not self.rewiring_aware:
            return []

        self._pending_rewiring_plan = self._plan_rewiring_once(
            starting_neighbors=starting_neighbors,
            new_neighbor_candidates=new_neighbor_candidates,
            max_rewires=max_rewires,
        )
        return self._pending_rewiring_plan["drop_ids"]

    def choose_new_neighbors(
        self,
        payoff_matrix: dict[tuple[str, str], tuple[str, str]],
        eligible_new_neighbors: set[int],
        num_needed_replacements: int,
        severed_nodes: set[int],
        starting_neighbors: set[int],
    ) -> list[int]:
        if not self.rewiring_aware:
            return []

        if self._pending_rewiring_plan is None:
            raise RuntimeError(
                "choose_new_neighbors() called without a pending rewiring "
                "plan. Expected choose_neighbors_to_sever() to run first in "
                "the same rewiring event."
            )

        try:
            add_ids = self._pending_rewiring_plan["add_ids"][:num_needed_replacements]
            allowed_add_ids = set(eligible_new_neighbors)
            return [node for node in add_ids if node in allowed_add_ids]
        finally:
            self._pending_rewiring_plan = None

    def _get_neighbors_of_node(self, node: int) -> set[int]:
        agent = self.model.node_to_agent[node]
        return {
            cell.coordinate
            for cell in agent.cell.neighborhood.cells
            if cell.coordinate != node and cell.agents
        }

    def serialize_history(self, history, oid):
        opponent_node = oid - 1
        if not history or not history[opponent_node]:
            return (
                f"This is the first move in your game with this opponent "
                f"({opponent_node}).\n"
            )

        prompt = (
            f"Here is the history of your games with this opponent "
            f"({opponent_node}) so far:\n"
        )

        for h in history[opponent_node]:
            prompt += (
                f"On turn {h['step']}, you chose {h['self_move']} and "
                f"your opponent chose {h['other_move']}.\n"
            )

        return prompt
