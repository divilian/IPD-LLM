import json
import requests

from mesa import Model
from mesa.discrete_space import Cell, CellAgent

from .base import IPDAgent, register_agent


MOVE_PLUS_RATIONALE_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "move": {
            "enum": ["C", "D"]
        },
        "reason": {
            "type": "string"
        },
    },
    "required": ["move", "reason"],
    "additionalProperties": False
}

MOVE_ONLY_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "move": {
            "enum": ["C", "D"]
        },
    },
    "required": ["move"],
    "additionalProperties": False
}

REWIRING_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "drop_ids": {
            "type": "array",
            "items": {"type": "string"}
        },
        "add_ids": {
            "type": "array",
            "items": {"type": "string"}
        },
        "reason": {
            "type": "string"
        }
    },
    "required": ["drop_ids", "add_ids"],
    "additionalProperties": False
}

@register_agent("SimpleLLM")
class SimpleLLMAgent(IPDAgent):
    def __init__(
        self,
        model: Model,
        cell: Cell,
        rewiring_aware: bool = False,
        relationship_data_mode: str = "summary",
        rewiring_recent_k: int = 3,
    ):
        super().__init__(model, cell)
        self.rewiring_aware = rewiring_aware
        self.relationship_data_mode = relationship_data_mode
        self.rewiring_recent_k = rewiring_recent_k
        self._pending_rewiring_plan = None


    def decide_against(
        self,
        other: "IPDAgent",
        payoff_matrix: dict[tuple[str, str], tuple[str, str]],
        give_rationale: bool,
    ) -> tuple[str, str]:
        prompt = f"""You are playing an iterated prisoner's dilemma for {self.model.num_iter} rounds.

Payoffs to you:
CC -> {payoff_matrix['C','C'][0]}
DD -> {payoff_matrix['D','D'][0]}
DC -> {payoff_matrix['D','C'][0]}
CD -> {payoff_matrix['C','D'][0]}

History against this opponent:
{self.serialize_history(self.history, other.unique_id)}

Turns remaining including this one: {self.model.num_iter - self.model.steps + 1}

Choose the move that maximizes your total payoff over the entire game."""

        if self.rewiring_aware:
            prompt += (
                "\nAfter this round, you will have the opportunity to sever "
                "connections with current opponents and have them replaced "
                "with new opponents drawn from your friends-of-friends.\n"
            )

        if give_rationale:
            num_predict = 256
            schema = MOVE_PLUS_RATIONALE_DECISION_SCHEMA
        else:
            num_predict = 16
            schema = MOVE_ONLY_DECISION_SCHEMA

        resp = self._call_ollama_json(
            prompt,
            num_predict,
            schema=schema,
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


    def _call_ollama(
        self,
        prompt: str,
        num_predict: int,
        system: str | None = None,
        response_format: str | dict | None = None,
        temperature: float = 0.0,
    ) -> str:
        payload = {
            "model": self.model.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "seed": 123,
                "temperature": temperature,
                "num_ctx": 2048,
                "num_predict": num_predict,
            },
        }

        if system is not None:
            payload["system"] = system

        if response_format is not None:
            payload["format"] = response_format

        r = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=120,
        )
        r.raise_for_status()

        data = r.json()

        if "response" not in data:
            raise RuntimeError(f"Unexpected Ollama response payload: {data}")

        if data.get("done") is False:
            raise RuntimeError(f"Ollama did not finish generation: {data}")

        done_reason = data.get("done_reason")
        if done_reason == "length":
            raise RuntimeError("Ollama output was truncated (done_reason='length'). Increase num_predict.")

        return data["response"].strip()


    def _call_ollama_json(
        self,
        prompt: str,
        num_predict: int,
        *,
        schema: dict,
        system: str | None = None,
    ) -> dict:
        effective_system = (
            system
            or "Return only valid JSON matching the provided schema. "
               "Do not output any text outside the JSON object."
        )

        text = self._call_ollama(
            prompt=prompt,
            num_predict=num_predict,
            system=effective_system,
            response_format=schema,
            temperature=0.0,
        )

        try:
            obj = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Model returned invalid JSON: {text[:500]!r}",
            ) from e

        if not isinstance(obj, dict):
            raise ValueError(
                f"Expected top-level JSON object, got {type(obj).__name__}",
            )

        return obj

#    def _call_ollama(self, prompt: str, num_predict: int) -> str:
#        r = requests.post(
#            "http://localhost:11434/api/generate",
#            json={
#                "model": self.model.ollama_model,
#                "prompt": prompt,
#                "options": {
#                    "seed": 123,
#                    "num_ctx": 2048,
#                    "num_predict": num_predict,
#                    "temperature": 0,
#                },
#                "stream": False,
#            },
#        )
#        data = r.json()
#        return data["response"]

    def shape(self) -> str:
        return "h"   # Hex

    def size(self) -> str:
        return 2000

    def summarize_relationship(self, history, recent_k=None):
        if recent_k is None:
            recent_k = self.rewiring_recent_k

        rounds_played = len(history)
        recent = history[-recent_k:]
        recent_pairs = [
            [m["self_move"], m["other_move"]]
            for m in recent
        ]
        other_defections = sum(m["other_move"] == "D" for m in history)
        other_cooperations = sum(m["other_move"] == "C" for m in history)
        mutual_c = sum(
            m["self_move"] == "C" and m["other_move"] == "C"
            for m in history
        )
        mutual_d = sum(
            m["self_move"] == "D" and m["other_move"] == "D"
            for m in history
        )

        trailing_d = 0
        for m in reversed(history):
            if m["other_move"] == "D":
                trailing_d += 1
            else:
                break

        return {
            "rounds_played": rounds_played,
            "recent_pairs": recent_pairs,
            "other_coop_rate": (
                other_cooperations / rounds_played if rounds_played else 0.0
            ),
            "other_defect_rate": (
                other_defections / rounds_played if rounds_played else 0.0
            ),
            "other_last_n_defections": trailing_d,
            "mutual_cooperation_count": mutual_c,
            "mutual_defection_count": mutual_d,
        }


    def _serialize_partner_for_rewiring(self, node: int, history):
        if self.relationship_data_mode == "raw":
            return {
                "node": node,
                "history": [
                    {
                        "step": move["step"],
                        "self_move": move["self_move"],
                        "other_move": move["other_move"],
                    }
                    for move in history
                ],
            }

        summary = self.summarize_relationship(history)
        summary["node"] = node
        return summary


    def _plan_rewiring_once(
        self,
        starting_neighbors: set[int],
        new_neighbor_candidates: set[int],
        max_rewires: int,
        give_rationale: bool = False,
    ) -> dict:
        current_neighbor_lines = []
        for node in sorted(starting_neighbors):
            hist = self.serialize_history(self.history, node + 1)
            current_neighbor_lines.append(
                f"- node={node}\n{hist}"
            )

        current_neighbor_text = (
            "\n".join(current_neighbor_lines)
            if current_neighbor_lines else
            "(none)"
        )

        candidate_lines = []
        for node in sorted(new_neighbor_candidates):
            mutual_contacts = sorted(
                starting_neighbors & self._get_neighbors_of_node(node)
            )
            candidate_lines.append(
                f"- node={node}, mutual_contacts={mutual_contacts}"
            )

        candidate_text = (
            "\n".join(candidate_lines)
            if candidate_lines else
            "(none)"
        )

        prompt = f"""You are deciding how to rewire your social network in a networked iterated prisoner's dilemma.

    You may sever up to {max_rewires} current neighbors and add up to {max_rewires} new neighbors.

    Your goal is to maximize your future total payoff.

    Current neighbors and your history against each:
    {current_neighbor_text}

    Available candidate new neighbors:
    {candidate_text}
    """

        if give_rationale:
            prompt += "\nInclude a brief reason."
        else:
            prompt += "\nDo not include a reason."

        resp = self._call_ollama_json(
            prompt=prompt,
            num_predict=192 if give_rationale else 96,
            schema=REWIRING_DECISION_SCHEMA,
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
