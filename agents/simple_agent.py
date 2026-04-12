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
            import ipdb ; ipdb.set_trace()
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


    def ask_llm_rewiring_decisions(
        self,
        current_neighbors: list["IPDAgent"],
        candidate_new_neighbors: list["IPDAgent"],
        max_drops: int,
        max_adds: int,
        give_rationale: bool,
    ) -> tuple[list[str], list[str], str]:
        current_neighbor_lines = []
        for agent in current_neighbors:
            hist = self.serialize_history(self.history, agent.unique_id)
            current_neighbor_lines.append(
                f"- id={agent.unique_id}\n{hist}"
            )

        if not current_neighbor_lines:
            current_neighbor_text = "(none)"
        else:
            current_neighbor_text = "\n".join(current_neighbor_lines)

        candidate_lines = []
        for agent in candidate_new_neighbors:
            mutuals = sorted(
                set(self.graph.neighbors(self)) & set(self.graph.neighbors(agent))
            ) if hasattr(self, "graph") else []
            mutual_ids = [a.unique_id if hasattr(a, "unique_id") else str(a) for a in mutuals]

            candidate_lines.append(
                f"- id={agent.unique_id}, mutual_contacts={mutual_ids}"
            )

        if not candidate_lines:
            candidate_text = "(none)"
        else:
            candidate_text = "\n".join(candidate_lines)

        prompt = f"""You are deciding how to rewire your social network in a networked iterated prisoner's dilemma.

    You may drop up to {max_drops} current neighbors and add up to {max_adds} new neighbors.

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

        drop_ids = resp["drop_ids"]
        add_ids = resp["add_ids"]
        reason = resp.get("reason", "(none)")

        if not isinstance(drop_ids, list) or not all(isinstance(x, str) for x in drop_ids):
            raise ValueError(f"Invalid drop_ids returned by LLM: {drop_ids!r}")

        if not isinstance(add_ids, list) or not all(isinstance(x, str) for x in add_ids):
            raise ValueError(f"Invalid add_ids returned by LLM: {add_ids!r}")

        allowed_drop_ids = {agent.unique_id for agent in current_neighbors}
        allowed_add_ids = {agent.unique_id for agent in candidate_new_neighbors}

        drop_ids = [x for x in drop_ids if x in allowed_drop_ids]
        add_ids = [x for x in add_ids if x in allowed_add_ids]

        drop_ids = list(dict.fromkeys(drop_ids))[:max_drops]
        add_ids = list(dict.fromkeys(add_ids))[:max_adds]

        with open(self.model.llm_out_file, "a", encoding="utf-8") as f:
            print("---------------------------------------------------", file=f)
            print("REWIRING DECISION", file=f)
            print(f"DROP: {drop_ids}", file=f)
            print(f"ADD: {add_ids}", file=f)
            print(f"RATIONALE: {reason}", file=f)

        return drop_ids, add_ids, reason


    def rewire_as_desired(
        self,
        payoff_matrix: dict[tuple[str, str], tuple[str, str]],
    ):
        return
        if self.model.debug:
            print(f"***************************************\n"
                f"{self.unique_id} (node {self.node}) is considering rewiring")

        my_foafs = self._get_foaf_nodes()

        # Keep track of who my neighbors are at the start of rewiring.
        # We will exclude all of them from replacement candidates, so that if I
        # sever a connection with someone during this rewiring step, I do not
        # immediately reconnect to that same node.
        starting_neighbors = self._get_current_neighbors()

        partner_data = []
        for node, history in self.history.items():
            if self.is_adjacent_to_node(node):
                partner_data.append(
                    self._serialize_partner_for_rewiring(node, history)
                )

        decisions = self.ask_llm_rewiring_decisions(partner_data, payoff_matrix)
        drop_nodes = {
            d["node"] for d in decisions
            if d["action"] == "DROP" and self.is_adjacent_to_node(d["node"])
        }

        # Keep track of how many connections I've severed, so that I know how
        # many new connections to make when I'm done severing. (So as to
        # preserve this node's degree to the extent possible.)
        self.num_needed_replacements = 0
        severed_nodes = set()

        for node in drop_nodes:
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
                eligible_foafs -= {new_friend_node}
                if self.model.debug:
                    print(f"I'm now friends with {new_friend_node}!")
            else:
                if self.model.debug:
                    print(f"Yikes! Nobody to replace severed connection with.")
                return

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
