import json
import requests
from mesa import Model
from mesa.discrete_space import Cell, CellAgent

from .base import IPDAgent, register_agent


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

Choose the action that maximizes your total payoff over the entire game.
"""

        if self.rewiring_aware:
            prompt += (
                "\nAfter this round, you may have the opportunity to sever "
                "connections with current opponents and have them replaced with "
                "new opponents drawn from your friends-of-friends.\n"
            )

        if give_rationale:
            prompt += """
Reply in exactly this format:
C, short reason
or
D, short reason
"""
            num_predict = 128
        else:
            prompt += "Reply with exactly one character: C, or D."
            num_predict = 4

        prompt += "Your move: "

        r = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": self.model.ollama_model,
                "prompt": prompt,
                "options": {
                    "seed": 123,
                    "num_ctx": 1024,
                    "num_predict": num_predict,
                },
                "stream": False,
            },
        )
        resp = r.json()["response"]

        if give_rationale:
            decision, rationale = resp.split(",", 1)
        else:
            decision = resp[0].upper()
            rationale = None

        with open(self.model.llm_out_file, "a", encoding="utf-8") as f:
            print(
                "---------------------------------------------------\n"
                + self.serialize_history(self.history, other.unique_id),
                file=f,
            )
            print(f"DECISION: {decision}. RATIONALE:{rationale}", file=f)

        decision = decision.strip().upper()
        rationale = rationale.strip() if rationale else rationale
        return decision, rationale

    def shape(self) -> str:
        return "h"   # Hex

    def size(self) -> str:
        return 2000

    def is_adjacent_to_node(self, x):
        return any(
            (cell.coordinate == x for cell in self.cell.neighborhood.cells)
        )

    def summarize_relationship(self, history, recent_k=None):
        if recent_k is None:
            recent_k = self.rewiring_recent_k

        rounds_played = len(history)
        recent = history[-recent_k:]
        recent_pairs = [
            [m["self_action"], m["other_action"]]
            for m in recent
        ]
        other_defections = sum(m["other_action"] == "D" for m in history)
        other_cooperations = sum(m["other_action"] == "C" for m in history)
        mutual_c = sum(
            m["self_action"] == "C" and m["other_action"] == "C"
            for m in history
        )
        mutual_d = sum(
            m["self_action"] == "D" and m["other_action"] == "D"
            for m in history
        )

        trailing_d = 0
        for m in reversed(history):
            if m["other_action"] == "D":
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
                        "self_action": move["self_action"],
                        "other_action": move["other_action"],
                    }
                    for move in history
                ],
            }

        summary = self.summarize_relationship(history)
        summary["node"] = node
        return summary

    def _call_ollama_json(self, prompt: str, num_predict: int = 256):
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": self.model.ollama_model,
                "prompt": prompt,
                "options": {
                    "seed": 123,
                    "num_ctx": 2048,
                    "num_predict": num_predict,
                },
                "stream": False,
            },
        )
        resp = r.json()["response"].strip()
        return json.loads(resp)

    def ask_llm_rewiring_decisions(self, partner_data, payoff_matrix):
        rewiring_notice = (
            "After each round, you may choose for each current partner whether to KEEP or DROP that connection. "
            "If you DROP a current partner, that relationship ends, and the environment will try to replace it "
            "with a new random opponent chosen from your friends-of-friends if one is available."
            if self.rewiring_aware else
            ""
        )

        partner_label = (
            "summaries of your relationships with current partners"
            if self.relationship_data_mode == "summary"
            else "full raw histories of your relationships with current partners"
        )

        prompt = f"""You are playing an iterated prisoner's dilemma for {self.model.num_iter} rounds.

Payoffs to you:
CC -> {payoff_matrix['C','C'][0]}
DD -> {payoff_matrix['D','D'][0]}
DC -> {payoff_matrix['D','C'][0]}
CD -> {payoff_matrix['C','D'][0]}

Your objective is to maximize your own total payoff over the entire game.
{rewiring_notice}

Below are {partner_label}.
Return JSON only in exactly this form:
{{"decisions": [{{"node": 1, "action": "KEEP"}}, {{"node": 2, "action": "DROP"}}]}}

Current partners:
{json.dumps(partner_data, indent=2)}
"""
        raw = self._call_ollama_json(prompt)
        return raw["decisions"]

    def rewire_as_desired(
        self,
        payoff_matrix: dict[tuple[str, str], tuple[str, str]],
    ):
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
                f"On turn {h['step']}, you chose {h['self_action']} and "
                f"your opponent chose {h['other_action']}.\n"
            )
        return prompt
