import requests
from mesa import Model
from mesa.discrete_space import CellAgent, Cell

from .base import IPDAgent, register_agent


@register_agent("SimpleLLM")
class SimpleLLMAgent(IPDAgent):
    def __init__(self, model: Model, cell: Cell):
        super().__init__(model, cell)
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

        r = requests.post("http://localhost:11434/api/generate",
            json={
                "model": self.model.ollama_model,
                "prompt": prompt,
                  "options": {
                    "seed": 123,
                    "num_ctx": 1024,
                    "num_predict": num_predict,
                  },
                "stream": False
            }
        )
        resp = r.json()['response']

        if give_rationale:
            decision, rationale = resp.split(",",1)
        else:
            decision = resp[0].upper()
            rationale = None
        with open(self.model.llm_out_file, "a", encoding="utf-8") as f:
            print(
                "---------------------------------------------------\n" +
                self.serialize_history(self.history, other.unique_id),
                file=f,
            )
            print(f"DECISION: {decision}. RATIONALE:{rationale}", file=f)

        return decision, rationale

    def shape(self) -> str:
        return "h"   # Hex

    def size(self) -> str:
        return 2000

    def serialize_history(self, history, oid):
        if not history or not history[oid-1]:
            return f"This is the first move in your game with this opponent ({oid-1}).\n"
        prompt = f"Here is the history of your games with this opponent ({oid-1}) so far:\n"
        for h in history[oid-1]:
            prompt += f"On turn {h['step']}, you chose {h['self_action']} and your opponent chose {h['other_action']}.\n"
        return prompt
