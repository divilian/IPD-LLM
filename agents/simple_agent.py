import requests
from mesa import Model, Agent

from .base import IPDAgent, register_agent


@register_agent("SimpleLLM")
class SimpleLLMAgent(IPDAgent):
    def __init__(self, model: Model, node: int):
        super().__init__(model, node)
    def decide_against(
        self,
        other: "IPDAgent",
        payoff_matrix: dict[tuple[str, str], tuple[str, str]],
    ) -> tuple[str, str]:
        prompt = f"""
You are a player in an iterated prisoner's dilemma game.
You will be playing for {self.model.num_iter} rounds.
In each round, you privately choose either C or D. Your opponent will also
privately choose between C and D.
If you both choose C, you will each get $3.
If you both choose D, you will each get $1.
If you choose differently, the one who chooses D will get $5 and the one who
chooses C will get nothing.

{self.serialize_history(self.history, other.unique_id)}

There are {self.model.num_iter - self.model.steps + 1} turns left in the game, including the one you are making a decision about now.

Your goal is to maximize your total accumulated winnings in the entire game
(all rounds).

What do you choose? You must choose either C or D for this turn.

Your answer must be in two comma-separated parts. The first part must be the
single letter C or D. Then, after a comma, you should give a short rationale
for why you made this move.

Example of a correct response:
C, Since my opponent has chosen C the past few rounds, I'll assume they will
continue to do so, and choose C now.

Example of an incorrect response:
Since my opponent has chosen C the past few rounds, I'll assume they will
continue to do so, and choose C now.

What is your response?
"""

        print(self.serialize_history(self.history, other.unique_id))

        r = requests.post("http://localhost:11434/api/generate",
            json={
                "model": "llama3.1",
                "prompt": prompt,
                  "options": {
                    "seed": 123,
                    "num_ctx": 2048,
                  },
                "stream": False
            }
        )
        resp = r.json()['response']
        decision, rationale = resp.split(",",1)
        print(f"DECISION: {decision}. RATIONALE: {rationale}")
        return decision, rationale

    def shape(self) -> str:
        return "h"   # Hex

    def serialize_history(self, history, oid):
        if not history:
            return f"This is the first move in your game with this opponent ({oid-1})."
        prompt = f"Here is the history of your games with this opponent ({oid-1}) so far:\n"
        for h in history[oid-1]:
            prompt += f"On turn {h['step']}, you chose {h['self_action']} and your opponent chose {h['other_action']}.\n"
        return prompt
