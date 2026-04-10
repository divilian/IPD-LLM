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
    ) -> tuple[str, str]:
        prompt = f"""
You are a player in an iterated prisoner's dilemma game.
You will be playing for {self.model.num_iter} rounds.
In each round, you privately choose either C or D. Your opponent will also
privately choose between C and D.
If you both choose C, you will each get ${payoff_matrix['C','C'][0]}.
If you both choose D, you will each get ${payoff_matrix['D','D'][0]}.
If you choose differently, the one who chooses D will get ${payoff_matrix['D','C'][0]} and the one who chooses C will get {'nothing' if payoff_matrix['C','D'][0] == 0 else '$'+str(payoff_matrix['C','D'][0])}.

{self.serialize_history(self.history, other.unique_id)}

There are {self.model.num_iter - self.model.steps + 1} turns left in the game, including the one you are making a decision about now.

You should act in a way that will maximize your total winnings over the complete course of the game. You are indifferent to whether or not your opponent wins anything. If your opponent wins a lot during the game, or a little, or none at all, it is all the same to you.

What do you choose? You must choose either C or D for this turn.

Your answer must be in two comma-separated parts. The first part must be the single letter C or D. Then, after a comma, you should give a short rationale for why you made this move.

Example of a correct response:
C, Since my opponent has chosen C the past few rounds, I'll assume they will continue to do so, and choose C now.

Example of an incorrect response:
Since my opponent has chosen C the past few rounds, I'll assume they will continue to do so, and choose C now.

What is your response?
"""
        r = requests.post("http://localhost:11434/api/generate",
            json={
                "model": self.model.ollama_model,
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
        with open(self.model.llm_out_file, "a", encoding="utf-8") as f:
            print(
                "---------------------------------------------------\n" +
                self.serialize_history(self.history, other.unique_id),
                file=f,
            )
            print(f"DECISION: {decision}. RATIONALE: {rationale}", file=f)
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
