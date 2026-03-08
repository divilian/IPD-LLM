import random

from .backend import LLMBackend, register_backend


@register_backend("mock")
class MockBackend(LLMBackend):

    @classmethod
    def from_args(cls, args) -> "LLMBackend":
        return cls()

    async def batch_decide(self, payloads):
        decisions = []

        for agent in payloads:
            for opp in agent["opponents"]:
                decisions.append({
                    "id": agent["id"],
                    "opponent": opp,
                    "move": random.choice(["C", "D"]),
                })

        return {"decisions": decisions}
