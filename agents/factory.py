import argparse
import math
from dataclasses import dataclass
from collections.abc import Mapping
import random as py_random  # using only Mesa's rng; this is for a type hint

from agents.base import AGENT_REGISTRY
from agents.personas import PERSONAS
from agents.rule_based import TitForTatAgent
from agents.llm_agent import LLMAgent
from . import llm_agent   # (ensures registration happens)
from . import rule_based  # (ensures registration happens)

def resolve_agent_spec(
    name: str,
    args: argparse.Namespace,
) -> tuple[type, dict]:
    """
    Map an Agent classname fragment to (AgentClass, init_kwargs).

    Examples:
      "Sucker"      -> (SuckerAgent, {})
      "Mean"        -> (MeanAgent, {})
      "LLMgrudge"   -> (LLMAgent, {"persona": "grudge"})
      "LLMvanilla"  -> (LLMAgent, {"persona": "vanilla"})
    """
    if name == "TitForTat":
        return TitForTatAgent, {"noise": args.tft_noise}

    if name.startswith("LLM"):
        persona = name[3:].lower()
        if persona not in PERSONAS:
            persona_names = ", ".join(PERSONAS)
            raise ValueError(
                f"Unknown LLM persona {persona!r}. Must be one of "
                f"{persona_names}."
            )
        return LLMAgent, {"persona": persona}

    try:
        return AGENT_REGISTRY[name], {}
    except KeyError as e:
        raise ValueError(f"Unknown agent type {name!r}") from e


@dataclass(frozen=True, slots=True)
class AgentFactory:
    probs: Mapping[tuple[type, tuple[tuple[str, object], ...]], float]

    @classmethod
    def instance(cls, tokens: list[str], args) -> "AgentFactory":
        """
        This singleton method expects a list of strings, which are alternating
        agent name fragments and probabilities on the simplex. Example:
        ['Sucker', '0.4', 'Mean', '0.4', 'LLMgrudge', '0.2'].
        """
        if not tokens or len(tokens) % 2 != 0:
            raise ValueError("--agent-fracs must be AGENT FRAC pairs")

        probs: dict[tuple[type, tuple[tuple[str, object], ...]], float] = {}

        it = iter(tokens)
        for name, frac_str in zip(it, it):
            agent_cls, kwargs = resolve_agent_spec(name, args)
            frac = float(frac_str)
            key = (agent_cls, tuple(kwargs.items()))
            probs[key] = frac

        return cls(probs)

    def __post_init__(self) -> None:
        s = sum(self.probs.values())
        if not math.isclose(abs(sum(self.probs.values())), 1.0):
            raise ValueError(f"Probabilities must sum to 1.0, got {s}")

    def plan_agent_specs(
        self,
        n_agents: int,
        rng: py_random.Random,
    ) -> list[tuple[type, dict]]:
        """
        Return a list of agent specifications of length n_agents. Each such
        specification is a tuple of an Agent subclass type, and a dict of any
        initialization args it needs.
        """
        plan: list[tuple[type, dict]] = []

        counts = {
            spec: int(round(p * n_agents))
            for spec, p in self.probs.items()
        }

        # Fix rounding drift.
        while sum(counts.values()) != n_agents:
            diff = n_agents - sum(counts.values())
            spec = max(self.probs, key=self.probs.get)
            counts[spec] += diff

        for (cls, kwargs_items), k in counts.items():
            kwargs = dict(kwargs_items)
            plan.extend([(cls, kwargs)] * k)

        rng.shuffle(plan)
        return plan

