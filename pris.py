"""
pris.py

Iterated Prisoner's Dilemma on an Erdős–Rényi graph with:
- One agent per graph node
- Per-neighbor decisions (true tit-for-tat)
- Optional LLM-driven behavior
- Mesa 3.x-compatible (no mesa.time schedulers)

Install:
  pip install mesa networkx
Run:
  python pd_ergraph_mesa3.py
"""

from __future__ import annotations

import random
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import networkx as nx
from mesa import Agent, Model
from mesa.datacollection import DataCollector




def llm_decision(
    self_node: int,
    other_node: int,
    history: List[Dict],
    persona: str,
    step: int,
) -> str:
    """
    Decide "C" or "D" against a specific neighbor.

    Replace this stub with a real LLM API call.
    Keep it deterministic-ish if you care about reproducibility.
    """
    return random.choice(["C", "D"])


# ------------------------------------------------------------
# Agents
# ------------------------------------------------------------
class IPDAgent(Agent):

    def __init__(self, model: Model, node: int):
        super().__init__(model)
        self.node = node

        # history[other_node] = list of {step, self_action, other_action}
        self.history = defaultdict(list)

        # decisions[other_node] = action for THIS step only
        self.decisions = {}

        self.wealth = 0.0

    def record_interaction(
        self,
        step: int,
        other_node: int,
        self_action: str,
        other_action: str,
    ) -> None:
        self.history[other_node].append(
            {
                "step": step,
                "self_action": self_action,
                "other_action": other_action,
            }
        )

    def decide_against(self, other: "IPDAgent") -> str:
        raise NotImplementedError

    def step(self) -> None:
        """
        Decide actions for all neighbors (per-neighbor decision making).
        Actual payoff resolution is done in the Model.step().
        """
        self.decisions.clear()
        for nbr in self.model.graph.neighbors(self.node):
            other = self.model.node_to_agent[nbr]
            self.decisions[nbr] = self.decide_against(other)


class TitForTatAgent(IPDAgent):
    """Classic per-neighbor tit-for-tat (with optional noise)."""

    def __init__(self, model: Model, node: int, noise: float = 0.10):
        super().__init__(model, node)
        self.noise = noise

    def decide_against(self, other: IPDAgent) -> str:
        h = self.history[other.node]
        if not h:
            return random.choice(["C", "D"])

        if random.random() < self.noise:
            return random.choice(["C", "D"])

        return h[-1]["other_action"]


class LLMPDAgent(IPDAgent):
    """LLM-driven per-neighbor decisions."""

    def __init__(self, model: Model, node: int, persona: Optional[str] = None):
        super().__init__(model, node)
        self.persona = persona or (
            "You are cautious but fair: reciprocate cooperation, punish "
            "defection, and occasionally forgive to restore cooperation."
        )

    def decide_against(self, other: IPDAgent) -> str:
        return llm_decision(
            self_node=self.node,
            other_node=other.node,
            history=self.history[other.node],
            persona=self.persona,
            step=self.model.steps,  # Mesa 3.x counter (auto-managed)
        )


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
class IPDModel(Model):
    """
    Iterated Prisoner's Dilemma on a static Erdős–Rényi graph, Mesa 3.x style.

    Activation: self.agents.shuffle_do("step") (RandomActivation replacement).
    """

    def __init__(
        self,
        N, # num agents
        payoff_matrix: List[Tuple],
        edge_prob: float = 0.05,
        fraction_llm: float = 0.25,
        seed: Optional[int] = None,
        tft_noise: float = 0.10,
    ):
        super().__init__(seed=seed)  # Mesa 3.x: required, seed supported

        self.N = N
        self.payoff_matrix = payoff_matrix

        self.graph = nx.erdos_renyi_graph(N, edge_prob, seed=seed)

        # Create agents, one per node (store mapping for fast lookup)
        self.node_to_agent: Dict[int, IPDAgent] = {}

        for node in self.graph.nodes:
            if self.random.random() < fraction_llm:
                agent: IPDAgent = LLMPDAgent(self, node=node)
            else:
                agent = TitForTatAgent(self, node=node, noise=tft_noise)
            self.node_to_agent[node] = agent

        self.datacollector = DataCollector(
            model_reporters={
                "avg_payoff":
                    lambda m: sum(a.wealth for a in m.agents) / len(m.agents),
                "coop_rate": self._coop_rate,
                "avg_degree":
                    lambda m: (sum(dict(m.graph.degree()).values()) /
                        m.graph.number_of_nodes())
                    if m.graph.number_of_nodes()
                    else 0.0,
            }
        )

    def _coop_rate(self) -> float:
        total = 0
        coop = 0
        for a in self.agents:
            for action in a.decisions.values():
                total += 1
                if action == "C":
                    coop += 1
        return coop / total if total else 0.0

    def step(self) -> None:
        """
        One tick:
        1) Agents decide per neighbor (random activation order)
        2) Resolve each edge once using those dyadic decisions
        3) Record dyadic history + collect data
        """
        # 1) RandomActivation replacement in Mesa 3.x
        self.agents.shuffle_do("step")

        # 2) Resolve interactions per edge
        step_idx = self.steps  # Mesa auto-increments this per step call

        for i, j in self.graph.edges:
            ai = self.node_to_agent[i]
            aj = self.node_to_agent[j]

            a_i = ai.decisions[j]
            a_j = aj.decisions[i]

            p_i, p_j = self.payoff_matrix[(a_i, a_j)]
            ai.wealth += p_i
            aj.wealth += p_j

            ai.record_interaction(step=step_idx, other_node=j, self_action=a_i, other_action=a_j)
            aj.record_interaction(step=step_idx, other_node=i, self_action=a_j, other_action=a_i)

        # 3) Collect data
        self.datacollector.collect(self)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Iterated Prisoner's Dilemma on a graph."
    )
    parser.add_argument(
        "T",
        type=float,
        help="Temptation to defect"
    )
    parser.add_argument(
        "R",
        type=float,
        help="Reward for cooperating"
    )
    parser.add_argument(
        "P",
        type=float,
        help="Punishment for mutual defection"
    )
    parser.add_argument(
        "S",
        type=float,
        help="Sucker's payoff"
    )
    parser.add_argument(
        "N",
        type=int,
        help="Number of agents"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345
    )
    parser.add_argument(
        "--num_iter",
        type=int,
        default=100
    )
    return parser.parse_args()


# ------------------------------------------------------------
# Demo run
# ------------------------------------------------------------
if __name__ == "__main__":

    args = parse_args()
    assert (
        args.T > args.R > args.P > args.S
    ), "Prisoner's dilemma constraint #1 violated (T>R>P>S)."
    assert (
        2*args.R > args.S + args.T
    ), "Prisoner's dilemma constraint #2 violated (2R>T+S)."
        

    # ------------------------------------------------------------
    # Payoff matrix.
    # To be a valid prisoner's dilemma, T > R > P > S.
    # Also, to avoid alternating exploitation, 2R > T + S.
    # ------------------------------------------------------------
    payoff_matrix = {
        ("C", "C"): (args.R, args.R),
        ("C", "D"): (args.S, args.T),
        ("D", "C"): (args.T, args.S),
        ("D", "D"): (args.P, args.P),
    }

    model = IPDModel(
        N=args.N,
        edge_prob=0.05,
        payoff_matrix=payoff_matrix,
        fraction_llm=0.0,
        seed=123,
        tft_noise=0.00,
    )

    for t in range(args.num_iter):
        model.step()
        # quick metrics
        avg_payoff = sum(a.payoff for a in model.agents) / len(model.agents)
        coop_rate = model._coop_rate()
        print(f"Step {t+1:02d} | avg_payoff={avg_payoff:.2f} | coop_rate={coop_rate:.2f}")

