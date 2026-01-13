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
  python pris.py
"""

from __future__ import annotations

import requests
import subprocess
from tqdm import tqdm
import re
import time
import math
import argparse
from enum import Enum, auto
import random as py_random  # using only Mesa's rng; this is for a type hint
from collections import defaultdict, Counter
from collections.abc import Mapping
from typing import Dict, List, Tuple, Optional, Type
from dataclasses import dataclass
import logging

import polars as pl
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import networkx as nx
from mesa import Agent, Model
from mesa.datacollection import DataCollector

logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s"
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AgentFactory:
    probs: Mapping[Type, float]

    @classmethod
    def from_cli(
        cls,
        tokens: list[str],
    ) -> "AgentFactory":
        if not tokens or len(tokens) % 2 != 0:
            raise ValueError("--agent_fracs must be AGENT FRAC pairs")

        probs: dict[Type, float] = {}
        it = iter(tokens)
        for name, frac_str in zip(it, it):
            frac = float(frac_str)
            probs[globals()[name + "Agent"]] = frac

        return cls(probs)

    def __post_init__(self) -> None:
        s = sum(self.probs.values())
        if not math.isclose(abs(sum(self.probs.values())), 1.0):
            raise ValueError(f"Probabilities must sum to 1.0, got {s}")

    def plan_classes(
        self,
        n_agents: int,
        rng: py_random.Random,
    ) -> list[Type]:
        """
        Return a list of agent classes of length n_agents.
        """
        plan: list[Type] = []

        counts = {
            cls: int(round(p * n_agents))
            for cls, p in self.probs.items()
        }

        # Fix rounding drift.
        while sum(counts.values()) != n_agents:
            diff = n_agents - sum(counts.values())
            cls = max(self.probs, key=self.probs.get)
            counts[cls] += diff

        for cls, k in counts.items():
            plan.extend([cls] * k)

        rng.shuffle(plan)
        return plan

    def instantiate_all(
        self,
        rng: py_random.Random,
        model: mesa.Model,
        nodes: list[int],
        **kwargs
    ):
        """
        Given a list of node numbers, generate that many Agent subclasses,
        according to the probs mapping that this factory was given at
        instantiation time.
        """
        planned = self.plan_classes(len(nodes), rng=rng)
        return [cls(model, node, **kwargs) for cls, node in zip(planned,nodes)]


def compute_sbm_probs(
    sizes: list[int],
    avg_degree: float,
    homophily_weight: float,   # 0 = only diff-type edges; 1 = only same-type
) -> tuple[float, float]:
    """
    Compute (p_same, p_diff) for a vanilla SBM with:
      - block sizes
      - target average degree
      - homophily weight in [0,1]

    We allocate the expected-edge "budget" E across within/between pair sets,
    then convert expected edges to probabilities. If an extreme is infeasible
    (e.g., too few within-block pairs to hit avg_degree), we clamp at 1.0.
    """
    N = sum(sizes)
    if N <= 1:
        return 0.0, 0.0

    max_edges_same = sum(n * (n - 1) / 2 for n in sizes)
    max_edges_diff = sum(
        sizes[i] * sizes[j]
        for i in range(len(sizes))
        for j in range(i + 1, len(sizes))
    )

    # Target expected total edges for simple undirected graph.
    E = N * avg_degree / 2

    # Split the edge budget between same- and diff- type edges.
    E_same = homophily_weight * E
    E_diff = (1.0 - homophily_weight) * E

    # Convert expected edges to probabilities (clamp to [0,1]).
    p_same = 0.0 if max_edges_same == 0 else min(1.0, E_same / max_edges_same)
    p_diff = 0.0 if max_edges_diff == 0 else min(1.0, E_diff / max_edges_diff)

    return p_same, p_diff



def per_agent_type_stats(
    model: IPDModel,
) -> dict[str, float]:
    """
    Compute per-agent-type averages for the current step.
    Returns a flat dict suitable for a DataFrame row.
    """
    from collections import defaultdict

    agent_counts = defaultdict(int)
    coop_counts = defaultdict(int)
    decision_counts = defaultdict(int)
    wealth_sums = defaultdict(float)

    for agent in model.agents:
        cls = agent.__class__.__name__
        agent_counts[cls] += 1
        wealth_sums[cls] += agent.wealth
        for d in agent.decisions.values():
            decision_counts[cls] += 1
            if d == "C":
                coop_counts[cls] += 1

    row = {}
    for cls in agent_counts:
        row[f"{cls}Coop".replace("Agent", "")] = (
            coop_counts[cls] / decision_counts[cls]
            if decision_counts[cls] > 0
            else 0.0
        )
        row[f"{cls}$".replace("Agent", "",)] = (
            wealth_sums[cls] / agent_counts[cls]
            if agent_counts[cls] > 0
            else 0.0
        )
    return row


def get_prompt(payoff_matrix, history):
    return f"""
        You are a player in an Iterated Prisoner's Dilemma game. In each round,
        you and your opponent will choose to either cooperate or defect. If you both
        cooperate, you will both be awarded ${payoff_matrix[('C','C')][0]}. If you
        cooperate and your opponent defects, you will get
        ${payoff_matrix[('C','D')][0]} and your opponent will get
        ${payoff_matrix[('C','D')][1]}. If you defect and your opponent cooperates, you
        will get ${payoff_matrix[('D','C')][0]} and your opponent will get
        ${payoff_matrix[('D','C')][1]}. If you both defect, you will both be awarded
        ${payoff_matrix[('D','D')][0]}. This is the first iteration of the game
        (neither player has moved yet). Do you choose to Cooperate, or Defect? Give one
        word as your response: either the word Cooperate or the word Defect."
    """

def start_llm_server():
    subprocess.Popen(
        ["bash", "bin/start_llm.sh"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

def llm_decision(
    self_node: int,
    other_node: int,
    history: List[Dict],
    payoff_matrix: List[Tuple],
    persona: str,
    step: int,
) -> str:
    """
    Decide "C" or "D" against a specific neighbor.
    """
    try:
        r = requests.post(
            "http://127.0.0.1:8080/completion",
            json={
                "prompt": get_prompt(payoff_matrix, history) + "\nAnswer:",
                "n_predict": 5,
                "temperature": 0.0,
                "top_k": 1,
                "top_p": 1.0,
                "grammar": 'root ::= "Cooperate" | "Defect"',
                "stop": ["\n"],
            },
            timeout=10,
        )
        r.raise_for_status()
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
        requests.exceptions.HTTPError) as e:
        print("Starting Llama server...")
        start_llm_server()

    answer = r.json()["content"]
    if answer not in ["Cooperate","Defect"]:
        raise ValueError(f"LLM didn't follow instructions! Gave {answer}.")
    return answer[0]


# ------------------------------------------------------------
# Agents
# ------------------------------------------------------------
class IPDAgent(Agent):

    def __init__(self, model: Model, node: int):
        super().__init__(model)
        self.node = node
        self.current_iter_payment = 0

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

    def decide_against(
        self,
        other: "IPDAgent",
        payoff_matrix: List[Tuple],
    ) -> tuple[str, str]:
        """
        Make a decision against another agent. Return your decision ("C" or
        "D") and a description of the interaction (for logging).
        """
        raise NotImplementedError

    def shape(self) -> str:
        """
        Return the shape your node should be in the graph. See:
        https://matplotlib.org/stable/api/markers_api.html.
        """
        raise NotImplementedError

    def step(self) -> None:
        """
        Decide your actions for all neighbors (per-neighbor decision making).
        (Actual payoff resolution is done in the Model.step().)
        """
        self.decisions.clear()
        for nbr in self.model.graph.neighbors(self.node):
            other = self.model.node_to_agent[nbr]
            self.decisions[nbr], desc = self.decide_against(other,
payoff_matrix)
            logging.info(desc)

    def __str__(self) -> str:
        return (
            f"Node {self.model.agent_to_node[self]} "
            f"({self.__class__.__name__}) "
            f"with ${int(self.wealth)}"
        )

class SuckerAgent(IPDAgent):
    """Always cooperates."""

    def __init__(self, model: Model, node: int):
        super().__init__(model, node)

    def decide_against(
        self,
        other: "IPDAgent",
        payoff_matrix: List[Tuple],
    ) -> tuple[str, str]:
        log = f"I'm node {self.node} (Sucker), interacting with {other.node}. "
        log += "(C'ing as always.)"
        return "C", log

    def shape(self) -> str:
        return "o"   # Circle = "soft/friendly/harmless" vibe


class MeanAgent(IPDAgent):
    """Always defects."""

    def __init__(self, model: Model, node: int):
        super().__init__(model, node)

    def decide_against(
        self,
        other: "IPDAgent",
        payoff_matrix: List[Tuple],
    ) -> tuple[str, str]:
        log = f"I'm node {self.node} (Mean), interacting with {other.node}. "
        log += "(D'ing as always.)"
        return "D", log

    def shape(self) -> str:
        return "v"   # Down triangle = "mean/aggressive"


class TitForTatAgent(IPDAgent):
    """Classic per-neighbor tit-for-tat (with optional noise)."""

    def __init__(self, model: Model, node: int, noise: float = 0.10):
        super().__init__(model, node)
        self.noise = noise

    def decide_against(
        self,
        other: "IPDAgent",
        payoff_matrix: List[Tuple],
    ) -> tuple[str, str]:
        log = f"I'm node {self.node} (TFT), interacting with {other.node}. "
        h = self.history[other.node]
        if not h:
            choice = self.model.random.choice(["C", "D"])
            log += f"It's my first time! ({choice})."
            return choice, log

        if self.model.random.random() < self.noise:
            choice = self.model.random.choice(["C", "D"])
            log += f"I'm going random ({choice})."
            return choice, log

        log += (f"\n  Last time node {other.node} {h[-1]['other_action']}'d " +
            f"against me. So I'm {h[-1]['other_action']}'ing them this time.")
        return h[-1]["other_action"], log

    def shape(self) -> str:
        return "s"   # Square = "rule-based/fair/predictable"


class LLMAgent(IPDAgent):
    """LLM-driven per-neighbor decisions."""

    def __init__(self, model: Model, node: int, persona: Optional[str] = None):
        super().__init__(model, node)
        self.persona = persona or (
            "You are cautious but fair: reciprocate cooperation, punish "
            "defection, and occasionally forgive to restore cooperation."
        )

    def decide_against(
        self,
        other: "IPDAgent",
        payoff_matrix: List[Tuple],
    ) -> tuple[str, str]:
        decision = llm_decision(
            self_node=self.node,
            other_node=other.node,
            history=self.history[other.node],
            payoff_matrix=payoff_matrix,
            persona=self.persona,
            step=self.model.steps,  # Mesa 3.x counter (auto-managed)
        )
        log = f"I'm node {self.node} (LLM), interacting with {other.node}. "
        log += "I'm {decision}'ing."
        return decision, log

    def shape(self) -> str:
        return "h"   # Hexagon = "tech/engineered/complex"



class IPDModel(Model):
    """
    Iterated Prisoner's Dilemma on a static graph, Mesa 3.x style.

    Graph is generated via an SBM using avg_degree + homophily_weight.
    """

    def __init__(
        self,
        N,  # num agents
        avg_degree: float,
        homophily_weight: float,
        payoff_matrix: List[Tuple],
        agent_factory: AgentFactory,
        seed: int,
    ):
        super().__init__(seed=seed)

        self.N = N
        self.seed = seed
        self.payoff_matrix = payoff_matrix

        # Distinct classes = SBM blocks (stable order)
        classes = sorted(agent_factory.probs, key=lambda c: c.__name__)
        sizes = [ int(round(agent_factory.probs[c] * N)) for c in classes ]

        # ---------------------------------------------------------------
        # 2) Compute SBM probabilities from avg_degree + homophily_weight
        # ---------------------------------------------------------------
        p_same, p_diff = compute_sbm_probs(
            sizes=sizes,
            avg_degree=avg_degree,
            homophily_weight=homophily_weight,
        )

        k = len(classes)
        p = [[p_diff] * k for _ in range(k)]
        for i in range(k):
            p[i][i] = p_same

        # ------------------------------------------------------------
        # 3) Build SBM graph (nodes grouped by block)
        # ------------------------------------------------------------
        self.graph = nx.stochastic_block_model(sizes, p, seed=seed)

        nodes = list(self.graph.nodes)
        agents = []
        idx = 0
        for cls, sz in zip(classes, sizes):
            for _ in range(sz):
                node = nodes[idx]
                agents.append(cls(self, node))
                idx += 1

        self.node_to_agent = dict(zip(nodes, agents))
        self.agent_to_node = dict(zip(agents, nodes))

        self.datacollector = DataCollector(
            model_reporters={
                "avg_payoff":
                    lambda m: sum(a.wealth for a in m.agents) / len(m.agents),
                "coop_rate": self._coop_rate,
                "avg_degree":
                    lambda m: (
                        sum(dict(m.graph.degree()).values())
                        / m.graph.number_of_nodes()
                    )
                    if m.graph.number_of_nodes()
                    else 0.0,
            }
        )

    def assortativity(self) -> float:
        """
        Return the graph's assortativity by agent type. A value of 0 means
        agents are indifferent to what type they connected to. A positive value
        means they're more likely to connect to the same type (a TitForTat to
        other TitForTats, e.g.). A negative value means they're more likely to
        connect to different types (a TitForTat to Means and Suckers, e.g.)
        """
        nx.set_node_attributes(
            self.graph,
            {node: agent.__class__.__name__
             for node, agent in self.node_to_agent.items()},
            name="agent_type",
        )
        return nx.attribute_assortativity_coefficient(self.graph, "agent_type")

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
        # Reset payment totals for this new round.
        for a in self.agents:
            a.current_iter_payment = 0.0

        # Tell each agent to make its decision.
        self.agents.shuffle_do("step")

        # Now, for each game that was played, record it and tally its winnings.
        for i, j in self.graph.edges:
            ai = self.node_to_agent[i]
            aj = self.node_to_agent[j]

            a_i = ai.decisions[j]
            a_j = aj.decisions[i]

            p_i, p_j = self.payoff_matrix[(a_i, a_j)]
            ai.current_iter_payment += p_i
            aj.current_iter_payment += p_j

            ai.record_interaction(
                step=self.steps,
                other_node=j,
                self_action=a_i,
                other_action=a_j,
            )
            aj.record_interaction(
                step=self.steps,
                other_node=i,
                self_action=a_j,
                other_action=a_i,
            )

        # Award the winnings, but scale by the node's degree so that
        # higher-degree nodes don't have an inherent advantage.
        for a in self.agents:
            k = self.graph.degree[a.node]
            if k > 0:
                a.wealth += a.current_iter_payment / k

        self.datacollector.collect(self)

    def agent_mix(self) -> dict[str, int]:
        """
        Return counts of agents by concrete subclass name.
        """
        return dict(
            Counter(agent.__class__.__name__ for agent in self.agents)
        )

    def setup_plotting(self) -> None:
        """
        This method must be called once before subsequent calls to .plot() are
        made. It sets up instance variables:
            - fig, ax (matplotlib objects used thereafter)
            - pos (layout positions of nodes)
            - cmap (the colormap used to plot wealth as color)
            - norm (the normalizer used to scale wealths for plotting)
        """
        self.pos = nx.spring_layout(self.graph, seed=m.seed, k=1.2)
        self.cmap = mpl.colormaps["coolwarm"]  # blue->white->red
        self.fig, self.ax = plt.subplots(constrained_layout=True)
        self.ax.set_axis_off()
        self.norm = Normalize(
            vmin=0,
            vmax=estimate_expected_avg_wealth(self.graph),
            clip=True,
        )
        sm = ScalarMappable(norm=self.norm, cmap=self.cmap)
        sm.set_array([])
        self.fig.colorbar(sm, label="wealth", ax=self.ax)

    def plot(self):
        self.ax.clear()
        nx.draw_networkx_edges(
            self.graph,
            pos=self.pos,
            edge_color="black",
            width=1.0,
            ax=self.ax
        )
        nx.draw_networkx_labels(
            self.graph,
            pos=self.pos,
            font_size=10,
            font_color="black",
            ax=self.ax
        )
        nodes = list(self.graph.nodes())
        colors = [self.cmap(self.norm(w)) for w in monies]
        shapes = [self.node_to_agent[i].shape() for i in nodes]
        for shape in set(shapes):
            idx = [i for i, s in enumerate(shapes) if s == shape]
            nx.draw_networkx_nodes(
                self.graph,
                pos=self.pos,
                nodelist=[nodes[i] for i in idx],
                node_color=[colors[i] for i in idx],
                node_shape=shape,
                node_size=350,
                ax=self.ax,
            )
        self.fig.suptitle(f"Iteration {t+1} of {args.num_iter}")
        plt.pause(0.1)

    def __str__(self):
        ret_val = "with this agent mix:\n"
        am = [ f"{c:>18}:{n:>5}" for c, n in self.agent_mix().items() ]
        ret_val += "\n".join(am)
        ret_val += f"\nThe graph has {self.graph.size()} edges "
        ret_val += f"and assortativity {self.assortativity():.3f}."
        return ret_val


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
        "--agent-fracs",
        nargs="+",
        metavar=("AGENT", "FRAC"),
        help="Agent mix as pairs: AGENT FRAC AGENT FRAC ...",
        default=["Sucker", 1.0]
    )
    parser.add_argument(
        "--avg_degree",
        type=float,
        default=0.20,
        help="Target average degree of graph. (For ER, this is edge_prob.)"
    )
    parser.add_argument(
        "--homophily-weight",
        type=float,
        default=0.5,
        help="fraction of each agent's expected edge budget allocated to "
            "same-type agents",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Seed for all rng's."
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot animation."
    )
    parser.add_argument(
        "--num-iter",
        type=int,
        default=100
    )

    args = parser.parse_args()

    if not args.T > args.R > args.P > args.S:
        raise ValueError("PD constraint #1 violated (T>R>P>S).")
    if not 2*args.R > args.S + args.T:
        raise ValueError("PD constraint #2 violated (2R>T+S).")

    return args



def estimate_expected_avg_wealth(g: Graph):
    """
    Completely back-of-the-envelope estimate of "about how much should each
    agent expect to win during this situation?" The crude formula assumes an
    independent 50/50 chance of choosing to defect or cooperate.
    Note that we compute "per_iter" here (not "per_encounter") because we're
    discounting each agent's per-iteration winnings by its degree (which is its
    number of encounters).
    """
    avg_agent_per_iter = .25 * (args.R + args.T + args.S + args.P)
    return avg_agent_per_iter * args.num_iter



def print_stats(stats: pl.DataFrame, last_n=20):
    cols = stats.columns

    things = sorted({
        re.sub(r"(Coop|\$)$", "", c)
        for c in cols
        if c.endswith("Coop") or c.endswith("$")
    })

    ordered = (
        [f"{t}Coop" for t in things if f"{t}Coop" in cols] +
        [f"{t}$"    for t in things if f"{t}$"    in cols]
    )

    other = [c for c in cols if c not in ordered]

    stats = stats.select(other + ordered)

    with pl.Config(
        tbl_hide_dataframe_shape=True,
        tbl_hide_column_data_types=True,
        float_precision=2,
        tbl_cell_numeric_alignment="RIGHT",
    ):
        print(stats.tail(last_n))


def interact_with_model(m: IPDModel):

    def node_prompt(m):
        return f"Enter node (0-{len(m.agents)-1},'done'): "
    def neigh_prompt(m,n):
        neigh_list = ','.join([ str(k) for k in m.graph.neighbors(n) ])
        return (
            f"  Enter neighbor of {n} ({neigh_list},'done'): "
        )

    node_num_str = input(node_prompt(m))
    while node_num_str != "done":
        n = int(node_num_str)
        print(m.node_to_agent[n])
        neighs = m.graph.neighbors(n)
        if neighs:
            print("Neighbors:")
            for neigh in neighs:
                print(f"  - {m.node_to_agent[neigh]}")
            node_num_str = input(neigh_prompt(m,n))
            while node_num_str != "done":
                neigh = int(node_num_str)
                if neigh in m.graph.neighbors(n):
                    ncn = m.node_to_agent[neigh].__class__.__name__
                    print(f"History with {ncn} {neigh}:")
                    print(
                        pl.DataFrame(m.node_to_agent[n].history[neigh]).rename(
                            {
                                'self_action':f'Node {n}',
                                'other_action':f'Node {neigh}'
                            }
                        )
                    )
                else:
                    print(f"  (Node {n} not adjacent to {neigh}.)")
                node_num_str = input(neigh_prompt(m,n))
                
        node_num_str = input(node_prompt(m))

if __name__ == "__main__":

    args = parse_args()

    stats = []

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

    factory = AgentFactory.from_cli(args.agent_fracs)

    m = IPDModel(
        N=args.N,
        avg_degree=args.avg_degree,
        homophily_weight=args.homophily_weight,
        payoff_matrix=payoff_matrix,
        agent_factory=factory,
        seed=args.seed,
    )
    print(f"Running {m}")

    if args.plot:
        m.setup_plotting()

    for t in tqdm(range(args.num_iter)):
        m.step()
        monies = [m.node_to_agent[i].wealth for i in range(m.N)]
        if args.plot:
            m.plot()
        row = {"step": t + 1}
        row.update(per_agent_type_stats(m))
        stats.append(row)

    stats = pl.DataFrame(stats)
    print_stats(stats)

    interact_with_model(m)
