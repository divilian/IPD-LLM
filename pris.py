#!/usr/bin/env python
"""
pris.py

Iterated Prisoner's Dilemma on a Stochastic Block Model graph with:
- One agent per graph node
- Sucker, Mean, TitForTat, and LLM agents
- Configurable homophily based on agent type

Install:
  pip install mesa networkx
Run syntax:
  python pris.py -h
"""

from __future__ import annotations

import asyncio
import requests
import httpx
import textwrap
import subprocess
from tqdm import tqdm
import re
import json
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


#---------------------- Simulation and general functions ---------------------

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


def ensure_ollama_running(ollama_model: str):
    """
    Ensure Ollama daemon is running and the required model is available.
    If daemon is not running, start it.
    If model is not present, pull it.
    """

    def daemon_alive() -> bool:
        try:
            r = requests.get(
                "http://127.0.0.1:11434/api/tags",
                timeout=0.5,
            )
            return r.status_code == 200
        except requests.RequestException:
            return False

    # Start daemon (if needed).
    if not daemon_alive():
        print("Starting Ollama daemon...")
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        deadline = time.time() + 30
        while time.time() < deadline:
            if daemon_alive():
                print("Ollama daemon is ready.")
                break
            time.sleep(0.25)
        else:
            raise RuntimeError(
                "Ollama daemon did not start within 30 seconds."
            )

    # Ensure model is available.
    r = requests.get(
        "http://127.0.0.1:11434/api/tags",
        timeout=2.0,
    )
    r.raise_for_status()

    available_models = {
        m["name"]
        for m in r.json().get("models", [])
    }

    if ollama_model not in available_models:
        print(f"Model '{ollama_model}' not found locally.")
        print("Pulling model from Ollama registry...")
        subprocess.run(
            ["ollama", "pull", ollama_model],
            check=True,
        )
        print("Model pull complete.")


def start_llm_server():
    subprocess.Popen(
        ["bash", "bin/start_llm.sh"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    # Wait until it's good and ready.
    print("Waiting until ready...")
    deadline = time.time() + 30   # Give 30 secs
    while time.time() < deadline:
        try:
            r = requests.post(
                "http://127.0.0.1:8080",
                json={ "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                },
                 timeout=0.5,
            )
            # 503 = model not loaded yet → keep waiting
            if r.status_code == 503:
                time.sleep(0.25)
                continue
            print("...ready!")
            return  # server is ready
        except requests.RequestException:
            print("...still waiting...")
            time.sleep(0.25)
    raise RuntimeError("LLM server did not become ready in time")


def estimate_expected_avg_wealth(g: nx.Graph):
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


#--------------------------- Agent generation -------------------------------

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
            self.decisions[nbr], desc = self.decide_against(
                other,
                self.model.payoff_matrix
            )
            logging.info(desc)

    def __str__(self) -> str:
        return (
            f"Node {self.node} (agent id {self.unique_id}) "
            f"{self.__class__.__name__} "
            f"with ${int(self.wealth)}"
        )


#--------------------------- Rule-based agents -------------------------------

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

        log += (
            f"\n  Last time node {other.node} {h[-1]['other_action']}'d "
            + f"against me. So I'm {h[-1]['other_action']}'ing them this time."
        )
        return h[-1]["other_action"], log

    def shape(self) -> str:
        return "s"   # Square = "rule-based/fair/predictable"


#------------------------------- LLM Agents -----------------------------------

class AgentMode(Enum):
    MECHANISTIC = auto()   # Rule-bound and deterministic.
    DELIBERATIVE = auto()  # Autonomous, strategic, social, free to interpret.


@dataclass(frozen=True)
class Persona:
    system: str        # Strategy description.
    mode: AgentMode    # Strategy description.
    sees_payoffs: bool


personas = {
    'tft': Persona(
        system=textwrap.dedent("""
            Your behavior is fixed: you must exactly repeat your opponent's
            most recent action. If this is the first move, choose randomly.
            """
        ).strip(),
        mode=AgentMode.MECHANISTIC,
        sees_payoffs=False,
    ),

    'vanilla': Persona(
        system=textwrap.dedent("""
            You should choose in a way that tries to maximize your total
            rewards over time.
            """
        ).strip(),
        mode=AgentMode.DELIBERATIVE,
        sees_payoffs=True,
    ),
}


def serialize_summary(history: List[Dict]) -> str:
    if not history:
        return "Summary: This is the first move of the game."

    last = history[-1]
    return (
        "Summary:\n"
        f"Opponent's most recent action: {last['other_action']}"
    )


def serialize_history(history: List[Dict]) -> str:
    if not history:
        return textwrap.dedent("""
            This is the first iteration of the game (neither player has moved
            yet).
        """).strip()

    choices = {"C": "Cooperated", "D": "Defected"}

    lines: List[str] = [
        "Here is the history of your interactions with this opponent so far:"
    ]

    for move in history:
        step = move["step"]
        you = choices[move["self_action"]]
        they = choices[move["other_action"]]
        lines.append(f"    On move {step}, you {you} and they {they}.")

    text = "\n".join(lines)
    return text


def serialize_payoffs(
    payoff_matrix: Dict[Tuple[str, str], Tuple[float, float]],
) -> str:
    """
    payoff_matrix maps (your_action, opponent_action) -> (your_payoff,
    opponent_payoff) where actions are "C" (Cooperate) or "D" (Defect).
    """
    return (
        "Payoff matrix:\n"
        "Actions: C = Cooperate, D = Defect\n\n"
        f"If you choose C and your opponent chooses C: "
        f"you get {payoff_matrix[('C','C')][0]}, "
        f"your opponent gets {payoff_matrix[('C','C')][1]}.\n"
        f"If you choose C and your opponent chooses D: "
        f"you get {payoff_matrix[('C','D')][0]}, "
        f"your opponent gets {payoff_matrix[('C','D')][1]}.\n"
        f"If you choose D and your opponent chooses C: "
        f"you get {payoff_matrix[('D','C')][0]}, "
        f"your opponent gets {payoff_matrix[('D','C')][1]}.\n"
        f"If you choose D and your opponent chooses D: "
        f"you get {payoff_matrix[('D','D')][0]}, "
        f"your opponent gets {payoff_matrix[('D','D')][1]}."
    )


IPD_WORLD_PROMPT = textwrap.dedent("""
You are a player in an Iterated Prisoner's Dilemma game.

In each round, you and your opponent each choose exactly one action: Cooperate
or Defect. You and your opponent will each receive rewards based on your
actions, according to a payoff matrix.

The game is repeated with the same opponent over multiple rounds.
""").strip()


SYSTEM_PROMPT_MECHANISTIC = "\n\n".join([
    IPD_WORLD_PROMPT,
    textwrap.dedent("""
    You are a deterministic agent in this simulation.
    You must follow your behavioral rules exactly.
    You must not reinterpret your goals or constraints.
    You are not role-playing or persuading.
    Your output must be exactly one word: Cooperate or Defect.
    """).strip(),
])


SYSTEM_PROMPT_DELIBERATIVE = "\n\n".join([
    IPD_WORLD_PROMPT,
    textwrap.dedent("""
    You are an autonomous agent in a computer simulation.
    You may reason, strategize, persuade, and role-play
    according to your persona.
    Your output must be exactly one word: Cooperate or Defect.
    """).strip()
])


def get_prompt(payoff_matrix, history: List[Dict]) -> str:
    prompt = textwrap.dedent(f"""
        You are a player in an Iterated Prisoner's Dilemma game. In each round,
        you and your opponent will choose to either cooperate or defect. If you
        both cooperate, you'll both be awarded ${payoff_matrix[('C','C')][0]}.
        If you cooperate and your opponent defects, you will get
        ${payoff_matrix[('C','D')][0]} and your opponent will get
        ${payoff_matrix[('C','D')][1]}. If you defect and your opponent
        cooperates, you will get ${payoff_matrix[('D','C')][0]} and your
        opponent will get ${payoff_matrix[('D','C')][1]}. If you both defect,
        you will both be awarded ${payoff_matrix[('D','D')][0]}.
    """).strip()
    if not history:
        prompt += """
            This is the first iteration of the game (neither player has moved
            yet).
        """
    else:
        prompt += serialize_history(history)

    prompt += textwrap.dedent("""
        Do you choose to Cooperate, or Defect?
    """).strip()
    return prompt


class LLMAgent(IPDAgent):
    """LLM-driven per-neighbor decisions."""

    def __init__(
        self,
        model: Model,
        node: int,
        persona: str,
    ):
        super().__init__(model, node)
        if persona not in personas:
            persona_names = ", ".join(personas.keys())
            raise ValueError(f"{persona} not one of {persona_names}.")
        self.persona = persona

    def decide_against(
        self,
        other: "IPDAgent",
        payoff_matrix: List[Tuple],
    ) -> tuple[str, str]:

        decision = self.decisions[other.node]
        log = f"I'm node {self.node} (LLM), interacting with {other.node}. "
        log += f"I'm {decision}'ing."
        return decision, log

    def shape(self) -> str:
        return "h"   # Hexagon = "tech/engineered/complex"


#---------------------- Agent mix parsing / factory --------------------------

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
        if persona not in personas:
            persona_names = ", ".join(personas)
            raise ValueError(
                f"Unknown LLM persona {persona!r}. Must be one of "
                f"{persona_names}."
            )
        return LLMAgent, {"persona": persona}

    try:
        return globals()[name + "Agent"], {}
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


#------------------------------- Model ---------------------------------------

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
        ollama_model_name: str,
        seed: int,
    ):
        super().__init__(seed=seed)

        self.N = N
        self.seed = seed
        self.payoff_matrix = payoff_matrix
        self.ollama_model_name = ollama_model_name

        # Distinct specs = SBM blocks (stable order)
        specs = sorted(
            agent_factory.probs,
            key=lambda spec: (spec[0].__name__, spec[1]),
        )
        sizes = [int(round(agent_factory.probs[c] * N)) for c in specs]

        # Compute SBM probabilities from avg_degree + homophily_weight.
        p_same, p_diff = compute_sbm_probs(
            sizes=sizes,
            avg_degree=avg_degree,
            homophily_weight=homophily_weight,
        )

        k = len(specs)
        p = [[p_diff] * k for _ in range(k)]
        for i in range(k):
            p[i][i] = p_same

        # Build SBM graph (nodes grouped by block).

        # The graph seed might have to dynamically change, since the provided
        # seed might not produce a connected graph. So, for cleanliness, keep
        # track of this possibly-different seed in a new inst var.
        self.graph_seed = seed
        logging.info(f"Trying graph seed {self.graph_seed}...")
        self.graph = nx.stochastic_block_model(sizes, p, seed=self.graph_seed)
        while not nx.is_connected(self.graph):
            self.graph_seed = self.graph_seed + 1
            logging.info(f"Trying graph seed {self.graph_seed}...")
            logging.info(f"p = {p}")
            self.graph = nx.stochastic_block_model(
                sizes, p, seed=self.graph_seed
            )

        nodes = list(self.graph.nodes)
        idx = 0
        self.node_to_agent = {}
        for (agent_cls, kwargs_items), sz in zip(specs, sizes):
            init_kwargs = dict(kwargs_items)
            for _ in range(sz):
                node = nodes[idx]
                agent = agent_cls(self, node, **init_kwargs)
                self.node_to_agent[node] = agent
                idx += 1
        self.agent_to_node = {
            agent: node
            for node, agent in self.node_to_agent.items()
        }

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

        # Peace of mind that node IDs and agent IDs haven't drifted weirdly.
        assert set(self.node_to_agent.keys()) == set(self.graph.nodes)
        assert all(a.node == n for n, a in self.node_to_agent.items())

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
        asyncio.run(self._step_async())

    def _build_batch_prompt(self, agent_payloads):
        return f"""
            You are making decisions for multiple independent agents in a
            Prisoner's Dilemma simulation.

            For each agent below, decide either:
            C = Cooperate
            D = Defect

            Return ONLY valid JSON in this exact format:

            {{
              "decisions": [
                {{"id": 1, "move": "C"}},
                {{"id": 2, "move": "D"}}
              ]
            }}

            Agents:

            {json.dumps(agent_payloads, indent=2)}

            You MUST return a decision for EVERY agent listed.
            If any agent is missing, your response is invalid.
            Do not omit any agent.
        """

    async def _step_async(self) -> None:
        """
        One tick:
        1) Agents decide per neighbor (LLM decisions async)
        2) Resolve each edge once using those dyadic decisions
        3) Record dyadic history + collect data
        """
        # Reset payment totals for this new round.
        for a in self.agents:
            a.current_iter_payment = 0.0
            a.decisions.clear()

        llm_agents = [a for a in self.agents if isinstance(a, LLMAgent)]

        # Make LLM agent decisions.
        if llm_agents:

            agent_payloads = []
            for agent in llm_agents:
                agent_payloads.append({
                    "id": agent.node,   # use node, not unique_id
                    "persona": agent.persona,
                    "neighbors": list(self.graph.neighbors(agent.node)),
                    "wealth": agent.wealth,
                })

            batch_prompt = self._build_batch_prompt(agent_payloads)

            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post(
                    "http://127.0.0.1:11434/api/chat",
                    json={
                        "model": self.ollama_model_name,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a strategic decision engine.",
                            },
                            {"role": "user", "content": batch_prompt},
                        ],
                        "temperature": 0.0,
                        "stream": False,
                    },
                )
                r.raise_for_status()


            content = r.json()["message"]["content"]

            match = re.search(r"\{.*\}", content, re.DOTALL)
            if not match:
                raise ValueError("No JSON found in LLM output")

            parsed = json.loads(match.group(0))
            decision_map = {d["id"]: d["move"] for d in parsed["decisions"]}

            # Verified all parts of the output existed:
            expected_ids = {a.node for a in llm_agents}
            returned_ids = set(decision_map.keys())

            if returned_ids != expected_ids:
                raise ValueError(
                    f"LLM returned incomplete decisions.\n"
                    f"Expected: {expected_ids}\n"
                    f"Returned: {returned_ids}\n"
                    f"Raw content:\n{content}"
                )



            print("LLM agent nodes:",
                  sorted(a.node for a in llm_agents))
            print("Returned IDs:",
                  sorted(decision_map.keys()))



            for agent in llm_agents:
                for nbr in self.graph.neighbors(agent.node):
                    agent.decisions[nbr] = decision_map[agent.node]

        # Make non-LLM agent decisions.
        for agent in self.node_to_agent.values():
            if not isinstance(agent, LLMAgent):
                for nbr in self.graph.neighbors(agent.node):
                    decision, desc = agent.decide_against(
                        self.node_to_agent[nbr],
                        self.payoff_matrix
                    )
                    agent.decisions[nbr] = decision
                    logging.info(desc)

        # Phase 4: resolve payoffs.
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

        # Award the winnings, but scale by the node's degree.
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
        plt.show(block=False)

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
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        time.sleep(0.01)

    def __str__(self):
        ret_val = "with this agent mix:\n"
        am = [f"{c:>18}:{n:>5}" for c, n in self.agent_mix().items()]
        ret_val += "\n".join(am)
        ret_val += f"\nThe graph has {self.graph.size()} edges "
        ret_val += f"and assortativity {self.assortativity():.3f}."
        return ret_val


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


#----------------------------------- main ------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Iterated Prisoner's Dilemma on a graph.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "N",
        type=int,
        help="Number of agents",
    )
    parser.add_argument(
        "--T",
        type=float,
        help="Temptation to defect (default 5)",
        default=5.0,
    )
    parser.add_argument(
        "--R",
        type=float,
        help="Reward for cooperating (default 3)",
        default=3.0,
    )
    parser.add_argument(
        "--P",
        type=float,
        help="Punishment for mutual defection (default 1)",
        default=1.0,
    )
    parser.add_argument(
        "--S",
        type=float,
        help="Sucker's payoff (default 0)",
        default=0.0,
    )
    agent_types = ["Sucker", "Mean", "TitForTat"]
    agent_types += [f"LLM{p}" for p in personas]
    parser.add_argument(
        "--agent-fracs",
        nargs="+",
        metavar=("AGENT", "FRAC"),
        default=["Sucker", 0.5, "Mean", 0.5],
        help=(
            "Agent mix as pairs: AGENT FRAC AGENT FRAC ...\n"
            "AGENT is one of:\n"
            + "".join(f"  - {agent_type}\n" for agent_type in agent_types)
            + "Ex: --agent-fracs Sucker 0.4 Mean 0.4 LLMgrudge 0.2\n"
            + "(default Sucker 0.5, Mean 0.5)"
        ),
    )
    parser.add_argument(
        "--avg-degree",
        type=float,
        default=3.0,
        help="Target average degree of nodes in graph."
    )
    parser.add_argument(
        "--homophily-weight",
        type=float,
        default=0.5,
        help="Fraction of each agent's expected edge budget allocated to\n"
             "same-type agents. (default 0.5)",
    )
    parser.add_argument(
        "--num-iter",
        type=int,
        default=25,
        help="Number of simulation iterations."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Seed for rng's; starting (walking) seed for graph rng."
    )
    parser.add_argument(
        "--tft-noise",
        type=float,
        default=0.10,
        help="TitForTatAgent noise rate in [0,1]. (default 0.10)",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default="llama3.1:latest",
        help="Ollama model to use for LLM agents."
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot animation."
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Launch interactive node analyzer."
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Set logging level.",
    )

    args = parser.parse_args()

    if not args.T > args.R > args.P > args.S:
        raise ValueError("PD constraint #1 violated (T>R>P>S).")
    if not 2*args.R > args.S + args.T:
        raise ValueError("PD constraint #2 violated (2R>T+S).")
    if not (0.0 <= args.tft_noise <= 1.0):
        raise ValueError("--tft-noise must be between 0 and 1")

    return args


def interact_with_model(m: IPDModel):

    def node_prompt(m):
        return (
            f"Enter node ({','.join([str(n) for n in sorted(m.graph.nodes)])},"
            "'done'): "
        )

    def neigh_prompt(m, n):
        neigh_list = ','.join([str(k) for k in m.graph.neighbors(n)])
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
            node_num_str = input(neigh_prompt(m, n))
            while node_num_str != "done":
                neigh = int(node_num_str)
                if neigh in m.graph.neighbors(n):
                    ncn = m.node_to_agent[neigh].__class__.__name__
                    print(f"History with {ncn} {neigh}:")
                    print(
                        pl.DataFrame(m.node_to_agent[n].history[neigh]).rename(
                            {
                                'self_action': f'Node {n}',
                                'other_action': f'Node {neigh}'
                            }
                        )
                    )
                else:
                    print(f"  (Node {n} not adjacent to {neigh}.)")
                node_num_str = input(neigh_prompt(m, n))

        node_num_str = input(node_prompt(m))


if __name__ == "__main__":

    args = parse_args()

    ensure_ollama_running(args.ollama_model)

    logging.basicConfig(
        level=args.log_level,
        format="%(message)s"
    )

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

    factory = AgentFactory.instance(args.agent_fracs, args)

    m = IPDModel(
        N=args.N,
        avg_degree=args.avg_degree,
        homophily_weight=args.homophily_weight,
        payoff_matrix=payoff_matrix,
        agent_factory=factory,
        ollama_model_name=args.ollama_model,
        seed=args.seed,
    )
    print(f"Running {m}")

    if args.plot:
        m.setup_plotting()

    for t in tqdm(range(args.num_iter)):
        m.step()
        monies = [m.node_to_agent[n].wealth for n in m.graph.nodes]
        if args.plot:
            m.plot()
        row = {"step": t + 1}
        row.update(per_agent_type_stats(m))
        stats.append(row)

    stats = pl.DataFrame(stats)
    print_stats(stats)

    if args.analyze:
        interact_with_model(m)
