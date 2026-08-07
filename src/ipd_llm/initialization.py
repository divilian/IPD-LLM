"""Reproducible simulator initialization."""

from collections.abc import Sequence
from dataclasses import dataclass
import random

import networkx as nx

from ipd_llm.agents import AgentSpec
from ipd_llm.rng import derive_seed


@dataclass(frozen=True)
class Initialization:
    """One reusable graph and agent placement."""

    graph: nx.Graph
    node_to_spec: dict[int, AgentSpec]


def create_initialization(
    seed: int,
    agent_specs: Sequence[AgentSpec],
    k: int,
    p: float,
) -> Initialization:
    """Create a reproducible connected Watts-Strogatz initialization."""

    graph_seed = derive_seed(seed, "graph")
    placement_seed = derive_seed(seed, "placement")

    graph_rng = random.Random(graph_seed)
    placement_rng = random.Random(placement_seed)

    graph = nx.connected_watts_strogatz_graph(
        n=len(agent_specs),
        k=k,
        p=p,
        seed=graph_rng,
    )

    shuffled_specs = list(agent_specs)
    placement_rng.shuffle(shuffled_specs)

    node_to_spec = dict(
        zip(
            sorted(graph.nodes),
            shuffled_specs,
            strict=True,
        )
    )

    return Initialization(
        graph=graph,
        node_to_spec=node_to_spec,
    )
