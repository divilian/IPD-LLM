"""Reproducible simulator initialization."""

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import random

import networkx as nx

from ipd_llm.agents import AgentSpec


@dataclass(frozen=True)
class Initialization:
    """One reusable graph and agent placement."""

    seed: int
    graph_seed: int
    placement_seed: int
    graph: nx.Graph
    node_to_spec: dict[int, AgentSpec]


def _derive_seed(seed: int, stream: str) -> int:
    """Derive one stable 64-bit stream seed."""

    data = f"{seed}:{stream}".encode("utf-8")
    digest = hashlib.sha256(data).digest()
    return int.from_bytes(digest[:8], byteorder="big")


def create_initialization(
    seed: int,
    agent_specs: Sequence[AgentSpec],
    k: int,
    p: float,
) -> Initialization:
    """Create a reproducible connected Watts-Strogatz initialization."""

    graph_seed = _derive_seed(seed, "graph")
    placement_seed = _derive_seed(seed, "placement")

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
        seed=seed,
        graph_seed=graph_seed,
        placement_seed=placement_seed,
        graph=graph,
        node_to_spec=node_to_spec,
    )
