from dataclasses import fields

import networkx as nx

from ipd_llm.agents import AgentSpec
from ipd_llm.initialization import create_initialization
from ipd_llm.policies import (
    AlwaysCooperate,
    AlwaysDefect,
    GrimTrigger,
    Pavlov,
    TitForTat,
)


def specs() -> list[AgentSpec]:
    return [
        AgentSpec(AlwaysCooperate()),
        AgentSpec(AlwaysDefect()),
        AgentSpec(TitForTat()),
        AgentSpec(GrimTrigger()),
        AgentSpec(Pavlov()),
        AgentSpec(AlwaysCooperate()),
        AgentSpec(AlwaysDefect()),
        AgentSpec(TitForTat()),
        AgentSpec(GrimTrigger()),
        AgentSpec(Pavlov()),
        AgentSpec(AlwaysCooperate()),
        AgentSpec(AlwaysDefect()),
    ]


def policy_types(initialization) -> dict[int, type]:
    return {
        node: type(spec.action_policy)
        for node, spec in initialization.node_to_spec.items()
    }


def test_same_seed_reproduces_graph_and_placement():
    first = create_initialization(
        seed=12345,
        agent_specs=specs(),
        k=4,
        p=0.25,
    )
    second = create_initialization(
        seed=12345,
        agent_specs=specs(),
        k=4,
        p=0.25,
    )

    assert set(first.graph.edges) == set(second.graph.edges)
    assert policy_types(first) == policy_types(second)


def test_initialization_contains_only_realized_state():
    initialization = create_initialization(
        seed=12345,
        agent_specs=specs(),
        k=4,
        p=0.25,
    )

    assert [field.name for field in fields(initialization)] == [
        "graph",
        "node_to_spec",
    ]


def test_placement_is_independent_of_graph_generation():
    first = create_initialization(
        seed=12345,
        agent_specs=specs(),
        k=2,
        p=0.0,
    )
    second = create_initialization(
        seed=12345,
        agent_specs=specs(),
        k=4,
        p=1.0,
    )

    assert policy_types(first) == policy_types(second)


def test_initial_graph_is_connected():
    initialization = create_initialization(
        seed=12345,
        agent_specs=specs(),
        k=4,
        p=0.25,
    )

    assert nx.is_connected(initialization.graph)


def test_initialization_maps_every_node_to_one_spec():
    initialization = create_initialization(
        seed=12345,
        agent_specs=specs(),
        k=4,
        p=0.25,
    )

    assert set(initialization.node_to_spec) == set(
        initialization.graph.nodes
    )


def test_initialization_preserves_population_composition():
    original = specs()
    initialization = create_initialization(
        seed=12345,
        agent_specs=original,
        k=4,
        p=0.25,
    )

    original_types = sorted(
        type(spec.action_policy).__name__
        for spec in original
    )
    placed_types = sorted(
        type(spec.action_policy).__name__
        for spec in initialization.node_to_spec.values()
    )

    assert placed_types == original_types


def test_initialization_does_not_reorder_input_sequence():
    original = specs()
    original_order = [
        type(spec.action_policy)
        for spec in original
    ]

    create_initialization(
        seed=12345,
        agent_specs=original,
        k=4,
        p=0.25,
    )

    assert [
        type(spec.action_policy)
        for spec in original
    ] == original_order
