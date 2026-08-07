import json

import networkx as nx

from ipd_llm.agents import AgentSpec
from ipd_llm.initialization import Initialization
from ipd_llm.policies import (
    Action,
    AlwaysCooperate,
    AlwaysDefect,
)
from ipd_llm.records import (
    InitializationRecord,
    InteractionRecord,
    append_records,
)


def initialization_record() -> InitializationRecord:
    graph = nx.Graph()
    graph.add_nodes_from([0, 1, 2])
    graph.add_edges_from([(0, 1), (1, 2)])

    initialization = Initialization(
        graph=graph,
        node_to_spec={
            0: AgentSpec(AlwaysCooperate()),
            1: AgentSpec(AlwaysDefect()),
            2: AgentSpec(AlwaysCooperate()),
        },
    )
    return InitializationRecord.from_initialization(initialization)


def interaction_record(
    simulation_round: int = 1,
    first_node: int = 0,
    second_node: int = 1,
    first_action: Action = Action.COOPERATE,
    second_action: Action = Action.DEFECT,
    first_payoff: float = 0,
    second_payoff: float = 5,
) -> InteractionRecord:
    return InteractionRecord(
        simulation_round=simulation_round,
        first_node=first_node,
        second_node=second_node,
        first_action=first_action,
        second_action=second_action,
        first_payoff=first_payoff,
        second_payoff=second_payoff,
    )


def read_json_lines(path):
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
    ]


def test_append_records_writes_one_json_object_per_line(tmp_path):
    path = tmp_path / "records.jsonl"

    append_records(
        path,
        [
            interaction_record(simulation_round=1),
            interaction_record(simulation_round=2),
        ],
    )

    assert len(path.read_text().splitlines()) == 2


def test_append_records_serializes_initialization_record(tmp_path):
    path = tmp_path / "records.jsonl"

    append_records(path, [initialization_record()])

    assert read_json_lines(path) == [
        {
            "record_type": "initialization",
            "nodes": [0, 1, 2],
            "edges": [[0, 1], [1, 2]],
            "node_to_spec": {
                "0": {"action_policy": "always_cooperate"},
                "1": {"action_policy": "always_defect"},
                "2": {"action_policy": "always_cooperate"},
            },
        }
    ]


def test_initialization_record_captures_realized_state():
    graph = nx.Graph()
    graph.add_edge(0, 1)
    initialization = Initialization(
        graph=graph,
        node_to_spec={
            0: AgentSpec(AlwaysCooperate()),
            1: AgentSpec(AlwaysDefect()),
        },
    )

    record = InitializationRecord.from_initialization(initialization)

    graph.add_edge(1, 2)
    initialization.node_to_spec[0] = AgentSpec(AlwaysDefect())

    assert record.nodes == (0, 1)
    assert record.edges == ((0, 1),)
    assert isinstance(
        record.node_to_spec[0][1].action_policy,
        AlwaysCooperate,
    )


def test_append_records_serializes_all_fields(tmp_path):
    path = tmp_path / "records.jsonl"

    append_records(path, [interaction_record()])

    assert read_json_lines(path) == [
        {
            "record_type": "interaction",
            "simulation_round": 1,
            "first_node": 0,
            "second_node": 1,
            "first_action": "C",
            "second_action": "D",
            "first_payoff": 0,
            "second_payoff": 5,
        }
    ]


def test_append_records_preserves_iteration_order(tmp_path):
    path = tmp_path / "records.jsonl"

    append_records(
        path,
        [
            interaction_record(simulation_round=3),
            interaction_record(simulation_round=1),
            interaction_record(simulation_round=2),
        ],
    )

    assert [
        data["simulation_round"]
        for data in read_json_lines(path)
    ] == [3, 1, 2]


def test_append_records_does_not_rewrite_existing_contents(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text('{"existing":true}\n')

    append_records(path, [interaction_record()])

    lines = path.read_text().splitlines()

    assert lines[0] == '{"existing":true}'
    assert json.loads(lines[1])["record_type"] == "interaction"
