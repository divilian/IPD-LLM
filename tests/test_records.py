import json

from ipd_llm.policies import Action
from ipd_llm.records import InteractionRecord, append_records


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
