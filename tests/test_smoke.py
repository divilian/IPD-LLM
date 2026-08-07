import json

from ipd_llm.model import run_smoke_test


def test_smoke_simulation_runs_and_writes_jsonl(tmp_path):
    output_path = tmp_path / "smoke-records.jsonl"

    model = run_smoke_test(
        output_path,
        rounds=3,
    )

    assert output_path.exists()
    lines = output_path.read_text().splitlines()

    assert len(lines) == len(model.records) + 1
    assert len(lines) > 0

    first = json.loads(lines[0])

    assert first["record_type"] == "initialization"
    assert set(first) == {
        "record_type",
        "nodes",
        "edges",
        "node_to_spec",
    }

    second = json.loads(lines[1])

    assert second["record_type"] == "interaction"
    assert second["simulation_round"] == 1
    assert second["first_action"] in {"C", "D"}
    assert second["second_action"] in {"C", "D"}
