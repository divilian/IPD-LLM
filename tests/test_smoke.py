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

    assert len(lines) == len(model.records)
    assert len(lines) > 0

    first = json.loads(lines[0])

    assert first["record_type"] == "interaction"
    assert first["simulation_round"] == 1
    assert first["first_action"] in {"C", "D"}
    assert first["second_action"] in {"C", "D"}
