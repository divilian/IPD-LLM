"""Immutable records of simulator activity."""

from collections.abc import Iterable
from dataclasses import dataclass
import json
from pathlib import Path

from ipd_llm.policies import Action


@dataclass(frozen=True)
class InteractionRecord:
    """One completed Prisoner's Dilemma interaction between two agents."""

    simulation_round: int
    first_node: int
    second_node: int
    first_action: Action
    second_action: Action
    first_payoff: float
    second_payoff: float


def _interaction_record_to_dict(
    record: InteractionRecord,
) -> dict[str, int | float | str]:
    """Convert an interaction record to its stable JSON representation."""

    return {
        "record_type": "interaction",
        "simulation_round": record.simulation_round,
        "first_node": record.first_node,
        "second_node": record.second_node,
        "first_action": record.first_action.value,
        "second_action": record.second_action.value,
        "first_payoff": record.first_payoff,
        "second_payoff": record.second_payoff,
    }


def append_records(
    path: str | Path,
    records: Iterable[InteractionRecord],
) -> None:
    """Append interaction records to a JSONL file in iteration order.

    Existing contents are left untouched.  Callers should therefore pass only
    records that have not already been written to the target file.
    """

    with Path(path).open("a", encoding="utf-8") as file:
        for record in records:
            data = _interaction_record_to_dict(record)
            file.write(json.dumps(data, separators=(",", ":")))
            file.write("\n")
