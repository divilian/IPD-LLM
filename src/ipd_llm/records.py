"""Immutable records of simulator activity."""

from collections.abc import Iterable
from dataclasses import dataclass
import json
from pathlib import Path

from ipd_llm.agents import AgentSpec
from ipd_llm.initialization import Initialization
from ipd_llm.policies import (
    Action,
    AlwaysCooperate,
    AlwaysDefect,
    GrimTrigger,
    Pavlov,
    TitForTat,
)


@dataclass(frozen=True)
class InitializationRecord:
    """One realized simulator starting state."""

    nodes: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    node_to_spec: tuple[tuple[int, AgentSpec], ...]

    @classmethod
    def from_initialization(
        cls,
        initialization: Initialization,
    ) -> "InitializationRecord":
        """Capture one initialization as an immutable record."""

        edges = tuple(
            sorted(
                tuple(sorted(edge))
                for edge in initialization.graph.edges
            )
        )
        node_to_spec = tuple(
            sorted(initialization.node_to_spec.items())
        )

        return cls(
            nodes=tuple(sorted(initialization.graph.nodes)),
            edges=edges,
            node_to_spec=node_to_spec,
        )


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


Record = InitializationRecord | InteractionRecord


_POLICY_NAMES = {
    AlwaysCooperate: "always_cooperate",
    AlwaysDefect: "always_defect",
    TitForTat: "tit_for_tat",
    GrimTrigger: "grim_trigger",
    Pavlov: "pavlov",
}


def _policy_name(spec: AgentSpec) -> str:
    """Return the stable serialized name for one action policy."""

    policy_type = type(spec.action_policy)

    try:
        return _POLICY_NAMES[policy_type]
    except KeyError as error:
        raise TypeError(
            f"Unsupported action policy: {policy_type.__name__}"
        ) from error


def _initialization_record_to_dict(
    record: InitializationRecord,
) -> dict[str, object]:
    """Convert an initialization record to stable JSON."""

    return {
        "record_type": "initialization",
        "nodes": list(record.nodes),
        "edges": [
            list(edge)
            for edge in record.edges
        ],
        "node_to_spec": {
            str(node): {
                "action_policy": _policy_name(spec),
            }
            for node, spec in record.node_to_spec
        },
    }


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


def _record_to_dict(record: Record) -> dict[str, object]:
    """Convert any supported record to stable JSON."""

    if isinstance(record, InitializationRecord):
        return _initialization_record_to_dict(record)

    return _interaction_record_to_dict(record)


def append_records(
    path: str | Path,
    records: Iterable[Record],
) -> None:
    """Append records to a JSONL file in iteration order.

    Existing contents are left untouched.  Callers should therefore pass only
    records that have not already been written to the target file.
    """

    with Path(path).open("a", encoding="utf-8") as file:
        for record in records:
            data = _record_to_dict(record)
            file.write(json.dumps(data, separators=(",", ":")))
            file.write("\n")
