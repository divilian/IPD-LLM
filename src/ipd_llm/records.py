"""Immutable records of simulator activity."""

from dataclasses import dataclass

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
