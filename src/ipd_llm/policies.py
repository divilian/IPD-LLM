"""Prisoner's Dilemma action policies."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, Sequence


class Action(StrEnum):
    """An action in the Prisoner's Dilemma."""

    COOPERATE = "C"
    DEFECT = "D"


@dataclass(frozen=True, slots=True)
class Interaction:
    """One past interaction, represented from the actor's perspective."""

    simulation_round: int
    own_action: Action
    opponent_action: Action
    own_payoff: float


History = Sequence[Interaction]


class ActionPolicy(Protocol):
    """Interface for a Prisoner's Dilemma action policy."""

    def choose_action(self, history: History) -> Action:
        ...


class AlwaysCooperate:
    def choose_action(self, history: History) -> Action:
        return Action.COOPERATE


class AlwaysDefect:
    def choose_action(self, history: History) -> Action:
        return Action.DEFECT


class TitForTat:
    def choose_action(self, history: History) -> Action:
        if not history:
            return Action.COOPERATE
        return history[-1].opponent_action


class GrimTrigger:
    def choose_action(self, history: History) -> Action:
        if any(
            interaction.opponent_action == Action.DEFECT
            for interaction in history
        ):
            return Action.DEFECT

        return Action.COOPERATE


class Pavlov:
    def choose_action(self, history: History) -> Action:
        if not history:
            return Action.COOPERATE

        previous = history[-1]
        if previous.own_action == previous.opponent_action:
            return Action.COOPERATE
        return Action.DEFECT
