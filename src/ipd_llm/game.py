"""Prisoner's Dilemma payoff mechanics."""

from dataclasses import dataclass

from ipd_llm.policies import Action


@dataclass(frozen=True, slots=True)
class PayoffMatrix:
    """Payoffs for temptation, reward, punishment, and sucker outcomes."""

    temptation: float
    reward: float
    punishment: float
    sucker: float

    def __post_init__(self) -> None:
        if not (
            self.temptation
            > self.reward
            > self.punishment
            > self.sucker
        ):
            raise ValueError("Payoffs must satisfy T > R > P > S.")

        if 2 * self.reward <= self.temptation + self.sucker:
            raise ValueError("Payoffs must satisfy 2R > T + S.")


def resolve_interaction(
    first_action: Action,
    second_action: Action,
    matrix: PayoffMatrix,
) -> tuple[float, float]:
    """Return payoffs for two simultaneous Prisoner's Dilemma actions."""

    if first_action == Action.COOPERATE:
        if second_action == Action.COOPERATE:
            return matrix.reward, matrix.reward
        return matrix.sucker, matrix.temptation

    if second_action == Action.COOPERATE:
        return matrix.temptation, matrix.sucker
    return matrix.punishment, matrix.punishment
