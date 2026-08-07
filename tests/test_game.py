import pytest

from ipd_llm.game import PayoffMatrix, resolve_interaction
from ipd_llm.policies import Action


@pytest.fixture
def matrix() -> PayoffMatrix:
    return PayoffMatrix(
        temptation=5,
        reward=3,
        punishment=1,
        sucker=0,
    )


@pytest.mark.parametrize(
    ("first_action", "second_action", "expected"),
    [
        (Action.COOPERATE, Action.COOPERATE, (3, 3)),
        (Action.COOPERATE, Action.DEFECT, (0, 5)),
        (Action.DEFECT, Action.COOPERATE, (5, 0)),
        (Action.DEFECT, Action.DEFECT, (1, 1)),
    ],
)
def test_resolve_interaction(
    matrix,
    first_action,
    second_action,
    expected,
):
    assert resolve_interaction(
        first_action,
        second_action,
        matrix,
    ) == expected


@pytest.mark.parametrize(
    ("temptation", "reward", "punishment", "sucker"),
    [
        (3, 3, 1, 0),
        (5, 3, 3, 0),
        (5, 3, 1, 1),
        (3, 4, 1, 0),
    ],
)
def test_payoff_matrix_requires_strict_ordering(
    temptation,
    reward,
    punishment,
    sucker,
):
    with pytest.raises(ValueError, match="T > R > P > S"):
        PayoffMatrix(
            temptation=temptation,
            reward=reward,
            punishment=punishment,
            sucker=sucker,
        )


def test_payoff_matrix_requires_mutual_cooperation_constraint():
    with pytest.raises(ValueError, match=r"2R > T \+ S"):
        PayoffMatrix(
            temptation=6,
            reward=3,
            punishment=1,
            sucker=0,
        )
