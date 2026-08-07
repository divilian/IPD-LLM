import pytest

from ipd_llm.policies import (
    Action,
    AlwaysCooperate,
    AlwaysDefect,
    GrimTrigger,
    Interaction,
    Pavlov,
    TitForTat,
)


def interaction(
    own_action: Action,
    opponent_action: Action,
    simulation_round: int = 1,
) -> Interaction:
    return Interaction(
        simulation_round=simulation_round,
        own_action=own_action,
        opponent_action=opponent_action,
        own_payoff=0.0,
    )


@pytest.mark.parametrize("history", [(), (interaction(Action.DEFECT, Action.DEFECT),)])
def test_always_cooperate_always_cooperates(history):
    assert AlwaysCooperate().choose_action(history) == Action.COOPERATE


@pytest.mark.parametrize("history", [(), (interaction(Action.COOPERATE, Action.COOPERATE),)])
def test_always_defect_always_defects(history):
    assert AlwaysDefect().choose_action(history) == Action.DEFECT


def test_tit_for_tat_cooperates_initially():
    assert TitForTat().choose_action(()) == Action.COOPERATE


@pytest.mark.parametrize("opponent_action", [Action.COOPERATE, Action.DEFECT])
def test_tit_for_tat_copies_opponents_previous_action(opponent_action):
    history = (interaction(Action.COOPERATE, opponent_action),)
    assert TitForTat().choose_action(history) == opponent_action


def test_tit_for_tat_uses_most_recent_interaction():
    history = (
        interaction(Action.COOPERATE, Action.DEFECT, simulation_round=1),
        interaction(Action.DEFECT, Action.COOPERATE, simulation_round=2),
    )
    assert TitForTat().choose_action(history) == Action.COOPERATE


def test_grim_trigger_cooperates_initially():
    assert GrimTrigger().choose_action(()) == Action.COOPERATE


def test_grim_trigger_cooperates_while_opponent_has_never_defected():
    history = (
        interaction(Action.COOPERATE, Action.COOPERATE, simulation_round=1),
        interaction(Action.DEFECT, Action.COOPERATE, simulation_round=2),
    )
    assert GrimTrigger().choose_action(history) == Action.COOPERATE


def test_grim_trigger_defects_after_opponent_has_ever_defected():
    history = (
        interaction(Action.COOPERATE, Action.DEFECT, simulation_round=1),
        interaction(Action.DEFECT, Action.COOPERATE, simulation_round=2),
    )
    assert GrimTrigger().choose_action(history) == Action.DEFECT


def test_pavlov_cooperates_initially():
    assert Pavlov().choose_action(()) == Action.COOPERATE


@pytest.mark.parametrize(
    ("own_action", "opponent_action", "expected"),
    [
        (Action.COOPERATE, Action.COOPERATE, Action.COOPERATE),
        (Action.COOPERATE, Action.DEFECT, Action.DEFECT),
        (Action.DEFECT, Action.COOPERATE, Action.DEFECT),
        (Action.DEFECT, Action.DEFECT, Action.COOPERATE),
    ],
)
def test_pavlov_follows_1001_cooperation_vector(
    own_action,
    opponent_action,
    expected,
):
    history = (interaction(own_action, opponent_action),)
    assert Pavlov().choose_action(history) == expected


def test_pavlov_uses_most_recent_interaction():
    history = (
        interaction(Action.COOPERATE, Action.DEFECT, simulation_round=1),
        interaction(Action.DEFECT, Action.DEFECT, simulation_round=2),
    )
    assert Pavlov().choose_action(history) == Action.COOPERATE
