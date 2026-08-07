from mesa import Agent, Model

from ipd_llm.agents import IPDAgent
from ipd_llm.policies import (
    Action,
    AlwaysCooperate,
    Interaction,
    TitForTat,
)


def interaction(
    own_action: Action,
    opponent_action: Action,
    simulation_round: int,
) -> Interaction:
    return Interaction(
        simulation_round=simulation_round,
        own_action=own_action,
        opponent_action=opponent_action,
        own_payoff=0.0,
    )


def test_agent_inherits_from_mesa_agent():
    model = Model()
    agent = IPDAgent(model, AlwaysCooperate())

    assert isinstance(agent, Agent)


def test_mesa_assigns_integer_unique_ids():
    model = Model()
    first = IPDAgent(model, AlwaysCooperate())
    second = IPDAgent(model, AlwaysCooperate())

    assert first.unique_id == 1
    assert second.unique_id == 2


def test_agent_uses_neutral_identifier():
    model = Model()
    agent = IPDAgent(model, AlwaysCooperate())

    assert agent.identifier == "A01"


def test_mesa_registers_agent_with_model():
    model = Model()
    agent = IPDAgent(model, AlwaysCooperate())

    assert agent in model.agents


def test_agent_starts_with_empty_history():
    model = Model()
    agent = IPDAgent(model, AlwaysCooperate())

    assert agent.history_with(2) == ()


def test_agent_records_history_by_opponent():
    model = Model()
    agent = IPDAgent(model, AlwaysCooperate())
    versus_2 = interaction(
        Action.COOPERATE,
        Action.DEFECT,
        simulation_round=1,
    )
    versus_3 = interaction(
        Action.COOPERATE,
        Action.COOPERATE,
        simulation_round=1,
    )

    agent.record_interaction(2, versus_2)
    agent.record_interaction(3, versus_3)

    assert agent.history_with(2) == (versus_2,)
    assert agent.history_with(3) == (versus_3,)


def test_agent_preserves_interaction_order():
    model = Model()
    agent = IPDAgent(model, AlwaysCooperate())
    first = interaction(
        Action.COOPERATE,
        Action.COOPERATE,
        simulation_round=1,
    )
    second = interaction(
        Action.COOPERATE,
        Action.DEFECT,
        simulation_round=2,
    )

    agent.record_interaction(2, first)
    agent.record_interaction(2, second)

    assert agent.history_with(2) == (first, second)


def test_history_view_does_not_expose_mutable_internal_list():
    model = Model()
    agent = IPDAgent(model, AlwaysCooperate())
    first = interaction(
        Action.COOPERATE,
        Action.COOPERATE,
        simulation_round=1,
    )

    agent.record_interaction(2, first)
    history = agent.history_with(2)

    assert isinstance(history, tuple)


def test_agent_uses_history_for_requested_opponent_only():
    model = Model()
    agent = IPDAgent(model, TitForTat())
    versus_2 = interaction(
        Action.COOPERATE,
        Action.DEFECT,
        simulation_round=1,
    )
    versus_3 = interaction(
        Action.COOPERATE,
        Action.COOPERATE,
        simulation_round=1,
    )

    agent.record_interaction(2, versus_2)
    agent.record_interaction(3, versus_3)

    assert agent.choose_action(2) == Action.DEFECT
    assert agent.choose_action(3) == Action.COOPERATE


def test_agent_policy_receives_empty_history_for_new_opponent():
    model = Model()
    agent = IPDAgent(model, TitForTat())

    assert agent.choose_action(99) == Action.COOPERATE
