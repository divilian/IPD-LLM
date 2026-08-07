from mesa import Agent, Model

from ipd_llm.agents import AgentSpec, IPDAgent
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


def make_agent(
    model: Model,
    node: int = 0,
    policy=None,
) -> IPDAgent:
    if policy is None:
        policy = AlwaysCooperate()

    return IPDAgent(
        model,
        node,
        AgentSpec(action_policy=policy),
    )


def test_agent_inherits_from_mesa_agent():
    model = Model()
    agent = make_agent(model)

    assert isinstance(agent, Agent)


def test_mesa_assigns_integer_unique_ids():
    model = Model()
    first = make_agent(model, node=0)
    second = make_agent(model, node=1)

    assert first.unique_id == 1
    assert second.unique_id == 2


def test_agent_stores_graph_node_separately_from_unique_id():
    model = Model()
    agent = make_agent(model, node=0)

    assert agent.node == 0
    assert agent.unique_id == 1


def test_agent_uses_neutral_identifier():
    model = Model()
    agent = make_agent(model)

    assert agent.identifier == "A01"


def test_mesa_registers_agent_with_model():
    model = Model()
    agent = make_agent(model)

    assert agent in model.agents


def test_agent_starts_with_empty_history():
    model = Model()
    agent = make_agent(model)

    assert agent.history_with(1) == ()


def test_agent_records_history_by_opponent_node():
    model = Model()
    agent = make_agent(model)
    versus_1 = interaction(
        Action.COOPERATE,
        Action.DEFECT,
        simulation_round=1,
    )
    versus_2 = interaction(
        Action.COOPERATE,
        Action.COOPERATE,
        simulation_round=1,
    )

    agent.record_interaction(1, versus_1)
    agent.record_interaction(2, versus_2)

    assert agent.history_with(1) == (versus_1,)
    assert agent.history_with(2) == (versus_2,)


def test_agent_preserves_interaction_order():
    model = Model()
    agent = make_agent(model)
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

    agent.record_interaction(1, first)
    agent.record_interaction(1, second)

    assert agent.history_with(1) == (first, second)


def test_history_view_does_not_expose_mutable_internal_list():
    model = Model()
    agent = make_agent(model)
    first = interaction(
        Action.COOPERATE,
        Action.COOPERATE,
        simulation_round=1,
    )

    agent.record_interaction(1, first)
    history = agent.history_with(1)

    assert isinstance(history, tuple)


def test_agent_uses_history_for_requested_opponent_only():
    model = Model()
    agent = make_agent(model, policy=TitForTat())
    versus_1 = interaction(
        Action.COOPERATE,
        Action.DEFECT,
        simulation_round=1,
    )
    versus_2 = interaction(
        Action.COOPERATE,
        Action.COOPERATE,
        simulation_round=1,
    )

    agent.record_interaction(1, versus_1)
    agent.record_interaction(2, versus_2)

    assert agent.choose_action(1) == Action.DEFECT
    assert agent.choose_action(2) == Action.COOPERATE


def test_agent_policy_receives_empty_history_for_new_opponent():
    model = Model()
    agent = make_agent(model, policy=TitForTat())

    assert agent.choose_action(99) == Action.COOPERATE
