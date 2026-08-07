import networkx as nx
import pytest

from ipd_llm.agents import AgentSpec
from ipd_llm.game import PayoffMatrix
from ipd_llm.model import IPDModel
from ipd_llm.policies import (
    Action,
    AlwaysCooperate,
    AlwaysDefect,
    TitForTat,
)


PAYOFFS = PayoffMatrix(
    temptation=5,
    reward=3,
    punishment=1,
    sucker=0,
)


def spec(action_policy) -> AgentSpec:
    return AgentSpec(action_policy=action_policy)


def two_agent_model(first_policy, second_policy) -> IPDModel:
    graph = nx.Graph()
    graph.add_edge(0, 1)

    return IPDModel(
        graph,
        [spec(first_policy), spec(second_policy)],
        PAYOFFS,
    )


def test_model_creates_one_ipd_agent_per_node():
    model = two_agent_model(
        AlwaysCooperate(),
        AlwaysDefect(),
    )

    assert len(model.agents) == 2
    assert set(model.node_to_agent) == {0, 1}


def test_graph_nodes_and_mesa_ids_are_separate():
    model = two_agent_model(
        AlwaysCooperate(),
        AlwaysDefect(),
    )

    assert model.node_to_agent[0].node == 0
    assert model.node_to_agent[0].unique_id == 1
    assert model.node_to_agent[1].node == 1
    assert model.node_to_agent[1].unique_id == 2


def test_model_builds_reverse_agent_to_node_mapping():
    model = two_agent_model(
        AlwaysCooperate(),
        AlwaysDefect(),
    )

    first = model.node_to_agent[0]
    second = model.node_to_agent[1]

    assert model.agent_to_node[first] == 0
    assert model.agent_to_node[second] == 1


def test_graph_node_count_must_match_agent_spec_count():
    graph = nx.Graph()
    graph.add_nodes_from([0, 1, 2])

    with pytest.raises(ValueError):
        IPDModel(
            graph,
            [
                spec(AlwaysCooperate()),
                spec(AlwaysCooperate()),
            ],
            PAYOFFS,
        )


def test_step_increments_simulation_round():
    model = two_agent_model(
        AlwaysCooperate(),
        AlwaysCooperate(),
    )

    model.step()

    assert model.simulation_round == 1


def test_step_resolves_mutual_cooperation():
    model = two_agent_model(
        AlwaysCooperate(),
        AlwaysCooperate(),
    )

    model.step()

    first = model.node_to_agent[0].history_with(1)[0]
    second = model.node_to_agent[1].history_with(0)[0]

    assert first.own_action == Action.COOPERATE
    assert first.opponent_action == Action.COOPERATE
    assert first.own_payoff == 3
    assert second.own_action == Action.COOPERATE
    assert second.opponent_action == Action.COOPERATE
    assert second.own_payoff == 3


def test_step_resolves_cooperation_against_defection():
    model = two_agent_model(
        AlwaysCooperate(),
        AlwaysDefect(),
    )

    model.step()

    cooperator = model.node_to_agent[0].history_with(1)[0]
    defector = model.node_to_agent[1].history_with(0)[0]

    assert cooperator.own_action == Action.COOPERATE
    assert cooperator.opponent_action == Action.DEFECT
    assert cooperator.own_payoff == 0
    assert defector.own_action == Action.DEFECT
    assert defector.opponent_action == Action.COOPERATE
    assert defector.own_payoff == 5


def test_each_edge_generates_one_interaction_per_round():
    graph = nx.complete_graph(3)
    model = IPDModel(
        graph,
        [
            spec(AlwaysCooperate()),
            spec(AlwaysCooperate()),
            spec(AlwaysCooperate()),
        ],
        PAYOFFS,
    )

    model.step()

    assert len(model.node_to_agent[0].history_with(1)) == 1
    assert len(model.node_to_agent[0].history_with(2)) == 1
    assert len(model.node_to_agent[1].history_with(0)) == 1
    assert len(model.node_to_agent[1].history_with(2)) == 1
    assert len(model.node_to_agent[2].history_with(0)) == 1
    assert len(model.node_to_agent[2].history_with(1)) == 1


def test_tit_for_tat_uses_previous_round_not_current_round():
    model = two_agent_model(
        TitForTat(),
        AlwaysDefect(),
    )

    model.step()
    model.step()

    history = model.node_to_agent[0].history_with(1)

    assert history[0].own_action == Action.COOPERATE
    assert history[0].opponent_action == Action.DEFECT
    assert history[1].own_action == Action.DEFECT
    assert history[1].opponent_action == Action.DEFECT


def test_fixed_network_does_not_change_during_steps():
    model = two_agent_model(
        AlwaysCooperate(),
        AlwaysDefect(),
    )
    initial_edges = set(model.graph.edges)

    model.step()
    model.step()

    assert set(model.graph.edges) == initial_edges


def test_model_handles_watts_strogatz_graph():
    graph = nx.watts_strogatz_graph(
        n=12,
        k=4,
        p=0.25,
        seed=12345,
    )
    model = IPDModel(
        graph,
        [
            spec(AlwaysCooperate())
            for _ in graph.nodes
        ],
        PAYOFFS,
    )

    model.step()

    for first_node, second_node in graph.edges:
        first_history = model.node_to_agent[first_node].history_with(
            second_node
        )
        second_history = model.node_to_agent[second_node].history_with(
            first_node
        )

        assert len(first_history) == 1
        assert len(second_history) == 1
