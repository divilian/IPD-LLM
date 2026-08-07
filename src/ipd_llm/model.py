"""Mesa model for networked Iterated Prisoner's Dilemma."""

from collections.abc import Sequence

import networkx as nx
from mesa import Model

from ipd_llm.agents import AgentSpec, IPDAgent
from ipd_llm.records import InteractionRecord
from ipd_llm.game import PayoffMatrix, resolve_interaction
from ipd_llm.initialization import Initialization
from ipd_llm.policies import Action, Interaction


class IPDModel(Model):
    """A fixed-network Iterated Prisoner's Dilemma model."""

    def __init__(
        self,
        graph: nx.Graph,
        agent_specs: Sequence[AgentSpec],
        payoff_matrix: PayoffMatrix,
    ) -> None:
        super().__init__()

        if graph.number_of_nodes() != len(agent_specs):
            raise ValueError(
                "Graph node count must match agent spec count."
            )

        self.graph = graph.copy()
        self.payoff_matrix = payoff_matrix
        self.simulation_round = 0
        self.records: list[InteractionRecord] = []
        self.node_to_agent: dict[int, IPDAgent] = {}
        self.agent_to_node: dict[IPDAgent, int] = {}

        for node, spec in zip(
            sorted(self.graph.nodes),
            agent_specs,
        ):
            agent = IPDAgent(self, node, spec)
            self.node_to_agent[node] = agent
            self.agent_to_node[agent] = node

    @classmethod
    def from_initialization(
        cls,
        initialization: Initialization,
        payoff_matrix: PayoffMatrix,
    ) -> "IPDModel":
        """Build a model from one saved initialization."""

        nodes = sorted(initialization.graph.nodes)
        agent_specs = [
            initialization.node_to_spec[node]
            for node in nodes
        ]

        return cls(
            initialization.graph,
            agent_specs,
            payoff_matrix,
        )

    def step(self) -> None:
        """Run one fixed-network simulation round."""

        self.simulation_round += 1
        actions = self._collect_actions()
        self._resolve_interactions(actions)

    def _collect_actions(self) -> dict[tuple[int, int], Action]:
        """Collect every action before resolving any interaction."""

        actions: dict[tuple[int, int], Action] = {}

        for first_node, second_node in self.graph.edges:
            first = self.node_to_agent[first_node]
            second = self.node_to_agent[second_node]
            actions[first_node, second_node] = first.choose_action(
                second_node
            )
            actions[second_node, first_node] = second.choose_action(
                first_node
            )

        return actions

    def _resolve_interactions(
        self,
        actions: dict[tuple[int, int], Action],
    ) -> None:
        """Resolve every edge using the already-collected actions."""

        for first_node, second_node in self.graph.edges:
            first_action = actions[first_node, second_node]
            second_action = actions[second_node, first_node]
            first_payoff, second_payoff = resolve_interaction(
                first_action,
                second_action,
                self.payoff_matrix,
            )

            first = self.node_to_agent[first_node]
            second = self.node_to_agent[second_node]

            first.record_interaction(
                second_node,
                Interaction(
                    simulation_round=self.simulation_round,
                    own_action=first_action,
                    opponent_action=second_action,
                    own_payoff=first_payoff,
                ),
            )
            second.record_interaction(
                first_node,
                Interaction(
                    simulation_round=self.simulation_round,
                    own_action=second_action,
                    opponent_action=first_action,
                    own_payoff=second_payoff,
                ),
            )

            self.records.append(
                InteractionRecord(
                    simulation_round=self.simulation_round,
                    first_node=first_node,
                    second_node=second_node,
                    first_action=first_action,
                    second_action=second_action,
                    first_payoff=first_payoff,
                    second_payoff=second_payoff,
                )
            )
