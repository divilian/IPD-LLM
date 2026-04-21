from typing import List, Tuple
from collections import Counter

from mesa import Model, DataCollector
from mesa.discrete_space import Network
import networkx as nx
import numpy as np

from agents.base import IPDAgent
from agents.factory import AgentFactory
from llm.ollama_backend import OllamaBackend


class IPDModel(Model):
    """
    Iterated Prisoner's Dilemma on a static graph, Mesa 3.x style.
    """

    def __init__(
        self,
        N,
        avg_degree: float,
        payoff_matrix: List[Tuple],
        p_same: float,
        p_diff: float,
        num_iter: int,
        agent_factory: AgentFactory,
        extra_agent_classes: list[type[IPDAgent]] | None,
        max_rewires: int,
        give_rationales: bool,
        llm_backend: OllamaBackend,
        ollama_model: str,
        llm_out_file: str,
        debug: bool,
        seed: int,
    ):
        super().__init__(rng=seed)

        self.N = N
        self.seed = seed
        self.payoff_matrix = payoff_matrix
        self.num_iter = num_iter
        self.max_rewires = max_rewires
        self.give_rationales = give_rationales
        self.llm_backend = llm_backend
        self.ollama_model = ollama_model
        self.llm_out_file = llm_out_file
        self.debug = debug
        self.extra_agent_classes = list(extra_agent_classes or [])
        self.total_agents = self.N + len(self.extra_agent_classes)

        base_plan = agent_factory.plan_agent_specs(self.N, self.random)
        extra_specs = [(agent_cls, {}) for agent_cls in self.extra_agent_classes]
        all_specs = base_plan + extra_specs

        if not all_specs:
            raise ValueError("Simulation must contain at least one agent")

        labels = [self._agent_block_label(agent_cls) for agent_cls, _ in all_specs]
        blocks: list[tuple[str, list[tuple[type, dict]]]] = []
        for label in labels:
            if not blocks or blocks[-1][0] != label:
                blocks.append((label, []))
        block_lists = {id(items): items for _, items in blocks}
        block_index = 0
        for spec in all_specs:
            label = self._agent_block_label(spec[0])
            while blocks[block_index][0] != label:
                block_index += 1
            block_lists[id(blocks[block_index][1])].append(spec)
        grouped_specs = [items for _, items in blocks]

        sizes = [len(group) for group in grouped_specs]

        num_agent_types = len(grouped_specs)
        p = p_diff * np.ones((num_agent_types, num_agent_types))
        for i in range(num_agent_types):
            p[i][i] = p_same
        graph = nx.stochastic_block_model(sizes, p, seed=self.seed)
        self.network = Network(graph, random=self.random)

        cells = list(self.network.all_cells.cells)
        self.node_to_agent = {}
        self.custom_player_nodes: set[int] = set()
        idx = 0
        for group in grouped_specs:
            for agent_cls, init_kwargs in group:
                cell = cells[idx]
                agent = agent_cls(self, cell, **init_kwargs)
                if agent_cls in self.extra_agent_classes:
                    setattr(agent, "render_as_player_oval", True)
                    self.custom_player_nodes.add(cell.coordinate)
                self.node_to_agent[cell.coordinate] = agent
                idx += 1

        self.agent_to_node = {
            agent: node
            for node, agent in self.node_to_agent.items()
        }

        self.datacollector = DataCollector(
            model_reporters={
                "avg_payoff": lambda m: sum(a.wealth for a in m.agents) / len(m.agents),
                "coop_rate": self._coop_rate,
                "avg_degree": lambda m: (
                    sum(dict(m.network.G.degree()).values())
                    / m.network.G.number_of_nodes()
                    if m.network.G.number_of_nodes()
                    else 0.0
                ),
            }
        )

        assert set(self.node_to_agent.keys()) == set(self.network.G.nodes)
        assert all(a.node == n for n, a in self.node_to_agent.items())

        if self.llm_backend:
            self.llm_backend.ensure_ollama_running()

    def _agent_block_label(self, agent_cls: type[IPDAgent]) -> str:
        return agent_cls.__name__

    def is_custom_player_node(self, node: int) -> bool:
        return node in self.custom_player_nodes

    def assortativity(self) -> float:
        """
        Return the graph's assortativity by agent type. A value of 0 means
        agents are indifferent to what type they connected to. A positive value
        means they're more likely to connect to the same type (a TitForTat to
        other TitForTats, e.g.). A negative value means they're more likely to
        connect to different types (a TitForTat to Means and Suckers, e.g.)
        """
        nx.set_node_attributes(
            self.network.G,
            {node: agent.__class__.__name__ for node, agent in self.node_to_agent.items()},
            name="agent_type",
        )
        return nx.attribute_assortativity_coefficient(
            self.network.G,
            "agent_type",
        )

    def request_foaf_info_from(self, agent, neighbor_node):
        """
        Agents call this method when they want to ask neighbors for FOAF info.
        If an agent asks for info from a neighbor that is not their FOAF, die.
        """
        if neighbor_node not in self.network.G.neighbors(agent.node):
            raise ValueError
        answer = self.node_to_agent[neighbor_node].inform_foaf(agent.node)
        # Depending on what they did, levy appropriate charges.
        return answer
        
    def _coop_rate(self) -> float:
        total = 0
        coop = 0
        for a in self.agents:
            for action in a.current_decisions.values():
                total += 1
                if action == "C":
                    coop += 1
        return coop / total if total else 0.0


    def step(self) -> None:
        # Reset agent state for this new round.
        for a in self.agents:
            a.current_iter_payment = 0
            a.current_decisions.clear()

        if self.debug:
            print(
                f"========== Agents running in iter {self.steps} "
                f"of {self.num_iter} =========="
            )
        self._run_agents()

        # Actually "commit" the moves by propagating to agent inst vars (wealth
        # and history).
        self._resolve_payoffs()

        # Permit all agents to sever and form connections as they desire.
        self._permit_rewiring()

        # Stuff happened, in case you care.
        self.datacollector.collect(self)

        if self.debug:
            print(
                f"========== Agents running in iter {self.steps} "
                f"of {self.num_iter} =========="
            )

    def _run_agents(self) -> None:
        for a in self.agents:
            for n in self.network.G.neighbors(a.node):
                output = a.decide_against(
                    n,
                    give_rationale=self.give_rationales
                )
                a.current_decisions[n], _ = output

    def _resolve_payoffs(self) -> None:
        processed = set()

        for a in self.agents:
            i = a.node
            for j in self.network.G.neighbors(i):
                if (j, i) in processed:
                    continue
                b = self.node_to_agent[j]
                move_i = a.current_decisions[j]
                move_j = b.current_decisions[i]
                payoff_i, payoff_j = self.payoff_matrix[(move_i, move_j)]
                a.current_iter_payment += payoff_i
                b.current_iter_payment += payoff_j
                a.record_interaction(j, move_i, move_j)
                b.record_interaction(i, move_j, move_i)
                processed.add((i, j))

        for a in self.agents:
            k = self.network.G.degree[a.node]
            if k > 0:
                a.wealth += a.current_iter_payment / k

    def _permit_rewiring(self) -> None:

        snapshot = self.network.G.copy()

        requests_by_agent = {
            a: a.request_rewire(self.max_rewires)
            for a in self.agents 
        }

        severs = set()
        adds = set()

        for agent, r in requests_by_agent.items():
            sever_nodes = tuple(sorted(r["nodes_to_sever"]))
            add_nodes = tuple(sorted(r["nodes_to_add"]))

            # this takes the first n in case lengths dont match
            n = min(len(sever_nodes), len(add_nodes), self.max_rewires)
            sever_nodes = sever_nodes[:n]
            add_nodes = add_nodes[:n]
 
            for s in sever_nodes:
                if snapshot.has_edge(agent.node, s):
                    severs.add((agent.node, s))
 
            for a in add_nodes:
                if self.is_foaf(snapshot, agent.node, a):
                    adds.add((agent.node, a))
 
        adds -= severs
 
        for u, v in severs:
            if self.network.G.has_edge(u, v):
                self.network.G.remove_edge(u, v)
 
        for u, v in adds:
            if not self.network.G.has_edge(u, v):
                self.network.G.add_edge(u, v)

        for sever in severs:
            cell1 = self.network[sever[0]]
            cell2 = self.network[sever[1]]
            cell1.disconnect(cell2)

        for add in adds:
            cell1 = self.network[sever[0]]
            cell2 = self.network[sever[1]]
            cell1.connect(cell2)

    def is_foaf(self, G, u, v):
        if u == v:
            return False
        if u not in G or v not in G:
            return False
        if G.has_edge(u, v):
            return False
        return any(True for _ in nx.common_neighbors(G, u, v))

    def estimate_expected_avg_wealth(self):
        avg_agent_per_iter = 0.25 * (
            self.payoff_matrix[("C", "C")][0]
            + self.payoff_matrix[("D", "C")][0]
            + self.payoff_matrix[("C", "D")][0]
            + self.payoff_matrix[("D", "D")][0]
        )
        return avg_agent_per_iter * self.num_iter

    def agent_mix(self) -> dict[str, int]:
        return dict(
            Counter(agent.__class__.__name__ for agent in self.agents)
        )

    def __str__(self):
        ret_val = "with this agent mix:\n"
        am = [f"{c:>18}:{n:>5}" for c, n in self.agent_mix().items()]
        ret_val += "\n".join(am)
        ret_val += f"\nThe graph has {self.network.G.size()} edges "
        ret_val += f"and assortativity {self.assortativity():.3f}."
        return ret_val
