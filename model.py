from typing import List, Tuple
import logging
from collections import Counter
import asyncio

from mesa import Model, DataCollector
from mesa.discrete_space import Network
import networkx as nx
import numpy as np

from agents.factory import AgentFactory
from agents.llm_agent import LLMAgent
from graph.sbm import compute_sbm_probs
from llm.prompt_builder import build_batch_prompt
from llm.backend import LLMBackend
from llm.ollama_backend import OllamaBackend
from llm.mock_backend import MockBackend


class IPDModel(Model):
    """
    Iterated Prisoner's Dilemma on a static graph, Mesa 3.x style.
    """

    def __init__(
        self,
        N,  # num agents
        avg_degree: float,
        payoff_matrix: List[Tuple],
        p_same: float,
        p_diff: float,
        num_iter: int,
        agent_factory: AgentFactory,
        llm_backend: LLMBackend,
        seed: int,
    ):
        super().__init__(seed=seed)

        self.N = N
        self.seed = seed
        self.payoff_matrix = payoff_matrix
        self.num_iter = num_iter
        self.llm_backend = llm_backend

        # Distinct specs = SBM blocks (stable order)
        specs = sorted(
            agent_factory.probs,
            key=lambda spec: (spec[0].__name__, spec[1]),
        )
        sizes = [int(round(agent_factory.probs[c] * N)) for c in specs]

        # Build SBM graph (nodes grouped by block).
        num_agent_types = len(specs)
        p = p_diff * np.ones((num_agent_types, num_agent_types))
        for i in range(num_agent_types):
            p[i][i] = p_same
        graph = nx.stochastic_block_model(sizes, p, seed=self.seed)
        self.network = Network(graph, random=self.random)

        cells = self.network.all_cells.cells
        idx = 0
        self.node_to_agent = {}
        for (agent_cls, kwargs_items), sz in zip(specs, sizes):
            init_kwargs = dict(kwargs_items)
            for _ in range(sz):
                cell = cells[idx]
                # Actually instantiate the agent.
                agent = agent_cls(self, cell, **init_kwargs)
                self.node_to_agent[cell.coordinate] = agent
                idx += 1
        self.agent_to_node = {
            agent: node
            for node, agent in self.node_to_agent.items()
        }

        self.datacollector = DataCollector(
            model_reporters={
                "avg_payoff":
                    lambda m: sum(a.wealth for a in m.agents) / len(m.agents),
                "coop_rate": self._coop_rate,
                "avg_degree":
                    lambda m: (
                        sum(dict(m.network.G.degree()).values())
                        / m.network.G.number_of_nodes()
                    )
                    if m.network.G.number_of_nodes()
                    else 0.0,
            }
        )

        # Peace of mind that node IDs and agent IDs haven't drifted weirdly.
        assert set(self.node_to_agent.keys()) == set(self.network.G.nodes)
        assert all(a.node == n for n, a in self.node_to_agent.items())

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
            {node: agent.__class__.__name__
             for node, agent in self.node_to_agent.items()},
            name="agent_type",
        )
        return nx.attribute_assortativity_coefficient(
            self.network.G,
            "agent_type",
        )

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
        asyncio.run(self._step_async())

    async def _step_async(self):

        # Reset agent state for this new round.
        for a in self.agents:
            a.current_iter_payment = 0
            a.current_decisions.clear()

        # Who are our LLM agents? They're the time-consuming ones.
        llm_agents = [a for a in self.agents if isinstance(a, LLMAgent)]

        if llm_agents and self.llm_backend is not None:

            # Build the "payloads" for each agent; that is, what will be
            # inserted into the batch prompt for the LLM to make decisions
            # about.
            payloads = [a.decision_context() for a in llm_agents]

            # Given all the necessary LLM agent state info, build the prompt.
            prompt = build_batch_prompt(payloads, self.payoff_matrix)

            # Twiddle our thumbs until the LLM gives us decisions on all of the
            # agents.
            response = await self.llm_backend.batch_decide(prompt)

            # Record the LLM's decisions in each agent.
            self._apply_llm_decisions(response['decisions'])

        # Okay, now we can run the fast-moving guys.
        self._run_rule_agents()

        # Actually "commit" the moves by propagating to agent inst vars (wealth
        # and history).
        self._resolve_payoffs()

        # Stuff happened, in case you care.
        self.datacollector.collect(self)

    def _apply_llm_decisions(self, decisions_for_round):
        """
        Unpack this list:
                [
                    {"id": 0, "opponent": 3, "move": "C"},
                    {"id": 0, "opponent": 1, "move": "D"},
                    {"id": 1, "opponent": 2, "move": "D"},
                    ...
                ]
        and add all that information to the agents' current_decisions dicts.
        """
        for d in decisions_for_round:
            aid = int(d["id"])
            oid = int(d["opponent"])
            move = d["move"]
            if move not in ("C", "D"):
                raise ValueError(f"Invalid move from LLM: {move}")
            if aid not in self.node_to_agent:
                raise ValueError(f"Unknown agent id from LLM! {aid}")
            if oid not in self.node_to_agent:
                raise ValueError(f"Unknown opponent id from LLM! {oid}")
            if oid not in set(self.network.G.neighbors(aid)):
                raise ValueError(
                    f"LLM gave non-neighbor opponent {oid} for agent {aid}!"
                )
            self.node_to_agent[aid].current_decisions[oid] = move

    def _run_rule_agents(self) -> None:
        rule_agents = [a for a in self.agents if not isinstance(a, LLMAgent)]
        for ra in rule_agents:
            for n in self.network.G.neighbors(ra.node):
                ra.current_decisions[n], _ = ra.decide_against(
                    self.node_to_agent[n],
                    self.payoff_matrix,
                )

    def _resolve_payoffs(self) -> None:
        """
        Resolve and commit all the information in the agents' current_decisions
        dicts. For each one, determine the output of that round of the game
        against the agent's opponent, award winnings, and record history.
        """
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

        # Award the winnings, but scale by the node's degree. This is to avoid
        # disadvantaging agents with fewer neighbors who of course therefore
        # play fewer rounds and hence have lower winnings.
        for a in self.agents:
            k = self.network.G.degree[a.node]
            if k > 0:
                a.wealth += a.current_iter_payment / k

    def estimate_expected_avg_wealth(self):
        """
        Completely back-of-the-envelope estimate of "about how much should each
        agent expect to win during this situation?" The crude formula assumes
        an independent 50/50 chance of choosing to defect or cooperate. Note
        that we compute "per_iter" here (not "per_encounter") because we're
        discounting each agent's per-iteration winnings by its degree (which
        is its number of encounters).
        """
        avg_agent_per_iter = 0.25 * (
            self.payoff_matrix[("C", "C")][0]
            + self.payoff_matrix[("D", "C")][0]
            + self.payoff_matrix[("C", "D")][0]
            + self.payoff_matrix[("D", "D")][0]
        )
        return avg_agent_per_iter * self.num_iter

    def agent_mix(self) -> dict[str, int]:
        """
        Return counts of agents by concrete subclass name.
        """
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
