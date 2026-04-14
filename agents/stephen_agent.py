from __future__ import annotations

from mesa import Model
from mesa.discrete_space import Cell

from .factory import register_agent
from .llm_agent import LLMAgent


@register_agent("Stephen")
class StephenAgent(LLMAgent):
    """
    Stephen's more opinionated LLM agent.

    This subclass keeps the same generic LLM-agent machinery as LLMAgent,
    but overrides the prompts to express Stephen's intended strategy:
    - optimize long-run total payoff
    - explicitly reason about future interaction consequences
    - use rewiring as part of that optimization problem
    """

    def __init__(
        self,
        model: Model,
        cell: Cell,
        rewiring_aware: bool = False,
        backend=None,
    ):
        super().__init__(
            model=model,
            cell=cell,
            rewiring_aware=rewiring_aware,
            backend=backend,
        )

    def system_prompt(self) -> str | None:
        return (
            "You are a strategic agent in an iterated prisoner's dilemma simulation. "
            "Your objective is to maximize your total payoff over the full interaction horizon, "
            "not just this round. Use the history of play to infer patterns in the opponent's behavior. "
            "Return a valid answer matching the requested format."
        )

    def build_decision_prompt(
        self,
        other,
        payoff_matrix: dict[tuple[str, str], tuple[str, str]],
    ) -> str:
        prompt = f"""You are playing an iterated prisoner's dilemma for {self.model.num_iter} rounds.

Payoffs to you:
CC -> {payoff_matrix['C', 'C'][0]}
DD -> {payoff_matrix['D', 'D'][0]}
DC -> {payoff_matrix['D', 'C'][0]}
CD -> {payoff_matrix['C', 'D'][0]}

History against this opponent:
{self.serialize_history(self.history, other.unique_id)}

Turns remaining including this one: {self.model.num_iter - self.model.steps + 1}

Choose the move that maximizes your total payoff over the entire game.
"""

        if self.rewiring_aware:
            prompt += (
                "\nAfter this round, you will have the opportunity to sever "
                "connections with current opponents and have them replaced "
                "with new opponents drawn from your friends-of-friends.\n"
                "Take this future rewiring opportunity into account when choosing your move.\n"
            )

        return prompt

    def build_rewiring_prompt(
        self,
        starting_neighbors: set[int],
        new_neighbor_candidates: set[int],
        max_rewires: int,
        give_rationale: bool,
    ) -> str:
        current_neighbor_lines = []
        for node in sorted(starting_neighbors):
            hist = self.serialize_history(self.history, node + 1)
            current_neighbor_lines.append(f"- node={node}\n{hist}")

        current_neighbor_text = (
            "\n".join(current_neighbor_lines) if current_neighbor_lines else "(none)"
        )

        candidate_lines = []
        for node in sorted(new_neighbor_candidates):
            mutual_contacts = sorted(starting_neighbors & self._get_neighbors_of_node(node))
            candidate_lines.append(
                f"- node={node}, mutual_contacts={mutual_contacts}"
            )

        candidate_text = "\n".join(candidate_lines) if candidate_lines else "(none)"

        prompt = f"""You are deciding how to rewire your social network in a networked iterated prisoner's dilemma.

You may sever up to {max_rewires} current neighbors and add up to {max_rewires} new neighbors.

Your goal is to maximize your future total payoff.

Current neighbors and your history against each:
{current_neighbor_text}

Available candidate new neighbors:
{candidate_text}

Choose which current neighbors to drop and which candidate neighbors to add.
Prefer keeping partners who appear likely to support good long-run payoffs,
and prefer adding candidates who seem promising based on the network information available.
"""

        if give_rationale:
            prompt += "\nInclude a brief reason."
        else:
            prompt += "\nDo not include a reason."

        return prompt
