from __future__ import annotations

from mesa import Model
from mesa.discrete_space import Cell

from .factory import register_agent
from .llm_agent import LLMAgent


@register_agent("StudentLLMTemplate")
class StudentLLMTemplate(LLMAgent):
    """
    Starting point for writing an LLM-based IPD agent.

    (You should edit this file, not llm_agent.py.)

    The base class LLMAgent already handles:
    - calling the LLM backend
    - requesting structured JSON output
    - logging decisions
    - rewiring compatibility with the simulator

    Your job is to design the agent's strategy by changing:
    - system_prompt()
    - build_decision_prompt()
    - optionally build_rewiring_prompt()

    A good workflow is:
    1. Start by editing build_decision_prompt().
    2. Run experiments and observe behavior.
    3. If your simulation includes rewiring, then edit build_rewiring_prompt().
    4. If you want a stronger overall "persona" or objective, edit system_prompt().
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
        """
        Optional: give the LLM a stable overall role or objective.

        Keep this fairly short. Put most of your strategy-specific instructions
        in build_decision_prompt() and build_rewiring_prompt().
        """
        return (
            "You are an agent in an iterated prisoner's dilemma simulation. "
            "Read the game state carefully and return a valid answer matching the requested format."
        )

    def build_decision_prompt(
        self,
        other,
        payoff_matrix: dict[tuple[str, str], tuple[str, str]],
    ) -> str:
        """
        Main place to experiment with your strategy.

        Questions you might think about:
        - Should your agent care most about the next round, or the long run?
        - Should it reward cooperation?
        - Should it punish defections?
        - Should it forgive sometimes?
        - How should it use the number of remaining turns?
        """
        prompt = f"""You are playing an iterated prisoner's dilemma.

Payoffs to you:
CC -> {payoff_matrix['C', 'C'][0]}
DD -> {payoff_matrix['D', 'D'][0]}
DC -> {payoff_matrix['D', 'C'][0]}
CD -> {payoff_matrix['C', 'D'][0]}

History against this opponent:
{self.serialize_history(self.history, other.unique_id)}

Turns remaining including this one: {self.model.num_iter - self.model.steps + 1}

Choose your next move.
"""

        if self.rewiring_aware:
            prompt += (
                "\nAfter this round, you may have an opportunity to sever connections "
                "with current opponents and have them replaced with new opponents drawn "
                "from your friends-of-friends.\n"
            )

        return prompt

    def build_rewiring_prompt(
        self,
        starting_neighbors: set[int],
        new_neighbor_candidates: set[int],
        max_rewires: int,
        give_rationale: bool,
    ) -> str:
        """
        Optional: customize how your agent thinks about rewiring.

        If you do not want to change rewiring behavior yet, you can leave this
        method alone and just focus on build_decision_prompt() first.
        """
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

Current neighbors and your history against each:
{current_neighbor_text}

Available candidate new neighbors:
{candidate_text}

Choose which current neighbors to drop and which candidate neighbors to add.
"""

        if give_rationale:
            prompt += "\nInclude a brief reason."
        else:
            prompt += "\nDo not include a reason."

        return prompt
