from __future__ import annotations

from mesa import Model
from mesa.discrete_space import Cell

from agents.factory import register_agent
from agents.llm_agent import LLMAgent


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
    - optionally inform_foaf()

    A good workflow is:
    1. Start by editing build_decision_prompt().
    2. Run experiments and observe behavior.
    3. If your simulation includes rewiring, then edit build_rewiring_prompt().
    4. If you want to lie or withhold information to others, then edit
        inform_foaf().
    5. If you want a stronger overall "persona," edit system_prompt().
    """

    def __init__(
        self,
        model: Model,
        cell: Cell,
        backend=None,
    ):
        super().__init__(
            model=model,
            cell=cell,
            backend=backend,
        )

    def system_prompt(self) -> str | None:
        """
        Keep this fairly short. Put most of your strategy-specific instructions
        in build_decision_prompt() and build_rewiring_prompt().
        """
        raise NotImplementedError

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
        raise NotImplementedError

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
        if give_rationale:
            return """
                Output exactly this JSON document:
                {'drop_ids':[], 'add_ids':[], 'reason':"I felt like it."}
            """
        else:
            return """
                Output exactly this JSON document:
                {'drop_ids':[], 'add_ids':[]}
            """

    def inform_foaf(
        self,
        inquirer: "IPDAgent",
    ) -> dict[int, list[dict[str, int | str]]] | None:
        """
        Return a reported interaction history keyed by opponent node number.
        Each value is a list of dicts with keys 'step' (the round number),
        'self_move', and 'other_move'.

        Such a history is trivially obtainable simply by accessing your own
        self.history. However, note that it is perfectly permissible -- and
        perhaps advantageous -- to lie about this history instead. If you do
        so, you will incur a cost. You can also return None to give the
        inquirer the hand. This also incurs a cost.
        """
        return self.history
