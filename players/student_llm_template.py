from __future__ import annotations

from mesa import Model
from mesa.discrete_space import Cell

from agents.factory import register_agent
from agents.llm_agent import LLMAgent


@register_agent("StudentLLMTemplate")
class StudentLLMTemplate(LLMAgent):
    """
    Starting point for writing an LLM-based IPD agent.

    (You should copy and edit this file, not llm_agent.py.)

    The base class LLMAgent already handles:
    - calling the LLM backend
    - requesting structured JSON output
    - rewiring compatibility with the simulator

    Your job is to design the agent's strategy by changing:
    - system_prompt()
    - build_decision_prompt()
    - (optionally) build_rewiring_prompt()
    - (optionally) inform_foaf()

    A good workflow is:
    1. Start by editing build_decision_prompt().
    2. Run experiments and observe behavior.
    3. If your strategy includes rewiring, then edit build_rewiring_prompt().
    4. If you want to lie to, or withhold information from, others, then edit
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
        return ""

    def build_decision_prompt(
        self,
        other,
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
        return """
            Output exactly this JSON document:
            {'move':'C'}
        """

    def build_rewiring_prompt(
        self,
        max_rewires: int,
    ) -> str:
        """
        Optional: customize how your agent thinks about rewiring.

        If you do not want to change rewiring behavior yet, you can leave this
        method alone and just focus on build_decision_prompt() first.
        """
        return """
            Output exactly this JSON document:
            {'nodes_to_drop':[], 'nodes_to_add':[]}
        """

    def inform_foaf(
        self,
        inquirer: int,
    ) -> dict[int, list[dict[str, int | str]] | None]:
        """
        This method is called when an agent is requesting information from you. 
        The integer parameter is the node number of the requesting agent.
        You have some choices here:

        1. The default option is for you to dutifully return a reported
        interaction history keyed by opponent node number. Each value is a
        list of dicts with keys 'step' (the round number), 'self_move', and
        'other_move'. The opponent node number keys must include all your
        current neighbors. See below for an example legal return value.

        If you choose this option, you will receive $1. (The Delaney variant.)

        2. You may refuse to provide your interaction history. However, you
        must still provide your neighbor node numbers. (The Hannah variant.)
        You will express these in a dict with those node numbers as keys and
        "None" as the values. See below for an example legal return value.

        If you choose this option, you will receive nothing.

        3. You may provide a reported interaction history with one or more
        factual errors. Important: this interaction history must still contain
        all your current neighbors as keys. You cannot lie about this. But
        the history of those neighbors with those opponents can be changed in
        any way. The legal format for the return value in this case is exactly
        the same as in option 1, above.

        If you choose this option, you will be charged $.5.


        Example return value for choices 1 or 3, above:

        { 3: [
            {'step':1,'self_move':'C','other_move':'D'},
            {'step':2,'self_move':'D','other_move':'D'}
          ],
          8: [
            {'step':2,'self_move':'D','other_move':'C'},
          ]
        }

        Example return value for choice 2, above:
    
        { 3: None, 8: None }

        """
        return self.history     # Sincerely yours, I.M.Truthful.
