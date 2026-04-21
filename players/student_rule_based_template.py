from agents.base import IPDAgent
from agents.factory import register_agent

@register_agent("StudentRuleBasedTemplate")
class StudentRuleBasedTemplate(IPDAgent):
    """
    Starting point for writing a rule-based IPD agent.

    (You should edit this file, not llm_agent.py.)

    Override any of these methods to customize behavior:
      - decide_against()            required
      - inform_foaf()               optional
      - request_rewire()            optional
    """

    def decide_against(self, other, give_rationale):
        """
        Make a decision against another agent (whose node number is passed).
        Return your decision ("C" or "D") and, if give_rationale is True, a
        description of the interaction (for logging). (If give_rationale is
        False, agents must still return a second value as the second element
        of the returned tuple, for simplicity.)

        So, to be concrete, this is a legal return value:

        return "C", ""
        """
        return "C", "I'm gullible"


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
        return self.history


    def request_rewire(
        self,
        max_rewires: int,
    ) -> dict[str, list[int]]:
        """
        Optionally do the following:

        (1) Specify up to max_rewires node IDs of agents you are currently
        adjacent to but no longer want to play with. (All node IDs in this
        list-to-sever must be your neighbors at the time this method is
        called.)

        (2) Specify exactly the same number of node IDs of agents you are not
        currently adjacent to but do want to play with. You can discover these
        by asking the model for information about your neighbor(s) via
        self.model.request_foaf_info_from(self, neighbor_node_num). (All node
        IDs in this list-to-add must *not* be your neighbors, but must be your
        FOAFs, at the time this method is called.)

        If you don't override this, you will never request a rewire. (This does
        not mean your neighbors will never change: anyone else in the zoo may
        choose to add/sever anybody, which includes you.)
        """
        return { 'nodes_to_sever': [], 'nodes_to_add': [] }

