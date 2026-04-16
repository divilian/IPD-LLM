from .base import IPDAgent
from .factory import register_agent

@register_agent("StudentRuleBasedTemplate")
class StudentRuleBasedTemplate(IPDAgent):
    """
    Starting point for writing a rule-based IPD agent.

    (You should edit this file, not llm_agent.py.)

    Override any of these methods to customize behavior:
      - decide_against()            required
      - choose_neighbors_to_sever() optional
      - choose_new_neighbors()      optional
      - inform_foaf()               optional

    or, if you want advanced custom rewiring behavior beyond just the
    "sever-then-replace" template:
      - rewire_as_desired()
    """

    def decide_against(self, other, payoff_matrix, give_rationale):

        move = "C"
        reason = "I always cooperate."
        return (move, reason) if give_rationale else (move, "")

    def choose_neighbors_to_sever(
        self,
        payoff_matrix,
        starting_neighbors,
        new_neighbor_candidates,
        max_rewires,
    ):
        return []

    def choose_new_neighbors(
        self,
        payoff_matrix,
        eligible_new_neighbors,
        num_needed_replacements,
        severed_nodes,
        starting_neighbors,
    ):
        eligible = list(eligible_new_neighbors)
        self.model.random.shuffle(eligible)
        return eligible[:num_needed_replacements]

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
