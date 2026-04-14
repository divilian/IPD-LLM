from .base import IPDAgent
from .factory import register_agent

@register_agent("StudentTemplate")
class StudentTemplateAgent(IPDAgent):
    """
    Minimal example agent for students.

    Override any of these methods to customize behavior:
      - decide_against()            required
      - choose_neighbors_to_sever() optional
      - choose_new_neighbors()      optional
      - inform_foaf()               optional
      - shape()                     optional
      - size()                      optional

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

    def inform_foaf(self, inquirer):
        return self.history

    def shape(self):
        return "o"

    def size(self):
        return 300
