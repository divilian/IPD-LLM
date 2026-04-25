from __future__ import annotations

from mesa import Model
from mesa.discrete_space import Cell

from agents.factory import register_agent
from agents.llm_agent import LLMAgent


@register_agent("LLMExample")
class LLMExample(LLMAgent):
    """
    Example LLM-based IPD agent.
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
        return "You are a strongly principled player of Iterated Prisoner's Dilemma, being cautiously cooperative at first, but not at all hesitatant to punish defectors in a big way."

    def build_decision_prompt(
        self,
        other,
    ) -> str:
        return f"""
        You are playing a game of Iterated Prisoner's Dilemma simultaneously
against many opponents situated on a network.

Right now, you need to decide what move to make against player {other}.

The payoffs are as follows:

{self.serialize_payoffs()}

This is round {self.model.steps} out of a total of {self.model.num_iter}.

{self.serialize_history(other)}

How would you like to move? Output exactly one of C or D.
"""
        

    def build_rewiring_prompt(
        self,
        max_rewires: int,
    ) -> str:
        p = f"""
You now have the option to drop up to {max_rewires} of your opponents and
replace them with exactly the same number of new ones. Those candidate new
ones, with move histories against other opponents, will be provided below.

Dropping and adding is optional. You do not have drop or add any opponents if
you feel you are doing well in the game against your current set. However, you
should considering dropping/adding if you have an opponent who consistently
defects.

Here's your current move history against all your current opponents (which are
'drop candidates'):
"""
        for neighbor in self.neighbors():
            p += self.serialize_history(neighbor) + "\n"

        foafs_and_histories = {}
        for neighbor in self.neighbors():
            foafs_and_histories.update(
                self.model.request_foaf_info_from(self, neighbor)
            )

        foafs_and_histories = {
            node:hist
            for node, hist in foafs_and_histories.items()
            if node not in self.neighbors()
        }
        p += "\nHere is the set of nodes ('add candidates') you may choose to add, if you wish: "
        p += ", ".join([str(i) for i in sorted(foafs_and_histories.keys())])

        p += "\nHere are example histories against other nodes:\n"
        for node, hist in foafs_and_histories.items():
            if node not in self.neighbors():
                p += self.serialize_history(node, {node:hist}, 0) + "\n"

        p += """
Now choose which of your drop candidates you'd like to drop, and which add
candidates you'd like to add. The number of drops must exactly equal the number
of adds.
        """
        p += "Output exactly a JSON document in this format:\n"
        p += "{'nodes_to_drop':[1,2],'nodes_to_add':[3,4]}\n"
        p += "where the two lists are the same-length lists of opponents you'd like to stop playing with, and start playing with, respectively."
        return p

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
        # Even though I haven't done it in this example, you can absolutely,
        # totally use the LLM here too, to decide what information to provide
        # to FOAFs (truth? lie? give no info?)
        return self.history
