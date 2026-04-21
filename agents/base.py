from collections import defaultdict

from mesa import Model
from mesa.discrete_space import CellAgent, Cell


class IPDAgent(CellAgent):
    """
    Public interface for tournament agents.

    Required override:
        - decide_against()

    Optional overrides:
        - inform_foaf()
        - rewire_as_desired()
    """

    def __init__(
        self,
        model: Model,
        cell: Cell,
    ):
        super().__init__(model)
        self.cell = cell

        self.current_iter_payment = 0

        # history[other_node] = list of {step, self_move, other_move}
        self.history = defaultdict(list)

        # current_decisions[other_node] = move for THIS step only
        self.__current_decisions = {}

        self.wealth = 0.0


    """
    OVERRIDABLE METHODS FOR SUBCLASSES
    """
    def decide_against(
        self,
        other: int,
        give_rationale: bool,
    ) -> tuple[str, str]:
        """
        Make a decision against another agent (whose node number is passed).
        Return your decision ("C" or "D") and, if give_rationale is True, a
        description of the interaction (for logging). (If give_rationale is
        False, agents must still return a second value as the second element
        of the returned tuple, for simplicity.)

        So, to be concrete, this is a legal return value:

        return "C", ""
        """
        raise NotImplementedError


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


    """
    THE CODE BELOW IS AVAILABLE FOR YOU TO CALL AT YOUR CONVENIENCE. DO NOT
    OVERRIDE ANY OF IT, THOUGH.
    """

    def neighbors(self) -> list[int]:
        """
        Return a list of your current neighbors. (Stephen thinks it would be
        more pure to return a list, but this way saves student headache.)
        """
        return list(self.model.network.G.neighbors(self.node))


    """
    *** STOP READING NOW ***

    THE CODE BELOW IS NON-OVERRIDABLE NUTS-N-BOLTS HELPER METHODS
    IF YOU MESS WITH THESE YOU WILL BE FLOGGED
    """

    def __get_reported_foaf_histories(
        self,
    ) -> dict[int, dict[int, list[dict[str, int | str]]]]:
        """
        Ask each current neighbor to report interaction histories with their
        neighbors (friends-of-friends from my perspective).

        Returns a nested dict:
            {
                informant_node: {
                    foaf_node: [
                        { "step": ..., "self_move": ..., "other_move": ... },
                     ...
                    ],
                    ...
                },
                ...
            }

        Each informant's contribution is exactly whatever their
        inform_foaf(self) method returns:
            - Truthful: real histories.
            - Refusal (None): no entry for that informant.
            - Lie: corrupted histories, as defined by the informant subclass.
        """
        reported: dict[int, dict[int, list[dict[str, int | str]]]] = {}

        # Iterate over my current neighbor cells
        for c in self.cell.neighborhood.cells:
            if not c.agents:
                continue

            informant = c.agents[0]
            # Skip if the cell contains myself
            if informant is self:
                continue

            informant_node = informant.node
            report = informant.inform_foaf(self)

            # Refusal: this informant contributes no information
            if report is None:
                continue

            # Defensive check: ensure the report is a dict-like structure
            if not isinstance(report, dict):
                if self.model.debug:
                    print(
                        f"Warning: informant {informant_node} returned "
                        f"non-dict from inform_foaf(): {type(report)}"
                    )
                continue

            # Store the report as-is under the informant's node id
            reported[informant_node] = report

        return reported

    def __get_foaf_nodes(self) -> set[int]:
        """
        Ask each current neighbor to report who their neighbors are
        (friends-of-friends from my perspective). This information is truthful.
        """
        reported = self.__get_reported_foaf_histories()

        foafs: set[int] = set()
        for informant_node, foaf_dict in reported.items():
            foafs.update(foaf_dict.keys())

        foafs.discard(self.node)
        return foafs

    def __actually_sever_connections(
        self,
        sever_targets: list[int],
    ) -> set[int]:
        severed_nodes = set()

        for node in sever_targets:
            if self.model.debug:
                print(f"You're dead to me, {node}.")

            self.model.network.remove_connection(
                self.cell,
                self.model.node_to_agent[node].cell,
            )
            severed_nodes.add(node)

        return severed_nodes

    def __actually_add_new_connections(
        self,
        chosen_new_neighbors: list[int],
    ) -> None:
        for new_friend_node in chosen_new_neighbors:
            self.model.network.add_connection(
                self.cell,
                self.model.node_to_agent[new_friend_node].cell,
            )
            if self.model.debug:
                print(f"I'm now friends with {new_friend_node}!")

    def __is_adjacent_to_node(self, x):
        return any(
            (cell.coordinate == x for cell in self.cell.neighborhood.cells)
        )

    def record_interaction(
        self,
        other_node: int,
        self_move: str,
        other_move: str,
    ) -> None:
        self.history[other_node].append(
            {
                "step": self.model.steps,
                "self_move": self_move,
                "other_move": other_move,
            }
        )

    def shape(self) -> str:
        """
        Return the shape your node should be in the graph. See:
        https://matplotlib.org/stable/api/markers_api.html.
        """
        return 's'


    def size(self) -> str:
        """
        Return the size your node should be in the graph.
        """
        return 300


    @property
    def node(self) -> int:
        return self.cell.coordinate

    @property
    def wealth(self) -> int:
        return self.__wealth

    @wealth.setter
    def wealth(self, value) -> None:
        self.__wealth = value

    @property
    def current_decisions(self) -> int:
        return self.__current_decisions

    display_name = None
    @classmethod
    def name(cls) -> str:
        if cls.display_name is not None:
            return cls.display_name
        return cls.__name__

    def __str__(self) -> str:
        return (
            f"Node {self.node} (agent id {self.unique_id}) "
            f"{self.__class__.__name__} "
            f"with ${int(self.wealth)}"
        )
