from collections import defaultdict

from mesa import Model
from mesa.discrete_space import CellAgent, Cell


class IPDAgent(CellAgent):
    """
    Public interface for tournament agents.

    Required override:
        - decide_against()

    Optional overrides:
        - choose_neighbors_to_sever()
        - choose_new_neighbors()
        - inform_foaf()
        - shape()
        - size()

    Notice there is no ".step()" method, here or in subclasses, as is
    traditional in Mesa. We instead require ".decide_against()" for each
    subclass.
    """
    def __init__(self, model: Model, cell: Cell):
        super().__init__(model)
        self.cell = cell
        self.current_iter_payment = 0

        # history[other_node] = list of {step, self_move, other_move}
        self.history = defaultdict(list)

        # current_decisions[other_node] = move for THIS step only
        self.current_decisions = {}

        self.wealth = 0.0

    """
    OVERRIDABLE METHODS FOR SUBCLASSES
    """
    def decide_against(
        self,
        other: "IPDAgent",
        payoff_matrix: dict[tuple[str, str], tuple[str, str]],
        give_rationale: bool,
    ) -> tuple[str, str]:
        """
        Make a decision against another agent. Return your decision ("C" or
        "D") and, if give_rationale is True, a description of the interaction
        (for logging). (If give_rationale is False, agents must still return an
        empty string as the second element of the returned tuple, for
        simplicity.)
        """
        raise NotImplementedError

    def rewire_as_desired(
        self,
        payoff_matrix: dict[tuple[str, str], tuple[str, str]],
        max_rewires: int,
    ) -> None:
        """
        Optionally do the following:
        (1) Sever up to max_rewires graph connections to agents you no longer
        want to play with.
        (2) For each such severance, form a new connection to another agent.
        You can discover these by asking your neighbors for information via
        `.inform_foaf()`.

        This uses the Template Method pattern. Subclasses may override hook
        methods like `choose_neighbors_to_sever()` and `choose_new_neighbors()`
        to control rewiring policy, or override this entire method if they need
        a completely different rewiring algorithm.
        """
        if self.model.debug:
            print(
                "***************************************\n"
                f"{self.unique_id} (node {self.node}) is considering rewiring"
            )

        current_neighbors = self._get_current_neighbors()
        my_foafs = self._get_foaf_nodes()
        new_neighbor_candidates = self._get_eligible_new_neighbor_candidates(
            candidate_nodes=my_foafs,
            starting_neighbors=current_neighbors,
            severed_nodes=set(),
        )

        sever_targets = self.choose_neighbors_to_sever(
            payoff_matrix=payoff_matrix,
            starting_neighbors=current_neighbors,
            new_neighbor_candidates=new_neighbor_candidates,
            max_rewires=max_rewires,
        )

        sever_targets = self._sanitize_sever_targets(
            sever_targets=sever_targets,
            current_neighbors=current_neighbors,
            max_rewires=max_rewires,
        )

        severed_nodes = self._actually_sever_connections(sever_targets)
        num_needed_replacements = len(severed_nodes)

        if self.model.debug:
            if num_needed_replacements:
                print(
                    f"I've terminated {num_needed_replacements} "
                    "connections, and will now try to make that many more."
                )
            else:
                print("Nobody terminated; no need to make more connections.")

        if not num_needed_replacements:
            return

        eligible_new_neighbors = self._get_eligible_new_neighbor_candidates(
            candidate_nodes=my_foafs,
            starting_neighbors=current_neighbors,
            severed_nodes=severed_nodes,
        )

        chosen_new_neighbors = self.choose_new_neighbors(
            payoff_matrix=payoff_matrix,
            eligible_new_neighbors=eligible_new_neighbors,
            num_needed_replacements=num_needed_replacements,
            severed_nodes=severed_nodes,
            starting_neighbors=current_neighbors,
        )

        chosen_new_neighbors = self._sanitize_new_neighbor_choices(
            chosen_new_neighbors=chosen_new_neighbors,
            eligible_new_neighbors=eligible_new_neighbors,
            num_needed_replacements=num_needed_replacements,
        )

        self._actually_add_new_connections(chosen_new_neighbors)

    def choose_neighbors_to_sever(
        self,
        payoff_matrix: dict[tuple[str, str], tuple[str, str]],
        starting_neighbors: set[int],
        new_neighbor_candidates: set[int],
        max_rewires: int,
    ) -> list[int]:
        """
        Hook method.

        Return a list of neighbor node IDs that this agent wants to sever.
        Default behavior: no rewiring.
        """
        return []

    def choose_new_neighbors(
        self,
        payoff_matrix: dict[tuple[str, str], tuple[str, str]],
        eligible_new_neighbors: set[int],
        num_needed_replacements: int,
        severed_nodes: set[int],
        starting_neighbors: set[int],
    ) -> list[int]:
        """
        Hook method.

        Return a list of node IDs to connect to.
        Default behavior: choose uniformly at random from eligible new neighbors.
        """
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

        Note that it is perfectly permissible to lie about this history, if you
        deem that advantageous. You can also return None to give the inquirer
        the hand.
        """
        return self.history

    def _get_current_neighbors(self) -> set[int]:
        return {
            cell.coordinate
            for cell in self.cell.neighborhood.cells
            if cell.coordinate != self.node and cell.agents
        }

    def _get_reported_foaf_histories(
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

    def _get_foaf_nodes(self) -> set[int]:
        """
        Ask each current neighbor to report who their neighbors are
        (friends-of-friends from my perspective). This information is not
        necessarily truthful.
        """
        reported = self._get_reported_foaf_histories()

        foafs: set[int] = set()
        for informant_node, foaf_dict in reported.items():
            foafs.update(foaf_dict.keys())

        foafs.discard(self.node)
        return foafs

    def _get_eligible_new_neighbor_candidates(
        self,
        candidate_nodes: set[int],
        starting_neighbors: set[int],
        severed_nodes: set[int],
    ) -> set[int]:
        """
        Filter a broad pool of possible new-neighbor targets down to the nodes
        that are actually eligible to be added as neighbors in the current
        rewiring event.

        candidate_nodes: a to-be-culled list of the nodes you could potentially
        add as neighbors

        starting_neighbors: your neighbors list as it was at the start of this
        rewiring event.

        severed_nodes: the nodes you disconnected from in the first phase of
        this rewiring event.
        """
        eligible_nodes = set(candidate_nodes)
        eligible_nodes -= {self.node}
        eligible_nodes -= starting_neighbors
        eligible_nodes -= severed_nodes

        # As an extra safety check, only consider nodes who are not currently
        # adjacent to me. This prevents accidental duplicate connections even
        # if the FOAF set is stale or overinclusive.
        return {
            node for node in eligible_nodes
            if not self._is_adjacent_to_node(node)
        }

    """
    NON-OVERRIDABLE NUTS-N-BOLTS HELPER METHODS
    IF YOU MESS WITH THESE YOU WILL BE FLOGGED
    """
    def _sanitize_sever_targets(
        self,
        sever_targets: list[int],
        current_neighbors: set[int],
        max_rewires: int,
    ) -> list[int]:
        """
        (Helper for .rewire_as_desired() template method.)
        Keep only valid, unique, currently-adjacent neighbors.
        Preserve input order and cap at max_rewires.
        """
        seen = set()
        cleaned = []

        for node in sever_targets:
            if node in seen:
                continue
            if node not in current_neighbors:
                continue
            if not self._is_adjacent_to_node(node):
                continue
            seen.add(node)
            cleaned.append(node)
            if len(cleaned) >= max_rewires:
                break

        return cleaned

    def _sanitize_new_neighbor_choices(
        self,
        chosen_new_neighbors: list[int],
        eligible_new_neighbors: set[int],
        num_needed_replacements: int,
    ) -> list[int]:
        """
        (Helper for .rewire_as_desired() template method.)
        Keep only valid, unique replacement candidates.
        Preserve input order and cap at the number of needed replacements.
        """
        seen = set()
        cleaned = []

        for node in chosen_new_neighbors:
            if node in seen:
                continue
            if node not in eligible_new_neighbors:
                continue
            seen.add(node)
            cleaned.append(node)
            if len(cleaned) >= num_needed_replacements:
                break

        return cleaned

    def _actually_sever_connections(
        self,
        sever_targets: list[int],
    ) -> set[int]:
        """
        (Helper for .rewire_as_desired() template method.)
        """
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

    def _actually_add_new_connections(
        self,
        chosen_new_neighbors: list[int],
    ) -> None:
        """
        (Helper for .rewire_as_desired() template method.)
        """
        for new_friend_node in chosen_new_neighbors:
            self.model.network.add_connection(
                self.cell,
                self.model.node_to_agent[new_friend_node].cell,
            )
            if self.model.debug:
                print(f"I'm now friends with {new_friend_node}!")

    def _is_adjacent_to_node(self, x):
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
