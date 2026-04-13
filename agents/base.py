from collections.abc import Callable
from typing import TypeVar
from collections import defaultdict

from mesa import Model
from mesa.discrete_space import CellAgent, Cell


T = TypeVar("T")

AGENT_REGISTRY: dict[str, type] = {}


def register_agent(name: str) -> Callable[[T], T]:
    """
    Decorator for new Agent IPDsubclasses.
    """
    def decorator(cls: T) -> T:
        AGENT_REGISTRY[name] = cls
        return cls
    return decorator


class IPDAgent(CellAgent):
    """
    Base class for all IPD agents. Note: there is no ".step()" method, here or
    in subclasses, as is traditional in Mesa. We instead require
    ".decide_against()" for each subclass.
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

    def is_adjacent_to_node(self, x):
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
    ) -> None:
        """
        Optionally do either or both of the following:
        (1) Sever graph connections to agents you no longer want to play with.
        (2) Create graph connections to other agents, which you can discover by
        asking your neighbors for information via `.inform_foaf()`.

        This uses the Template Method pattern. Subclasses may override hook
        methods like `choose_neighbors_to_sever()` and `choose_new_neighbors()`
        to control rewiring policy, or override this entire method if they need
        a completely different rewiring algorithm.
        """
        def sanitize_sever_targets(
            sever_targets: list[int],
            starting_neighbors: set[int],
        ) -> list[int]:
            """
            (Helper for .rewire_as_desired() template method.)
            Keep only valid, unique, currently-adjacent neighbors.
            Preserve input order.
            """
            seen = set()
            cleaned = []

            for node in sever_targets:
                if node in seen:
                    continue
                if node not in starting_neighbors:
                    continue
                if not self.is_adjacent_to_node(node):
                    continue
                seen.add(node)
                cleaned.append(node)

            return cleaned

        def sanitize_new_neighbor_choices(
            chosen_new_neighbors: list[int],
            eligible_foafs: set[int],
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
                if node not in eligible_foafs:
                    continue
                seen.add(node)
                cleaned.append(node)
                if len(cleaned) >= num_needed_replacements:
                    break

            return cleaned

        def sever_connections(sever_targets: list[int]) -> set[int]:
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

        def add_new_connections(chosen_new_neighbors: list[int]) -> None:
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

            if self.model.debug and not chosen_new_neighbors:
                print("Yikes! Nobody to replace severed connection with.")

        # *Actual start of default template method .rewire_as_desired().
        if self.model.debug:
            print(
                "***************************************\n"
                f"{self.unique_id} (node {self.node}) is considering rewiring"
            )

        my_foafs = self._get_foaf_nodes()
        starting_neighbors = self._get_current_neighbors()

        sever_targets = self.choose_neighbors_to_sever(
            payoff_matrix=payoff_matrix,
            starting_neighbors=starting_neighbors,
            my_foafs=my_foafs,
        )

        sever_targets = sanitize_sever_targets(
            sever_targets=sever_targets,
            starting_neighbors=starting_neighbors,
        )

        severed_nodes = sever_connections(sever_targets)
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

        eligible_foafs = self._get_eligible_rewiring_candidates(
            my_foafs,
            starting_neighbors,
            severed_nodes,
        )

        chosen_new_neighbors = self.choose_new_neighbors(
            payoff_matrix=payoff_matrix,
            eligible_foafs=eligible_foafs,
            num_needed_replacements=num_needed_replacements,
            severed_nodes=severed_nodes,
            starting_neighbors=starting_neighbors,
        )

        chosen_new_neighbors = sanitize_new_neighbor_choices(
            chosen_new_neighbors=chosen_new_neighbors,
            eligible_foafs=eligible_foafs,
            num_needed_replacements=num_needed_replacements,
        )

        add_new_connections(chosen_new_neighbors)

    def choose_neighbors_to_sever(
        self,
        payoff_matrix: dict[tuple[str, str], tuple[str, str]],
        starting_neighbors: set[int],
        my_foafs: set[int],
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
        eligible_foafs: set[int],
        num_needed_replacements: int,
        severed_nodes: set[int],
        starting_neighbors: set[int],
    ) -> list[int]:
        """
        Hook method.

        Return a list of node IDs to connect to.
        Default behavior: choose uniformly at random from eligible FOAFs.
        """
        eligible = list(eligible_foafs)
        self.model.random.shuffle(eligible)
        return eligible[:num_needed_replacements]


    def _foaf_candidates(self):
        reports = [
            c.agents[0].inform_foaf(self) for c in self.cell.neighborhood.cells
        ]
        foafs = {
            foaf_node
            for fr in reports
            if fr is not None
            for foaf_node in fr
        }
        foafs -= {self.node}
        return foafs

    def _apply_rewiring(self, drop_nodes, candidate_nodes):
        for node in drop_nodes:
            self.model.network.remove_connection(
                self.cell,
                self.model.node_to_agent[node].cell,
            )

        num_needed = len(drop_nodes)
        available = set(candidate_nodes)

        for _ in range(num_needed):
            available = {
                n for n in available
                if n != self.node and not self.is_adjacent_to_node(n)
            }
            if not available:
                return
            new_node = self.model.random.choice(list(available))
            self.model.network.add_connection(
                self.cell,
                self.model.node_to_agent[new_node].cell,
            )
            available.remove(new_node)

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

    def _get_eligible_rewiring_candidates(
        self,
        candidate_nodes: set[int],
        starting_neighbors: set[int],
        severed_nodes: set[int],
    ) -> set[int]:
        """
        Return the subset of candidate rewiring targets that are valid for
        immediate replacement.
        """
        # Remove from consideration:
        # - myself,
        # - anyone who was already my neighbor when this rewiring step began,
        # and
        # - anyone I just severed.
        #
        # Note that `severed_nodes` is logically redundant with
        # `starting_neighbors`, but I am leaving it explicit here because it
        # makes the intent clearer: someone I just dumped should not be
        # eligible as an immediate replacement.
        eligible_nodes = set(candidate_nodes)
        eligible_nodes -= {self.node}
        eligible_nodes -= starting_neighbors
        eligible_nodes -= severed_nodes

        # As an extra safety check, only consider nodes who are not currently
        # adjacent to me. This prevents accidental duplicate connections even
        # if the FOAF set is stale or overinclusive.
        return {
            node for node in eligible_nodes
            if not self.is_adjacent_to_node(node)
        }

    def shape(self) -> str:
        """
        Return the shape your node should be in the graph. See:
        https://matplotlib.org/stable/api/markers_api.html.
        """
        raise 's'

    def size(self) -> str:
        """
        Return the size your node should be in the graph.
        """
        return 300

    @property
    def node(self) -> int:
        return self.cell.coordinate

    def __str__(self) -> str:
        return (
            f"Node {self.node} (agent id {self.unique_id}) "
            f"{self.__class__.__name__} "
            f"with ${int(self.wealth)}"
        )
