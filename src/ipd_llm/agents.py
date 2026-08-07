"""Mesa agent state and direct interaction histories."""

from mesa import Agent, Model

from ipd_llm.policies import Action, ActionPolicy, History, Interaction


class IPDAgent(Agent):
    """One simulator agent and its direct interaction histories."""

    def __init__(
        self,
        model: Model,
        node: int,
        action_policy: ActionPolicy,
    ) -> None:
        super().__init__(model)
        self.node = node
        self.action_policy = action_policy
        self._histories: dict[int, list[Interaction]] = {}

    @property
    def identifier(self) -> str:
        """Return the neutral identifier used in prompts and logs."""

        return f"A{self.unique_id:02d}"

    def history_with(self, opponent_node: int) -> History:
        """Return this agent's direct history with one opponent."""

        return tuple(self._histories.get(opponent_node, ()))

    def choose_action(self, opponent_node: int) -> Action:
        """Choose an action using only direct history with the opponent."""

        return self.action_policy.choose_action(
            self.history_with(opponent_node)
        )

    def record_interaction(
        self,
        opponent_node: int,
        interaction: Interaction,
    ) -> None:
        """Append one direct interaction with an opponent."""

        self._histories.setdefault(
            opponent_node,
            [],
        ).append(interaction)
