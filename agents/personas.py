from enum import Enum, auto
from dataclasses import dataclass
import textwrap

class HistoryType(Enum):
    NONE = auto()    # for agents who don't need any history provided
    LAST = auto()    # for agents who need to know their opponents's last move
    FULL = auto()    # for agents who need the full history to decide move


@dataclass(frozen=True)
class Persona:
    instructions: str     # Strategy description.
    history: HistoryType  # How much stuff needs to go in the prompt
    sees_payoffs: bool


PERSONAS = {
    'tft': Persona(
        instructions=textwrap.dedent("""
            Your behavior is fixed: you must exactly repeat your opponent's
            most recent action. If this is the first move, choose randomly.
            """
        ).strip(),
        history=HistoryType.LAST,
        sees_payoffs=False,
    ),

    'vanilla': Persona(
        instructions=textwrap.dedent("""
            You should choose in a way that tries to maximize your total
            rewards over time.
            """
        ).strip(),
        history=HistoryType.NONE,
        sees_payoffs=True,
    ),

    'deep': Persona(
        instructions=textwrap.dedent("""
            You should think deeply about your past history of moves with each
            opponent to try and predict what they will do next, and respond
            accordingly.
            """
        ).strip(),
        history=HistoryType.FULL,
        sees_payoffs=True,
    ),
}

