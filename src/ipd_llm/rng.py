"""Independent pseudorandom-number streams for the simulator."""

from dataclasses import dataclass
import hashlib
import random


@dataclass
class RandomStreams:
    """Independent pseudorandom generators for distinct mechanisms.

    Each field is a separate ``random.Random`` instance.  This prevents a
    random draw in one simulation mechanism from advancing the generator used
    by another mechanism.  The streams also receive distinct derived seeds,
    so they do not begin with identical pseudorandom sequences.

    The fields are named for the mechanism that consumes the randomness, not
    for the type of agent making the decision.  For example, a rule-based
    agent that randomly chooses both a Prisoner's Dilemma action and a
    replacement partner must draw from ``action`` and
    ``replacement_selection``, respectively.

    ``information_request``, ``reputation_response``, and ``tie_severing`` are
    reserved for stochastic policies in those decision phases.
    ``replacement_selection`` covers stochastic choice among eligible
    replacement partners, including simulator-selected replacement in
    Condition 2.  ``candidate_sampling`` is used only if an experiment caps
    the candidate pool by random sampling.  ``reputation_corruption`` controls
    the stochastic C/D flips in deceptive reports.  ``conflict_resolution``
    controls seeded tie-breaking among incompatible rewiring proposals.

    LLM sampling is not represented here because its seeding, when available,
    belongs to the inference provider rather than Python's ``random`` module.
    """

    action: random.Random
    information_request: random.Random
    reputation_response: random.Random
    tie_severing: random.Random
    replacement_selection: random.Random
    candidate_sampling: random.Random
    reputation_corruption: random.Random
    conflict_resolution: random.Random


def derive_seed(seed: int, stream: str) -> int:
    """Derive one stable 64-bit seed for a named random stream."""

    data = f"{seed}:{stream}".encode("utf-8")
    digest = hashlib.sha256(data).digest()
    return int.from_bytes(digest[:8], byteorder="big")


def create_random_streams(seed: int) -> RandomStreams:
    """Create all runtime random streams from one initialization seed."""

    def rng(stream: str) -> random.Random:
        return random.Random(derive_seed(seed, stream))

    return RandomStreams(
        action=rng("action"),
        information_request=rng("information_request"),
        reputation_response=rng("reputation_response"),
        tie_severing=rng("tie_severing"),
        replacement_selection=rng("replacement_selection"),
        candidate_sampling=rng("candidate_sampling"),
        reputation_corruption=rng("reputation_corruption"),
        conflict_resolution=rng("conflict_resolution"),
    )
