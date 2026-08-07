from ipd_llm.rng import create_random_streams


STREAM_NAMES = (
    "action",
    "information_request",
    "reputation_response",
    "tie_severing",
    "replacement_selection",
    "candidate_sampling",
    "reputation_corruption",
    "conflict_resolution",
)


def test_same_seed_reproduces_each_runtime_stream():
    first = create_random_streams(12345)
    second = create_random_streams(12345)

    for name in STREAM_NAMES:
        first_rng = getattr(first, name)
        second_rng = getattr(second, name)

        assert [first_rng.random() for _ in range(5)] == [
            second_rng.random()
            for _ in range(5)
        ]


def test_runtime_streams_do_not_share_identical_sequences():
    streams = create_random_streams(12345)

    sequences = {
        name: tuple(
            getattr(streams, name).random()
            for _ in range(5)
        )
        for name in STREAM_NAMES
    }

    assert len(set(sequences.values())) == len(STREAM_NAMES)


def test_advancing_one_stream_does_not_advance_another():
    first = create_random_streams(12345)
    second = create_random_streams(12345)

    for _ in range(10):
        first.action.random()

    assert (
        first.replacement_selection.random()
        == second.replacement_selection.random()
    )
