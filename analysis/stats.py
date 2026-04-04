import re

import pandas as pd

from model import IPDModel

def per_agent_type_stats(
    model: IPDModel,
) -> dict[str, float]:
    """
    Compute per-agent-type averages for the current step.
    Returns a flat dict suitable for a DataFrame row.
    """
    from collections import defaultdict

    agent_counts = defaultdict(int)
    coop_counts = defaultdict(int)
    decision_counts = defaultdict(int)
    wealth_sums = defaultdict(float)

    for agent in model.agents:
        cls = agent.__class__.__name__
        agent_counts[cls] += 1
        wealth_sums[cls] += agent.wealth
        for d in agent.current_decisions.values():
            decision_counts[cls] += 1
            if d == "C":
                coop_counts[cls] += 1

    row = {}
    for cls in agent_counts:
        row[f"{cls}Coop".replace("Agent", "")] = (
            coop_counts[cls] / decision_counts[cls]
            if decision_counts[cls] > 0
            else 0.0
        )
        row[f"{cls}$".replace("Agent", "",)] = (
            wealth_sums[cls] / agent_counts[cls]
            if agent_counts[cls] > 0
            else 0.0
        )
    return row


def print_stats(stats: pd.DataFrame, last_n=20, plot_means=False):

    stats.drop('step', axis=1, inplace=True)

    pd.options.display.float_format = "{:.2f}".format

    cols = stats.columns

    things = sorted({
        re.sub(r"(Coop|\$)$", "", c)
        for c in cols
        if c.endswith("Coop") or c.endswith("$")
    })

    ordered = (
        [f"{t}Coop" for t in things if f"{t}Coop" in cols] +
        [f"{t}$"    for t in things if f"{t}$"    in cols]
    )

    other = [c for c in cols if c not in ordered]

    stats = stats[other + ordered]

    with pd.option_context(
        "display.width", None,
        "display.max_columns", None,
        "display.max_colwidth", None,
        "display.expand_frame_repr", False,
    ):
        print(stats.tail(last_n))

    if plot_means:
        to_plot = stats[[f"{t}$"    for t in things if f"{t}$"    in cols]]
        to_plot.plot(kind="line")
