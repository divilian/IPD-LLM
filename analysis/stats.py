from collections import defaultdict
from typing import Type
import re

import pandas as pd
from mesa.discrete_space import CellAgent

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


def print_agent_type_adjacency_matrix(
    network,
    weighted: bool = False,
) -> pd.DataFrame:
    """
    Print a row-normalized matrix showing, for each CellAgent subclass, the
    average percentage of its neighbors that belong to each other subclass.

    Parameters
    ----------
    network : mesa.discrete_space.Network
        A Mesa Network object whose cells contain agents.
    weighted : bool, default False
        If False, each agent contributes equally to the average for its row
        type. This is usually what "on average" means.

        If True, agents with more neighbors contribute more heavily, because
        the percentages are computed from pooled edge counts for each type.

    Returns
    -------
    pd.DataFrame
        The percentage matrix as a DataFrame (values are floats from 0 to 100).

    Notes
    -----
    - Rows are focal agent types.
    - Columns are neighbor agent types.
    - Each row sums to about 100% (unless a type has no neighbors at all).
    - After row-normalization, the matrix is *not* guaranteed to be symmetric,
      even if the underlying graph is undirected.
    """

    # Gather all agents from all cells.
    agents: list[CellAgent] = []
    for cell in network.all_cells:
        agents.extend(cell.agents)

    if not agents:
        print("(No agents found.)")
        return pd.DataFrame()

    agent_types: list[Type[CellAgent]] = sorted(
        {type(agent) for agent in agents},
        key=lambda cls: cls.__name__,
    )
    type_names = [cls.__name__ for cls in agent_types]

    # ------------------------------------------------------------
    # Option 1: true per-agent average of neighbor composition
    # ------------------------------------------------------------
    if not weighted:
        row_sums = defaultdict(lambda: defaultdict(float))
        row_counts = defaultdict(int)

        for agent in agents:
            focal_type = type(agent)

            # Count this agent's neighbors by type.
            neighbor_counts = defaultdict(int)
            total_neighbors = 0

            for neighbor_cell in agent.cell.neighborhood:
                for neighbor_agent in neighbor_cell.agents:
                    neighbor_counts[type(neighbor_agent)] += 1
                    total_neighbors += 1

            # If this agent has no neighbors, skip it.
            if total_neighbors == 0:
                continue

            for neighbor_type in agent_types:
                pct = 100 * neighbor_counts[neighbor_type] / total_neighbors
                row_sums[focal_type][neighbor_type] += pct

            row_counts[focal_type] += 1

        matrix = []
        for focal_type in agent_types:
            row = []
            n = row_counts[focal_type]
            for neighbor_type in agent_types:
                val = row_sums[focal_type][neighbor_type] / n if n > 0 else 0.0
                row.append(val)
            matrix.append(row)

    # ------------------------------------------------------------
    # Option 2: pooled-edge version (high-degree agents count more)
    # ------------------------------------------------------------
    else:
        pooled_counts = defaultdict(lambda: defaultdict(int))

        for agent in agents:
            focal_type = type(agent)
            for neighbor_cell in agent.cell.neighborhood:
                for neighbor_agent in neighbor_cell.agents:
                    pooled_counts[focal_type][type(neighbor_agent)] += 1

        matrix = []
        for focal_type in agent_types:
            total = sum(pooled_counts[focal_type].values())
            row = []
            for neighbor_type in agent_types:
                val = (
                    100 * pooled_counts[focal_type][neighbor_type] / total
                    if total > 0
                    else 0.0
                )
                row.append(val)
            matrix.append(row)

    df = pd.DataFrame(matrix, index=type_names, columns=type_names)

    # Pretty-print as percentages.
    pretty = df.map(lambda x: f"{x:.0f}%")
    print(pretty.to_string())

    return df


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
