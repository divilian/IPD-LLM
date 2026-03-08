def compute_sbm_probs(
    sizes: list[int],
    avg_degree: float,
    homophily_weight: float,   # 0 = only diff-type edges; 1 = only same-type
) -> tuple[float, float]:
    """
    Compute (p_same, p_diff) for a vanilla SBM with:
      - block sizes
      - target average degree
      - homophily weight in [0,1]

    We allocate the expected-edge "budget" E across within/between pair sets,
    then convert expected edges to probabilities. If an extreme is infeasible
    (e.g., too few within-block pairs to hit avg_degree), we clamp at 1.0.
    """
    N = sum(sizes)
    if N <= 1:
        return 0.0, 0.0

    max_edges_same = sum(n * (n - 1) / 2 for n in sizes)
    max_edges_diff = sum(
        sizes[i] * sizes[j]
        for i in range(len(sizes))
        for j in range(i + 1, len(sizes))
    )

    # Target expected total edges for simple undirected graph.
    E = N * avg_degree / 2

    # Split the edge budget between same- and diff- type edges.
    E_same = homophily_weight * E
    E_diff = (1.0 - homophily_weight) * E

    # Convert expected edges to probabilities (clamp to [0,1]).
    p_same = 0.0 if max_edges_same == 0 else min(1.0, E_same / max_edges_same)
    p_diff = 0.0 if max_edges_diff == 0 else min(1.0, E_diff / max_edges_diff)

    return p_same, p_diff
