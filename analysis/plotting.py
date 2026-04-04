import time

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import networkx as nx


def setup_plotting(model) -> dict:
    """
    This function must be called once before subsequent calls to plot() are
    made. It sets up plotting context variables:
        - fig, ax (matplotlib objects used thereafter)
        - pos (layout positions of nodes)
        - cmap (the colormap used to plot wealth as color)
        - norm (the normalizer used to scale wealths for plotting)
    and returns them in a "context" dict.
    """
    pos = nx.spring_layout(model.network.G, seed=model.seed, k=1.2)
    cmap = mpl.colormaps["coolwarm"]  # blue->white->red
    fig, ax = plt.subplots(constrained_layout=True, figsize=(9,8))
    ax.set_axis_off()
    norm = Normalize(
        vmin=0,
        vmax=model.estimate_expected_avg_wealth(),
        clip=True,
    )
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, label="wealth", ax=ax)
    plt.show(block=False)
    return {
        "pos": pos,
        "cmap": cmap,
        "fig": fig,
        "ax": ax,
        "norm": norm,
    }


def plot(model, ctx, monies, t, num_iter):
    # Update node positions gracefully.
    ctx["pos"] = nx.spring_layout(model.network.G, pos=ctx["pos"], k=1.2)
    ctx["ax"].clear()
    nx.draw_networkx_edges(
        model.network.G,
        pos=ctx["pos"],
        edge_color="black",
        width=1.0,
        ax=ctx["ax"]
    )
    nx.draw_networkx_labels(
        model.network.G,
        pos=ctx["pos"],
        font_size=10,
        font_color="black",
        ax=ctx["ax"]
    )
    nodes = list(model.network.G.nodes())
    colors = [ctx["cmap"](ctx["norm"](w)) for w in monies]
    shapes = [model.node_to_agent[i].shape() for i in nodes]
    sizes = [model.node_to_agent[i].size() for i in nodes]
    for shape in set(shapes):
        idx = [i for i, s in enumerate(shapes) if s == shape]
        nx.draw_networkx_nodes(
            model.network.G,
            pos=ctx["pos"],
            nodelist=[nodes[i] for i in idx],
            node_color=[colors[i] for i in idx],
            node_size=[sizes[i] for i in idx],
            node_shape=shape,
            ax=ctx["ax"],
        )
    ctx["fig"].suptitle(f"Iteration {t+1} of {num_iter}")
    ctx["fig"].canvas.draw_idle()
    ctx["fig"].canvas.flush_events()
    time.sleep(0.01)
