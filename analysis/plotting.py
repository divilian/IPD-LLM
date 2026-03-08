def setup_plotting(self) -> None:
    """
    This method must be called once before subsequent calls to .plot() are
    made. It sets up instance variables:
        - fig, ax (matplotlib objects used thereafter)
        - pos (layout positions of nodes)
        - cmap (the colormap used to plot wealth as color)
        - norm (the normalizer used to scale wealths for plotting)
    """
    self.pos = nx.spring_layout(self.graph, seed=m.seed, k=1.2)
    self.cmap = mpl.colormaps["coolwarm"]  # blue->white->red
    self.fig, self.ax = plt.subplots(constrained_layout=True)
    self.ax.set_axis_off()
    self.norm = Normalize(
        vmin=0,
        vmax=estimate_expected_avg_wealth(self.graph),
        clip=True,
    )
    sm = ScalarMappable(norm=self.norm, cmap=self.cmap)
    sm.set_array([])
    self.fig.colorbar(sm, label="wealth", ax=self.ax)
    plt.show(block=False)

def plot(self):
    self.ax.clear()
    nx.draw_networkx_edges(
        self.graph,
        pos=self.pos,
        edge_color="black",
        width=1.0,
        ax=self.ax
    )
    nx.draw_networkx_labels(
        self.graph,
        pos=self.pos,
        font_size=10,
        font_color="black",
        ax=self.ax
    )
    nodes = list(self.graph.nodes())
    colors = [self.cmap(self.norm(w)) for w in monies]
    shapes = [self.node_to_agent[i].shape() for i in nodes]
    for shape in set(shapes):
        idx = [i for i, s in enumerate(shapes) if s == shape]
        nx.draw_networkx_nodes(
            self.graph,
            pos=self.pos,
            nodelist=[nodes[i] for i in idx],
            node_color=[colors[i] for i in idx],
            node_shape=shape,
            node_size=350,
            ax=self.ax,
        )
    self.fig.suptitle(f"Iteration {t+1} of {args.num_iter}")
    self.fig.canvas.draw_idle()
    self.fig.canvas.flush_events()
    time.sleep(0.01)

def estimate_expected_avg_wealth(g: nx.Graph):
    """
    Completely back-of-the-envelope estimate of "about how much should each
    agent expect to win during this situation?" The crude formula assumes an
    independent 50/50 chance of choosing to defect or cooperate.
    Note that we compute "per_iter" here (not "per_encounter") because we're
    discounting each agent's per-iteration winnings by its degree (which is its
    number of encounters).
    """
    avg_agent_per_iter = .25 * (args.R + args.T + args.S + args.P)
    return avg_agent_per_iter * args.num_iter
