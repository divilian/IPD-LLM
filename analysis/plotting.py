import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse
import networkx as nx

DEF_VIDEO_FILE = "animation.mp4"
DEF_FRAME_PREFIX = "plot"


def player_display_name(agent) -> str:
    name_method = getattr(agent, "name", None)
    if callable(name_method):
        return str(name_method())
    return agent.__class__.__name__


def setup_plotting(model, interactive=True) -> dict:
    """
    This function must be called once before subsequent calls to plot() are
    made. It sets up plotting context variables:
    - fig, ax (matplotlib objects used thereafter)
    - pos (layout positions of nodes)
    - cmap (the colormap used to plot wealth as color)
    - norm (the normalizer used to scale wealths for plotting)
    and returns them in a "context" dict.

    If interactive is False, frames are written to a temporary directory so
    they can later be assembled into a video file.
    """
    pos = nx.spring_layout(model.network.G, seed=model.seed, k=1.2)
    cmap = mpl.colormaps["coolwarm"]   # blue->white->red
    fig, ax = plt.subplots(constrained_layout=True, figsize=(9, 8))
    ax.set_axis_off()
    norm = Normalize(
        vmin=0,
        vmax=model.estimate_expected_avg_wealth(),
        clip=True,
    )

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, label="wealth", ax=ax)

    temp_dir = None
    frame_dir = None
    if interactive:
        plt.show(block=False)
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="plot_frames_")
        frame_dir = Path(temp_dir.name)

    return {
        "pos": pos,
        "cmap": cmap,
        "fig": fig,
        "ax": ax,
        "norm": norm,
        "interactive": interactive,
        "frame_dir": frame_dir,
        "frame_idx": 0,
        "temp_dir": temp_dir,
    }


def _draw_player_oval(ax, pos, color, label, *, fontsize=10,
                      pad_pts=4, min_width=0.20, min_height=0.10):
    x, y = pos

    txt = ax.text(
        x, y, label,
        ha="center",
        va="center",
        multialignment="center",
        fontsize=fontsize,
        color="black",
        zorder=4,
    )

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    bbox = txt.get_window_extent(renderer=renderer)

    # padding in pixels
    pad_px = pad_pts * fig.dpi / 72.0

    bbox_padded = bbox.expanded(
        (bbox.width + 2 * pad_px) / bbox.width if bbox.width else 1.0,
        (bbox.height + 2 * pad_px) / bbox.height if bbox.height else 1.0,
    )

    (x0, y0), (x1, y1) = ax.transData.inverted().transform(
        [[bbox_padded.x0, bbox_padded.y0], [bbox_padded.x1, bbox_padded.y1]]
    )

    width = max(min_width, x1 - x0)
    height = max(min_height, y1 - y0)

    ellipse = Ellipse(
        (x, y),
        width=width,
        height=height,
        facecolor=color,
        edgecolor="black",
        linewidth=1.2,
        zorder=3,
    )
    ax.add_patch(ellipse)

    return ellipse, txt

def plot(model, ctx, monies, t, num_iter, interactive=True):
    # Update node positions gracefully.  
    ctx["pos"] = nx.spring_layout(model.network.G, pos=ctx["pos"], k=1.2)
    ctx["ax"].clear()
    ctx["ax"].set_axis_off()

    nx.draw_networkx_edges(
        model.network.G,
        pos=ctx["pos"],
        edge_color="black",
        width=1.0,
        ax=ctx["ax"]
    )

    nodes = list(model.network.G.nodes())
    colors = [ctx["cmap"](ctx["norm"](w)) for w in monies]

    ordinary_nodes = [n for n in nodes if not model.is_custom_player_node(n)]
    if ordinary_nodes:
        ordinary_shapes = [model.node_to_agent[i].shape() for i in ordinary_nodes]
        ordinary_sizes = [model.node_to_agent[i].size() for i in ordinary_nodes]
        ordinary_colors = [colors[nodes.index(i)] for i in ordinary_nodes]
        for shape in set(ordinary_shapes):
            idx = [i for i, s in enumerate(ordinary_shapes) if s == shape]
            nx.draw_networkx_nodes(
                model.network.G,
                pos=ctx["pos"],
                nodelist=[ordinary_nodes[i] for i in idx],
                node_color=[ordinary_colors[i] for i in idx],
                node_size=[ordinary_sizes[i] for i in idx],
                node_shape=shape,
                ax=ctx["ax"],
            )

    label_nodes = {}
    for i, node in enumerate(nodes):
        agent = model.node_to_agent[node]
        if model.is_custom_player_node(node):
            _draw_player_oval(
                ctx["ax"],
                ctx["pos"][node],
                colors[i],
                f"{player_display_name(agent)}\n({agent.node})",
            )
        else:
            label_nodes[node] = str(node)

    nx.draw_networkx_labels(
        model.network.G,
        pos=ctx["pos"],
        labels=label_nodes,
        font_size=10,
        font_color="black",
        ax=ctx["ax"]
    )

    ctx["fig"].suptitle(f"Iteration {t+1} of {num_iter}")

    effective_interactive = interactive and ctx.get("interactive", True)
    if effective_interactive:
        ctx["fig"].canvas.draw_idle()
        ctx["fig"].canvas.flush_events()
        time.sleep(0.01)
    else:
        frame_idx = ctx.setdefault("frame_idx", 0)
        frame_dir = ctx.get("frame_dir")
        if frame_dir is None:
            raise RuntimeError("Non-interactive plotting requires frame dir.")
        frame_path = Path(frame_dir) / f"{DEF_FRAME_PREFIX}{frame_idx:03d}.svg"
        ctx["fig"].savefig(frame_path, format="svg")
        ctx["frame_idx"] = frame_idx + 1


def _require_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found on PATH")


def finalize_plotting(ctx, output_file=DEF_VIDEO_FILE, fps=10):
    """
    Finalize plotting.

    - In interactive mode, this is a no-op.
    - In non-interactive mode, assemble frames into output_file and delete
      intermediate frame files afterward.
    """
    if ctx.get("interactive", True):
        return None

    output_path = frames_to_mp4(
        frame_dir=ctx["frame_dir"],
        output_file=output_file,
        fps=fps,
    )
    cleanup_plot_frames(ctx)
    return output_path


def frames_to_mp4(frame_dir, output_file=DEF_VIDEO_FILE, fps=10):
    """
    Assemble SVG frames like plot000.svg, plot001.svg, ... into an MP4.
    Requires ffmpeg.
    """
    _require_ffmpeg()
    frame_dir = Path(frame_dir)
    output_file = Path(output_file)
    pattern = str(frame_dir / f"{DEF_FRAME_PREFIX}%03d.svg")

    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            pattern,
            "-pix_fmt",
            "yuv420p",
            str(output_file),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    return output_file


def cleanup_plot_frames(ctx):
    """
    Remove any intermediate frame files and clean up temporary directories.
    """
    frame_dir = ctx.get("frame_dir")
    if frame_dir is not None:
        for frame_path in Path(frame_dir).glob(f"{DEF_FRAME_PREFIX}[0-9][0-9][0-9].svg"):
            frame_path.unlink(missing_ok=True)

    temp_dir = ctx.get("temp_dir")
    if temp_dir is not None:
        temp_dir.cleanup()
    ctx["temp_dir"] = None
    ctx["frame_dir"] = None
