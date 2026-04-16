#!/usr/bin/env python
"""
cli.py

Iterated Prisoner's Dilemma on a Stochastic Block Model graph with:
- One agent per graph node
- Sucker, Mean, TitForTat, and LLM agents with various personas
- Configurable homophily based on agent type

Install:
pip install mesa networkx
Run syntax:
python pris.py -h
"""
from pathlib import Path
import argparse
import importlib.util
import inspect
import logging
import sys
from tqdm import tqdm

import pandas as pd
import matplotlib.pyplot as plt

from model import IPDModel
from agents.base import IPDAgent
from agents.factory import AGENT_REGISTRY
from llm.ollama_backend import OllamaBackend
from agents.factory import AgentFactory
from analysis.stats import (
    per_agent_type_stats,
    print_stats,
    print_agent_type_adjacency_matrix,
)
from analysis.plotting import setup_plotting, plot, finalize_plotting


def parse_args():
    parser = argparse.ArgumentParser(
        description="Iterated Prisoner's Dilemma on a graph",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "N",
        type=int,
        help="Number of agents drawn from --agent-fracs",
    )
    parser.add_argument(
        "--T",
        type=float,
        help="Temptation to defect (def: 5)",
        default=5.0,
    )
    parser.add_argument(
        "--R",
        type=float,
        help="Reward for cooperating (def: 3)",
        default=3.0,
    )
    parser.add_argument(
        "--P",
        type=float,
        help="Punishment for mutual defection (def: 1)",
        default=1.0,
    )
    parser.add_argument(
        "--S",
        type=float,
        help="Sucker's payoff (def: 0)",
        default=0.0,
    )

    agent_types = [t for t in AGENT_REGISTRY]
    parser.add_argument(
        "--agent-fracs",
        nargs="+",
        metavar=("AGENT", "FRAC"),
        default=["Sucker", "0.5", "Mean", "0.5"],
        help=(
            "Agent mix as pairs: AGENT FRAC AGENT FRAC ...\n"
            "AGENT is one of:\n"
            + "".join(f" - {agent_type}\n" for agent_type in agent_types)
            + "Ex: --agent-fracs Sucker 0.4 Mean 0.4 Random 0.2\n"
            + "(def: Sucker 0.5, Mean 0.5)"
        ),
    )
    parser.add_argument(
        "--players",
        type=str,
        default=None,
        help=(
            "Dir of .py files with additional IPDAgent subclasses"
        ),
    )
    parser.add_argument(
        "--avg-deg",
        type=float,
        default=3.0,
        help="Target average degree of nodes in graph",
    )
    parser.add_argument(
        "--p-same",
        type=float,
        default=0.5,
        help=(
            "Agency prob between agents of the "
            "same type (def: 0.5)"
        ),
    )
    parser.add_argument(
        "--p-diff",
        type=float,
        default=0.1,
        help=(
            "Agency prob between agents of diff "
            "types (def: 0.1)"
        ),
    )
    parser.add_argument(
        "--num-iter",
        type=int,
        default=25,
        help="Number of simulation iterations (def: 25)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Seed for rng's; starting (walking) seed for graph rng",
    )
    parser.add_argument(
        "--tft-noise",
        type=float,
        default=0.1,
        help="Tit-for-tat noise rate for some agent types (def: 0.1)",
    )
    parser.add_argument(
        "--max-rewires",
        type=int,
        default=3,
        help=(
            "Number of disconnect/connect operations agents are allowed each "
            "round (def: 3)"
        ),
    )
    parser.add_argument(
        "--llm-rewiring-aware",
        action="store_true",
        help="LLMs make rewiring decisions? (def: false)",
    )
    parser.add_argument(
        "--give-rationales",
        action="store_true",
        help="Include rationales narrating moves in log files? (def: false)",
    )
    parser.add_argument(
        "--llm-out-file",
        type=str,
        default="llm.out",
        help="Name of file (or None) to store LLM output (def: llm.out)",
    )
    parser.add_argument(
        "--llm-out-file-clobber",
        action="store_true",
        help="Auto-clobber LLM output file if exists? (def: false)",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default="llama3.1:8b-instruct-q4_K_M",
        help="Ollama model to use for LLM agents",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Log debug messages",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot animation",
    )
    parser.add_argument(
        "--plot-interactive",
        action="store_true",
        help="Show plot live instead of writing plot.mp4",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Launch interactive node analyzer",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Set logging level",
    )

    args = parser.parse_args()

    if not args.T > args.R > args.P > args.S:
        raise ValueError("PD constraint #1 violated (T>R>P>S).")
    if not 2 * args.R > args.S + args.T:
        raise ValueError("PD constraint #2 violated (2R>T+S).")
    if not (0.0 <= args.tft_noise <= 1.0):
        raise ValueError("--tft-noise must be between 0 and 1")
    if args.players is not None and not Path(args.players).is_dir():
        raise ValueError(f"--players must point to a directory: {args.players}")

    return args


def _load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_player_classes(players_dir: str | None) -> list[type[IPDAgent]]:
    if not players_dir:
        return []

    player_dir_path = Path(players_dir)
    classes: list[type[IPDAgent]] = []
    seen: set[tuple[str, str]] = set()

    for py_file in sorted(player_dir_path.glob("*.py")):
        module_name = f"user_players_{py_file.stem}"
        module = _load_module_from_path(module_name, py_file)

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj is IPDAgent:
                continue
            if not issubclass(obj, IPDAgent):
                continue
            if obj.__module__ != module.__name__:
                continue

            key = (obj.__module__, obj.__name__)
            if key in seen:
                continue
            seen.add(key)
            classes.append(obj)

    return classes


def interact_with_model(m: IPDModel):
    def node_prompt(m):
        return (
            f"Enter node ({','.join([str(n) for n in sorted(m.network.G.nodes)])},"
            "'done'): "
        )

    def neigh_prompt(m, n):
        neigh_list = ','.join([str(k) for k in m.network.G.neighbors(n)])
        return f" Enter neighbor of {n} ({neigh_list},'done'): "

    node_num_str = input(node_prompt(m))
    while node_num_str != "done" and node_num_str != "":
        n = int(node_num_str)
        print(m.node_to_agent[n])
        neighs = m.network.G.neighbors(n)
        if neighs:
            print("Neighbors:")
            for neigh in neighs:
                print(f" - {m.node_to_agent[neigh]}")
                node_num_str = input(neigh_prompt(m, n))
                while node_num_str != "done":
                    neigh = int(node_num_str)
                    if neigh in m.network.G.neighbors(n):
                        ncn = m.node_to_agent[neigh].__class__.__name__
                        print(f"History with {ncn} {neigh}:")
                        print(
                            pd.DataFrame(m.node_to_agent[n].history[neigh]).rename(
                                columns={
                                    'self_action': f'Node {n}',
                                    'other_action': f'Node {neigh}'
                                }
                            )
                        )
                    else:
                        print(f" (Node {n} not adjacent to {neigh}.)")
                    node_num_str = input(neigh_prompt(m, n))

        node_num_str = input(node_prompt(m))


def setup_runtime(args):
    logging.basicConfig(
        level=args.log_level,
        format="%(message)s"
    )
    if any(['LLM' in a for a in args.agent_fracs]):
        out_path = Path(args.llm_out_file)
        if out_path.exists():
            if args.llm_out_file_clobber:
                out_path.unlink()
            else:
                clob = input(
                    f"Clobber existing {args.llm_out_file} file? (y/n) ",
                ).strip().lower()
                if clob == "y":
                    out_path.unlink()
        with open(out_path, "a", encoding="utf-8") as f:
            print("=============================================", file=f)


if __name__ == "__main__":
    args = parse_args()
    setup_runtime(args)

    stats = []

    # ------------------------------------------------------------
    # Payoff matrix.
    # To be a valid prisoner's dilemma, T > R > P > S.
    # Also, to avoid alternating exploitation, 2R > T + S.
    # ------------------------------------------------------------
    payoff_matrix = {
        ("C", "C"): (args.R, args.R),
        ("C", "D"): (args.S, args.T),
        ("D", "C"): (args.T, args.S),
        ("D", "D"): (args.P, args.P),
    }

    factory = AgentFactory.instance(args.agent_fracs, args)
    player_classes = load_player_classes(args.players)

    backend = OllamaBackend(
        model_name=args.ollama_model,
        host="http://localhost:11434",
        timeout=120,
        seed=123,
        num_ctx=2048,
    )

    m = IPDModel(
        N=args.N,
        avg_degree=args.avg_deg,
        payoff_matrix=payoff_matrix,
        p_same=args.p_same,
        p_diff=args.p_diff,
        num_iter=args.num_iter,
        agent_factory=factory,
        extra_agent_classes=player_classes,
        max_rewires=args.max_rewires,
        give_rationales=args.give_rationales,
        llm_out_file=args.llm_out_file,
        ollama_model=args.ollama_model,
        llm_backend=backend,
        debug=args.log,
        seed=args.seed,
    )

    print(f"Running {m}")
    print("Before simulation, agent-type adjacency:")
    print_agent_type_adjacency_matrix(m.network)

    if args.plot:
        # Plot once before any simulation steps.
        plot_context = setup_plotting(m, interactive=args.plot_interactive)
        monies = [m.node_to_agent[n].wealth for n in m.network.G.nodes]
        plot(
            m,
            plot_context,
            monies,
            t,
            args.num_iter,
            interactive=args.plot_interactive,
        )

    for t in tqdm(range(args.num_iter)):
        m.step()
        # Then, plot after each step.
        monies = [m.node_to_agent[n].wealth for n in m.network.G.nodes]
        if args.plot:
            plot(
                m,
                plot_context,
                monies,
                t,
                args.num_iter,
                interactive=args.plot_interactive,
            )
        row = {"step": t + 1}
        row.update(per_agent_type_stats(m))
        stats.append(row)

    print("After simulation, agent-type adjacency:")
    print_agent_type_adjacency_matrix(m.network)

    stats = pd.DataFrame(stats)
    print_stats(stats)

    if args.plot:
        if args.plot_interactive:
            plt.show()
        else:
            finalize_plotting(plot_context, output_file="animation.mp4")

    if args.analyze:
        interact_with_model(m)
