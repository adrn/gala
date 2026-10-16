"""Plot pytest-benchmark scaling results: median time vs. each benchmark parameter.

By default, results are fetched from the ``benchmark-results`` branch on GitHub (the
file for this gala release, or the latest ``main`` run). Set ``GALA_BENCH_JSON`` to
read a local file instead, and ``GALA_BENCH_OUT`` to a directory to save the figures.
"""

import json
import os
import urllib.request
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

import gala

URL = "https://raw.githubusercontent.com/adrn/gala/benchmark-results/results-{}.json"


def load_results():
    if path := os.environ.get("GALA_BENCH_JSON"):
        return json.loads(Path(path).read_text())

    names = ["main"]
    if "dev" not in gala.__version__:
        names.insert(0, f"v{gala.__version__}")  # tag file may not exist yet
    for name in names:
        try:
            with urllib.request.urlopen(URL.format(name), timeout=10) as f:
                return json.load(f)
        except OSError:
            continue
    return None


def plot_function(name, benches):
    params = list(benches[0]["params"])
    fig, axes = plt.subplots(
        1,
        len(params),
        figsize=(4.5 * len(params), 4),
        squeeze=False,
        layout="constrained",
    )
    for ax, param in zip(axes[0], params):
        others = [p for p in params if p != param]
        lines = defaultdict(list)
        for b in benches:
            key = tuple(b["params"][p] for p in others)
            lines[key].append((b["params"][param], b["stats"]["median"]))

        for key, points in sorted(lines.items()):
            x, t = zip(*sorted(points))
            label = ", ".join(f"{p}={v}" for p, v in zip(others, key)) or None
            ax.plot(x, t, marker="o", label=label)

        ax.set(xscale="log", yscale="log", xlabel=param, ylabel="median time [s]")
        if others:
            ax.legend(fontsize=8)
    fig.suptitle(name)
    return fig


results = load_results()
if results is None:
    fig, ax = plt.subplots(figsize=(6, 1))
    ax.text(0.5, 0.5, "Benchmark results unavailable", ha="center", va="center")
    ax.set_axis_off()
else:
    by_function = defaultdict(list)
    for b in results["benchmarks"]:
        by_function[b["name"].split("[")[0]].append(b)

    out = os.environ.get("GALA_BENCH_OUT")
    for name, benches in by_function.items():
        fig = plot_function(name, benches)
        if out:
            Path(out).mkdir(parents=True, exist_ok=True)
            fig.savefig(Path(out) / f"{name}.png", dpi=150)
