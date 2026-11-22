"""
Draws the report figures from docs/week09/data/runs.csv and curves.csv (written by AnnExperiments)
and writes docs/week09/data/summary.json.

  python tools/plot_experiments.py
"""
import csv
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = Path("docs/week09/data")
OUT = Path("docs/week09/images")
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTED, GRID = "#2b2b2a", "#6b6a64", "#e4e3de"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8981"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.titlesize": 11,
})


def load():
    runs = defaultdict(list)
    for r in csv.DictReader((DATA / "runs.csv").open(encoding="utf-8")):
        runs[r["condition"]].append(r)
    curves = defaultdict(lambda: defaultdict(list))  # condition -> seed -> [(epoch, sse)]
    for r in csv.DictReader((DATA / "curves.csv").open(encoding="utf-8")):
        v = float(r["sse"]) if r["sse"] != "NaN" else math.nan
        curves[r["condition"]][int(r["seed"])].append((int(r["epoch"]), v))
    return runs, curves


def learned(rows):
    return sum(int(r["learned"]) for r in rows)


def bars(ax, labels, values, n, color=BLUE):
    b = ax.bar(labels, values, color=color, width=0.65, edgecolor="white", linewidth=1.5)
    for rect, v in zip(b, values):
        ax.annotate(f"{v}/{n}", (rect.get_x() + rect.get_width() / 2, v), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=9, color=INK)
    ax.set_ylim(0, n * 1.12)
    ax.grid(axis="x", visible=False)


def fig_alpha(runs, curves, n):
    alphas = ["0.1", "0.4", "0.8", "1.5", "3", "10", "30", "100", "1000"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    vals = [learned(runs["alpha" + a]) for a in alphas]
    bars(axes[0], alphas, vals, n)
    nan = [sum(int(r["nan"]) for r in runs["alpha" + a]) for a in alphas]
    for i, k in enumerate(nan):
        if k:
            axes[0].annotate(f"{k} NaN", (i, vals[i]), xytext=(0, 22), textcoords="offset points",
                             ha="center", fontsize=9, color=ORANGE, fontweight="bold")
    axes[0].set_xlabel("Öğrenme oranı (alpha)")
    axes[0].set_ylabel(f"XOR'u öğrenen deneme ({n} seed)")
    axes[0].set_title("1000 epoch sonunda XOR'u öğrenen denemeler")
    for a, color in [("0.1", ORANGE), ("0.8", BLUE), ("3", AQUA)]:
        c = curves["alpha" + a]
        epochs = [e for e, _ in c[1]]
        med = np.nanmedian(np.array([[v for _, v in c[s]] for s in c]), axis=0)
        axes[1].plot(epochs, med, color=color, lw=2, label=f"alpha {a}")
        axes[1].annotate(f"{med[-1]:.3f}", (epochs[-1], med[-1]), xytext=(4, 0), textcoords="offset points",
                         va="center", fontsize=8, color=INK)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Hata kareleri toplamı (SSE, log ölçek)")
    axes[1].set_title("Medyan SSE (50 seed)")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "alpha.png", dpi=140)
    plt.close(fig)


def fig_curves(curves, runs):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    for ax, cond, title in [(axes[0], "default", "alpha 0.8, 1000 epoch: her çizgi bir deneme"),
                            (axes[1], "default_10000", "Aynı denemeler 10 000 epoch")]:
        ok = {int(r["seed"]) for r in runs[cond] if r["learned"] == "1"}
        for s, pts in curves[cond].items():
            e, v = zip(*pts)
            ax.plot(e, v, color=BLUE if s in ok else ORANGE, lw=0.8, alpha=0.55)
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("SSE (log ölçek)")
        ax.set_title(title)
        ax.axhline(0.01, color=MUTED, ls="--", lw=1)
    handles = [plt.Line2D([], [], color=BLUE, lw=2, label="XOR'u öğrendi"),
               plt.Line2D([], [], color=ORANGE, lw=2, label="yerel minimumda kaldı (SSE ≈ 0.8)"),
               plt.Line2D([], [], color=MUTED, lw=1, ls="--", label="SSE = 0.01")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "sse_curves.png", dpi=140)
    plt.close(fig)


def fig_activation(runs, n):
    acts = ["Sigmoid", "Tanh", "ReLU", "LeakyReLU", "Linear", "Step"]
    names = ["Sigmoid", "Tanh", "ReLU", "Leaky ReLU", "Doğrusal", "Step"]
    x = np.arange(len(acts))
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for off, a, color in [(-0.2, "0.1", ORANGE), (0.2, "0.8", BLUE)]:
        vals = [learned(runs[f"hidden_{act}_a{a}"]) for act in acts]
        b = ax.bar(x + off, vals, width=0.38, color=color, edgecolor="white", linewidth=1.5, label=f"alpha {a}")
        for rect, v in zip(b, vals):
            ax.annotate(str(v), (rect.get_x() + rect.get_width() / 2, v), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, names)
    ax.set_ylim(0, n * 1.12)
    ax.set_ylabel(f"XOR'u öğrenen deneme ({n} seed)")
    ax.set_xlabel("Gizli katmandaki aktivasyon fonksiyonu (çıktı katmanı sigmoid, 1000 epoch)")
    ax.set_title("Aktivasyon fonksiyonuna göre öğrenme")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "activation.png", dpi=140)
    plt.close(fig)


def fig_shape(runs, n):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    neurons = ["1", "2", "3", "4", "8"]
    bars(axes[0], neurons, [learned(runs["neurons" + k]) for k in neurons], n)
    axes[0].set_xlabel("Gizli katmandaki nöron sayısı (1 gizli katman)")
    axes[0].set_ylabel(f"XOR'u öğrenen deneme ({n} seed)")
    axes[0].set_title("Nöron sayısı")
    layers = ["0", "1", "2", "3"]
    bars(axes[1], layers, [learned(runs["layers" + k]) for k in layers], n, ORANGE)
    axes[1].set_xlabel("Gizli katman sayısı (katman başına 2 nöron)")
    axes[1].set_title("Katman sayısı")
    fig.suptitle("Ağın boyutu (alpha 0.8, 1000 epoch)")
    fig.tight_layout()
    fig.savefig(OUT / "network_shape.png", dpi=140)
    plt.close(fig)


def fig_functions():
    z = np.linspace(-6, 6, 600)
    sig = 1 / (1 + np.exp(-z))
    funcs = [("Step", np.where(z < 0, 0, 1.0), np.zeros_like(z), GREY),
             ("Sigmoid", sig, sig * (1 - sig), BLUE),
             ("Tanh", np.tanh(z), 1 - np.tanh(z) ** 2, AQUA),
             ("ReLU", np.maximum(z, 0), (z > 0).astype(float), ORANGE)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for name, f, d, color in funcs:
        axes[0].plot(z, f, color=color, lw=2, label=name)
        axes[1].plot(z, d, color=color, lw=2, label=name)
    axes[0].set_ylim(-1.2, 2.2)
    axes[0].set_title("Aktivasyon fonksiyonları")
    axes[1].set_title("Eğimleri (türev): geri yayılımda hatayı bunlar taşır")
    axes[1].axvspan(-2, 2, color=BLUE, alpha=0.06)
    axes[1].text(0, 1.08, "sigmoid için iyi eğim: −2 … 2", ha="center", fontsize=9, color=INK)
    axes[1].set_ylim(-0.1, 1.2)
    for ax in axes:
        ax.set_xlabel("Toplam (ağırlıklı girdiler − bias)")
        ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "activation_functions.png", dpi=140)
    plt.close(fig)


def summarize(runs):
    out = {"seeds": len(runs["default"]), "conditions": {}}
    for cond, rows in runs.items():
        sse = [float(r["final_sse"]) for r in rows if r["nan"] == "0"]
        first = [int(r["first_epoch_sse_below_0.01"]) for r in rows if r["first_epoch_sse_below_0.01"]]
        good = [float(r["final_sse"]) for r in rows if r["learned"] == "1" and r["nan"] == "0"]
        out["conditions"][cond] = {
            "learned": learned(rows), "nan": sum(int(r["nan"]) for r in rows),
            "median_sse": round(st.median(sse), 4) if sse else None,
            "median_sse_learned": round(st.median(good), 5) if good else None,
            "first_epoch_below_001_mean": round(st.mean(first), 1) if first else None,
            "correct_counts": {k: sum(1 for r in rows if r["correct"] == k) for k in "01234"},
        }
    stuck = [r for r in runs["default"] if r["learned"] == "0"]
    out["default_stuck_example"] = {k: stuck[0][k] for k in ("seed", "final_sse", "out0", "out1", "out2", "out3")} if stuck else None
    (DATA / "summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


def main():
    runs, curves = load()
    n = len(runs["default"])
    fig_alpha(runs, curves, n)
    fig_curves(curves, runs)
    fig_activation(runs, n)
    fig_shape(runs, n)
    fig_functions()
    summarize(runs)
    print("figures ->", OUT)


if __name__ == "__main__":
    main()
