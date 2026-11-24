"""Matplotlib plot functions (PDF + PNG output)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend — safe on CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#: Two-color sober institutional palette (baseline vs modernized).
_PALETTE = {"baseline": "#7F7F7F", "modernized": "#1F77B4"}


def plot_mos1_per_aor(metrics_df: pd.DataFrame, out: Path) -> None:
    """Per-AoR MOS-1 paired bars (baseline vs modernized)."""
    sub = metrics_df[(metrics_df["metric"] == "MOS-1") & (metrics_df["aor"] != "ALL")]
    if sub.empty:
        return
    pivot = sub.pivot(index="aor", columns="config_name", values="value").sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(pivot.index))
    w = 0.35
    if "baseline" in pivot.columns:
        ax.bar(x - w / 2, pivot["baseline"].values, w, label="Baseline A-29",
               color=_PALETTE["baseline"])
    if "modernized" in pivot.columns:
        ax.bar(x + w / 2, pivot["modernized"].values, w, label="Modernized A-29N",
               color=_PALETTE["modernized"])
    ax.axhline(0.80, color="red", linestyle="--", linewidth=1, label="MOS-1 target = 0.80")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=0)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("MOS-1")
    ax.set_title("MOS-1 per AoR")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_metric_paired_bars(metrics_df: pd.DataFrame, metric: str, out: Path) -> None:
    """Paired baseline vs modernized for a single metric over AoRs."""
    sub = metrics_df[(metrics_df["metric"] == metric) & (metrics_df["aor"] != "ALL")]
    if sub.empty:
        return
    pivot = sub.pivot(index="aor", columns="config_name", values="value").sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(pivot.index))
    w = 0.35
    if "baseline" in pivot.columns:
        ax.bar(x - w / 2, pivot["baseline"].values, w, label="Baseline",
               color=_PALETTE["baseline"])
    if "modernized" in pivot.columns:
        ax.bar(x + w / 2, pivot["modernized"].values, w, label="Modernized",
               color=_PALETTE["modernized"])
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=0)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} per AoR")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_lambda_sweep(sweep_df: pd.DataFrame, metric: str, out: Path) -> None:
    """Mean-with-CI line plot of a metric over λ, paired by configuration.

    The CI uses ±1 sample standard deviation across seeds.
    """
    sub = sweep_df[(sweep_df["metric"] == metric) & (sweep_df["aor"] == "ALL")]
    if sub.empty:
        return
    agg = sub.groupby(["lambda_per_hour", "config_name"])["value"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    for cfg_name, color in _PALETTE.items():
        cur = agg[agg["config_name"] == cfg_name].sort_values("lambda_per_hour")
        if cur.empty:
            continue
        ax.errorbar(cur["lambda_per_hour"], cur["mean"], yerr=cur["std"],
                    label=cfg_name.capitalize(), color=color,
                    marker="o", linewidth=2, capsize=3)
    ax.set_xlabel("λ (UAS / hour)")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs arrival rate")
    ax.legend(loc="best", frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
