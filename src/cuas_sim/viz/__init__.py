"""Static matplotlib plots for paired runs and sweeps.

Designed for the figures of the journal extension (heatmaps, paired bars,
MOE-1/2/3 surfaces). All functions take a path argument and save the figure
without a pop-up backend.
"""

from cuas_sim.viz.plots import (
    plot_lambda_sweep,
    plot_metric_paired_bars,
    plot_mos1_per_aor,
)

__all__ = [
    "plot_lambda_sweep",
    "plot_metric_paired_bars",
    "plot_mos1_per_aor",
]
