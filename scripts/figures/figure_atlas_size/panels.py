"""Panel functions for the atlas-size sensitivity figure.

- saturation_curves: validation reconstruction quality vs. number of atlas
  genes, for the 24 curated genes alone and for the same pool extended with
  stan, yrt and Tollo, both scored against the full 20-gene validation set.
- per_gene_loo: leave-one-out reconstruction quality per gene, exposing which
  genes the atlas basis represents poorly.
"""

import matplotlib.pyplot as plt
import numpy as np

from scripts.figures import figsize

# Colorblind-friendly RdBu accents.
TRAINING_COLOR = "#b2182b"
VALIDATION_COLOR = "#2166ac"

# One label + color per pool. The curated pool stays black, as the atlas
# does elsewhere in the paper.
POOL_STYLE = {
    "curated": ("24 curated genes", "#000000"),
    "extended": (r"+ $\mathit{stan}$, $\mathit{yrt}$, $\mathit{Tollo}$", "#0072B2"),
}


def saturation_curves(shared_data, output_base):
    """Validation mean Pearson r vs. atlas size, mean +/- s.d. per pool, one axis."""
    fig, ax = plt.subplots(figsize=figsize("1x1"))
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, color="0.8")

    # The two full-panel endpoints sit a few hundredths apart, so they are labelled
    # in different directions to keep the two annotations from overprinting.
    label_offsets = {"curated": (0, 5, "center", "bottom"), "extended": (4, 0, "left", "center")}

    ymax, xmax = 0, 0
    for pool_name, (label, color) in POOL_STYLE.items():
        pool = shared_data["pools"][pool_name]
        sizes = pool["sizes"]
        mean_of_means = np.array([pool["curve_mean"][k].mean() for k in sizes])
        std_of_means = np.array([pool["curve_mean"][k].std() for k in sizes])
        ax.fill_between(sizes, mean_of_means - std_of_means, mean_of_means + std_of_means,
                        color=color, alpha=0.15, linewidth=0)
        ax.plot(sizes, mean_of_means, "-o", color=color, lw=1.4, ms=2.6,
                markeredgewidth=0, label=label)
        # The last point of each sweep is the full panel -- the comparison the figure is for.
        dx, dy, ha, va = label_offsets[pool_name]
        ax.annotate(f"{mean_of_means[-1]:.3f}", xy=(sizes[-1], mean_of_means[-1]),
                    xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                    color=color, fontsize=5)
        ymax = max(ymax, (mean_of_means + std_of_means).max())
        xmax = max(xmax, sizes.max())

    ax.set_xlabel("Number of atlas genes")
    ax.set_ylabel("Validation reconstruction quality\n(mean Pearson $r$)")
    ax.set_xlim(0, xmax + 4)
    ax.set_ylim(0, max(0.8, ymax + 0.05))
    ax.tick_params(direction="out", length=3)
    ax.legend(loc="lower right", frameon=False, fontsize=5, handletextpad=0.6,
              labelspacing=0.3, borderpad=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def sampling_check(shared_data, output_base):
    """Quality vs atlas size, comparing the spread 50 random atlases report with the
    spread over every atlas of that size.

    Both bands are centred on the same curve as ``saturation_curves`` -- the sweep's own
    estimate -- so the only thing that differs between them is width, which is the
    question. The dashed contour needs the whole population, so it exists only where
    C(n, k) was small enough to enumerate; C(n, k) is symmetric, hence the two tails.
    """
    pool = shared_data["pools"]["curated"]
    sizes = pool["sizes"]
    rng = np.random.default_rng(0)

    mean, sd50, ex_sd = [], [], []
    for k in sizes:
        values = pool["curve_mean"][k]
        mean.append(values.mean())
        # Typical width a 50-atlas run reports, as the median over many resamples --
        # a single 50-draw realisation carries +/-20% scatter and would be arbitrary.
        if len(values) >= 50:
            sd50.append(np.median([rng.choice(values, size=50, replace=True).std()
                                   for _ in range(2000)]))
        else:
            sd50.append(values.std())
        ex_sd.append(values.std() if pool["is_exhaustive"][k] and len(values) > 1 else np.nan)

    mean, sd50, ex_sd = np.array(mean), np.array(sd50), np.array(ex_sd)

    # The enumerable sizes are the two tails, because C(n, k) is symmetric. Shade them
    # as contiguous runs so the dashed contour reads as a property of a region rather
    # than as two arbitrary floating segments.
    runs = []
    for k in (k for k in sizes if pool["is_exhaustive"][k]):
        if runs and k == runs[-1][-1] + 1:
            runs[-1].append(k)
        else:
            runs.append([k])

    fig, ax = plt.subplots(figsize=figsize("1x1"))
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, color="0.8")

    for i, run in enumerate(runs):
        ax.axvspan(run[0] - 0.5, run[-1] + 0.5, color="0.93", linewidth=0, zorder=0,
                   label="every atlas enumerated" if i == 0 else None)

    ax.fill_between(sizes, mean - sd50, mean + sd50, color="#0072B2", alpha=0.20,
                    linewidth=0, label="50 random atlases, $\\pm$ s.d.")
    ax.plot(sizes, mean, "-o", color="#0072B2", lw=1.4, ms=2.6, markeredgewidth=0)

    ax.plot(sizes, mean + ex_sd, "--", color="#D55E00", lw=1.0, dashes=(3, 2))
    ax.plot(sizes, mean - ex_sd, "--", color="#D55E00", lw=1.0, dashes=(3, 2),
            label="every atlas, $\\pm$ s.d.")

    ax.set_xlabel("Number of atlas genes")
    ax.set_ylabel("Validation reconstruction quality\n(mean Pearson $r$)")
    ax.set_xlim(0, sizes.max() + 2)
    ax.set_ylim(-0.05, 0.9)
    ax.tick_params(direction="out", length=3)
    ax.legend(loc="lower right", frameon=False, fontsize=5, handletextpad=0.6,
              labelspacing=0.3, borderpad=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def per_gene_loo(shared_data, output_base):
    """Per-gene leave-one-out reconstruction r (atlas = the other 43 genes)."""
    loo = shared_data["loo"]
    colors = [TRAINING_COLOR if p == "training" else VALIDATION_COLOR for p in loo["panel"]]

    fig, ax = plt.subplots(figsize=figsize("2x1"))
    ax.set_axisbelow(True)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, color="0.8")

    y = np.arange(len(loo))
    ax.barh(y, loo["r"].values, height=0.72, color=colors, edgecolor="white", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(loo.index, fontstyle="italic")
    ax.set_ylim(-0.6, len(loo) - 0.4)
    ax.axvline(0, color="black", linewidth=0.6)

    ax.set_xlabel("Leave-one-out reconstruction (Pearson $r$)")
    ax.tick_params(direction="out", length=3)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=TRAINING_COLOR),
        plt.Rectangle((0, 0), 1, 1, color=VALIDATION_COLOR),
    ]
    ax.legend(handles, ["Training gene", "Validation gene"], loc="lower right",
              frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig