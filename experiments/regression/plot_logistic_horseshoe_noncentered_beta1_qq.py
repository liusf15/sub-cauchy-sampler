import argparse
import os
import tempfile
from pathlib import Path

cache_dir = Path(tempfile.gettempdir()) / "sub_cauchy_matplotlib"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BETA1_COORD = 1
DEFAULT_DATE = "logistic_horseshoe_noncentered"
DEFAULT_REFERENCE = (
    "logistic_horseshoe_d20_n50_std_True_paramnoncentered_"
    "reference_n500000_thin500_chains20_nuts_reference.csv"
)
DEFAULT_OUTPUT = (
    "logistic_horseshoe_noncentered_beta1_qq_"
    "gibbs_hmc100k_imh_scs500k_nuts_ref_sd.pdf"
)

METHOD_SPECS = {
    "Gibbs": {
        "display": "Gibbs",
        "column": "Gibbs",
        "glob": (
            "logistic_horseshoe_d20_n50_std_True_seed*_"
            "gibbs_n500000_thin50_burnin100_pgalternate_quantiles.csv"
        ),
        "color": "seagreen",
        "marker": "^",
        "linestyle": "--",
    },
    "HMC": {
        "display": "HMC",
        "column": "HMC",
        "glob": (
            "logistic_horseshoe_d20_n50_std_True_paramnoncentered_seed*_"
            "hmc_n100000_thin10_burnin100_quantiles.csv"
        ),
        "color": "steelblue",
        "marker": "s",
        "linestyle": "-",
    },
    "IMH": {
        "display": "IS",
        "column": "IMH",
        "glob": (
            "logistic_horseshoe_d20_n50_std_True_paramnoncentered_seed*_"
            "imh_n500000_thin50_burnin100_stepsize0.01_quantiles.csv"
        ),
        "color": "mediumpurple",
        "marker": "D",
        "linestyle": ":",
    },
    "SCS": {
        "display": "SCS",
        "column": "SCP",
        "glob": (
            "logistic_horseshoe_d20_n50_std_True_paramnoncentered_seed*_"
            "scp_n500000_thin50_burnin100_stepsize0.05_lat1.7_"
            "affinecovariance_clipnone_algostepout_ntrain256_quantiles.csv"
        ),
        "color": "orangered",
        "marker": "o",
        "linestyle": "-.",
    },
}


def finite_mean(values, axis=0):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    counts = finite.sum(axis=axis)
    sums = np.where(finite, values, 0.0).sum(axis=axis)
    return np.divide(
        sums,
        counts,
        out=np.full_like(sums, np.nan, dtype=float),
        where=counts > 0,
    )


def read_quantiles(path, column):
    df = pd.read_csv(path, index_col=0)
    if column not in df.columns:
        raise KeyError(f"{path} is missing column {column!r}")
    return df[column].to_numpy(dtype=float)


def collect_paths(result_dir, method):
    spec = METHOD_SPECS[method]
    paths = sorted(result_dir.glob(spec["glob"]))
    column = f"{spec['column']}{BETA1_COORD}"
    paths = [
        path
        for path in paths
        if column in pd.read_csv(path, index_col=0, nrows=0).columns
    ]
    if not paths:
        raise FileNotFoundError(
            f"No {spec['display']} beta1 quantile files found in {result_dir} "
            f"matching {spec['glob']!r}"
        )
    return paths


def summarize_method(result_dir, method):
    spec = METHOD_SPECS[method]
    paths = collect_paths(result_dir, method)
    column = f"{spec['column']}{BETA1_COORD}"
    values = np.vstack([read_quantiles(path, column) for path in paths])
    sd = (
        np.nanstd(values, axis=0, ddof=1)
        if values.shape[0] > 1
        else np.zeros(values.shape[1])
    )
    return {
        "mean": finite_mean(values, axis=0),
        "sd": sd,
        "n_files": len(paths),
    }


def padded_limits(values):
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    lower = values.min()
    upper = values.max()
    pad = 0.05 * (upper - lower) if upper > lower else 1.0
    return lower - pad, upper + pad


def plot_method(ax, reference, result_dir, method):
    spec = METHOD_SPECS[method]
    summary = summarize_method(result_dir, method)
    mean = summary["mean"]
    sd = summary["sd"]

    ax.plot(
        reference,
        mean,
        color=spec["color"],
        marker=spec["marker"],
        linestyle=spec["linestyle"],
        linewidth=1.5,
        markersize=4,
    )
    if summary["n_files"] > 1:
        ax.fill_between(
            reference,
            mean - sd,
            mean + sd,
            color=spec["color"],
            alpha=0.18,
            linewidth=0,
        )

    limits = padded_limits(reference)
    if limits is not None:
        ax.set_xlim(*limits)
        ax.set_ylim(*limits)
        ax.plot(limits, limits, color="black", linestyle="--", linewidth=1)

    ax.set_title(spec["display"])


def main():
    parser = argparse.ArgumentParser(
        description="Plot beta1 Q-Q panels for logistic horseshoe noncentered results."
    )
    parser.add_argument("--rootdir", default="results")
    parser.add_argument("--plotdir", default="plots")
    parser.add_argument("--date", default=DEFAULT_DATE)
    parser.add_argument("--reference", default=DEFAULT_REFERENCE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    result_dir = Path(args.rootdir) / "regression" / args.date
    plot_dir = Path(args.plotdir) / "regression" / args.date
    reference_path = Path(args.reference)
    if not reference_path.is_absolute():
        reference_path = result_dir / reference_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = plot_dir / output_path

    reference_column = f"NUTS{BETA1_COORD}"
    reference = read_quantiles(reference_path, reference_column)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    methods = ["Gibbs", "HMC", "IMH", "SCS"]
    fig, axes = plt.subplots(
        1,
        len(methods),
        figsize=(3 * len(methods), 2.8),
        squeeze=False,
    )
    for ax, method in zip(axes[0], methods):
        plot_method(ax, reference, result_dir, method)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
