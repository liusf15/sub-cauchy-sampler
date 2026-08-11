import argparse
import os
import re
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


BETA_COORD = 1
DEFAULT_DFS = [2.0]
DEFAULT_METHODS = ["Gibbs", "HMC", "IMH", "SCP"]
DEFAULT_PRIOR_SCALE = 2.5

METHOD_SPECS = {
    "Gibbs": {
        "display": "Gibbs",
        "column": "Gibbs",
        "color": "seagreen",
        "marker": "^",
        "linestyle": "--",
    },
    "HMC": {
        "display": "HMC",
        "column": "HMC",
        "color": "steelblue",
        "marker": "s",
        "linestyle": "-",
    },
    "IMH": {
        "display": "IS",
        "column": "IMH",
        "color": "mediumpurple",
        "marker": "D",
        "linestyle": ":",
    },
    "SCP": {
        "display": "SCS",
        "column": "SCP",
        "color": "orangered",
        "marker": "o",
        "linestyle": "-.",
    },
}
METHOD_ALIASES = {
    "gibbs": "Gibbs",
    "hmc": "HMC",
    "imh": "IMH",
    "is": "IMH",
    "scp": "SCP",
    "scs": "SCP",
}
PLOT_SPECS = {
    "figure9": {
        "date": "logistic_t",
        "link": "logistic",
        "filename": "figure9_logistic_student_t_qq_{method_tag}_{uncertainty}.pdf",
    },
    "figure10": {
        "date": "robit_t",
        "link": "robit",
        "filename": "figure10_robit_student_t_qq_{method_tag}_{uncertainty}.pdf",
    },
}


def format_float(value):
    return f"{value:g}"


def normalize_methods(methods):
    if not methods:
        return DEFAULT_METHODS
    normalized = []
    for method in methods:
        key = method.lower()
        if key == "all":
            return DEFAULT_METHODS
        if key not in METHOD_ALIASES:
            raise ValueError(
                f"Unknown method {method!r}. Use Gibbs, HMC, IMH/IS, SCP/SCS, or all."
            )
        canonical = METHOD_ALIASES[key]
        if canonical not in normalized:
            normalized.append(canonical)
    return normalized


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


def padded_limits(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    lower = values.min()
    upper = values.max()
    pad = 0.05 * (upper - lower) if upper > lower else 1.0
    return lower - pad, upper + pad


def quantile_columns(path):
    return list(pd.read_csv(path, index_col=0, nrows=0).columns)


def read_quantiles(path, column):
    df = pd.read_csv(path, index_col=0)
    if column not in df.columns:
        raise KeyError(f"{path} is missing column {column!r}")
    return df[column].to_numpy(dtype=float)


def parse_prior_from_name(path):
    match = re.search(r"_prior_([-+0-9.eE]+)_([-+0-9.eE]+)_", path.name)
    if match is None:
        return None
    return float(match.group(1)), float(match.group(2))


def path_matches_prior(path, df, prior_scale):
    parsed = parse_prior_from_name(path)
    if parsed is None:
        return False
    file_df, file_prior_scale = parsed
    return np.isclose(file_df, df) and np.isclose(file_prior_scale, prior_scale)


def collect_method_paths(result_dir, link, df, prior_scale, method, coord):
    spec = METHOD_SPECS[method]
    column = f"{spec['column']}{coord}"
    paths = [
        path
        for path in sorted(result_dir.glob(f"{link}_*_quantiles.csv"))
        if path_matches_prior(path, df, prior_scale) and column in quantile_columns(path)
    ]
    if not paths:
        raise FileNotFoundError(
            f"No {spec['display']} beta{coord} quantile files found in {result_dir} "
            f"for df={format_float(df)} and prior_scale={format_float(prior_scale)}."
        )
    return paths


def parse_reference_name(path):
    match = re.search(
        r"_reference_n(?P<nsample>\d+)_thin(?P<thinning>\d+)_"
        r"chains(?P<chains>\d+)_nuts_reference\.csv$",
        path.name,
    )
    if match is None:
        return {"nsample": None, "thinning": None, "chains": None}
    return {key: int(value) for key, value in match.groupdict().items()}


def reference_sort_key(path):
    parsed = parse_reference_name(path)
    return (
        parsed["nsample"] or -1,
        parsed["chains"] or -1,
        -(parsed["thinning"] or 0),
        path.stat().st_mtime,
    )


def find_reference_path(result_dir, link, df, prior_scale, coord):
    column = f"NUTS{coord}"
    paths = [
        path
        for path in sorted(result_dir.glob(f"{link}_*_nuts_reference.csv"))
        if path_matches_prior(path, df, prior_scale) and column in quantile_columns(path)
    ]
    if not paths:
        return None
    return max(paths, key=reference_sort_key)


def reference_quantiles(result_dir, link, df, prior_scale, coord, fallback_paths):
    column = f"NUTS{coord}"
    reference_path = find_reference_path(result_dir, link, df, prior_scale, coord)
    if reference_path is not None:
        return read_quantiles(reference_path, column)

    values = []
    for path in fallback_paths:
        if column in quantile_columns(path):
            values.append(read_quantiles(path, column))
    if not values:
        raise FileNotFoundError(
            f"No separate NUTS reference or {column} fallback columns found in {result_dir}."
        )
    return finite_mean(np.vstack(values), axis=0)


def summarize_method(paths, method, coord):
    column = f"{METHOD_SPECS[method]['column']}{coord}"
    values = np.vstack([read_quantiles(path, column) for path in paths])
    sd = (
        np.nanstd(values, axis=0, ddof=1)
        if values.shape[0] > 1
        else np.zeros(values.shape[1])
    )
    return {
        "mean": finite_mean(values, axis=0),
        "sd": sd,
        "se": sd / np.sqrt(values.shape[0]),
        "n_files": values.shape[0],
    }


def axis_limits(reference, mean, band, axis_source):
    if axis_source == "reference":
        return padded_limits(reference)
    return padded_limits(np.concatenate([reference, mean, mean - band, mean + band]))


def plot_method(ax, reference, summary, method, uncertainty, axis_source):
    spec = METHOD_SPECS[method]
    mean = summary["mean"]
    band = summary[uncertainty]

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
            mean - band,
            mean + band,
            color=spec["color"],
            alpha=0.18,
            linewidth=0,
        )

    limits = axis_limits(reference, mean, band, axis_source)
    if limits is not None:
        ax.set_xlim(*limits)
        ax.set_ylim(*limits)
        ax.plot(limits, limits, color="black", linestyle="--", linewidth=1)


def plot_student_t_grid(
    rootdir,
    plotdir,
    plot_key,
    date,
    dfs,
    prior_scale,
    coord,
    methods,
    uncertainty,
    axis_source,
):
    plot_spec = PLOT_SPECS[plot_key]
    result_dir = Path(rootdir) / "regression" / date
    output_dir = Path(plotdir) / "regression" / date
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        len(dfs),
        len(methods),
        figsize=(3 * len(methods), 2.8 * len(dfs)),
        squeeze=False,
    )
    for row, df in enumerate(dfs):
        method_paths = {
            method: collect_method_paths(
                result_dir,
                plot_spec["link"],
                df,
                prior_scale,
                method,
                coord,
            )
            for method in methods
        }
        fallback_paths = next(iter(method_paths.values()))
        reference = reference_quantiles(
            result_dir,
            plot_spec["link"],
            df,
            prior_scale,
            coord,
            fallback_paths,
        )

        for col, method in enumerate(methods):
            ax = axes[row, col]
            summary = summarize_method(method_paths[method], method, coord)
            plot_method(ax, reference, summary, method, uncertainty, axis_source)
            if row == 0:
                ax.set_title(METHOD_SPECS[method]["display"])
            if len(dfs) > 1 and col == 0:
                ax.text(
                    0.04,
                    0.84,
                    f"df={format_float(df)}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                )

    method_tag = "_".join(methods)
    output_path = output_dir / plot_spec["filename"].format(
        method_tag=method_tag,
        uncertainty=uncertainty,
    )
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")
    return output_path


def selected_plot_keys(figure):
    if figure == "all":
        return ["figure9", "figure10"]
    if figure == "logistic_t":
        return ["figure9"]
    if figure == "robit_t":
        return ["figure10"]
    return [figure]


def main():
    parser = argparse.ArgumentParser(
        description="Build logistic-t and robit-t beta1 Q-Q grids from quantile CSVs."
    )
    parser.add_argument("--rootdir", default="results")
    parser.add_argument("--plotdir", default="plots")
    parser.add_argument(
        "--date",
        default=None,
        help=(
            "Regression result subdirectory. If omitted, figure9 uses logistic_t "
            "and figure10 uses robit_t."
        ),
    )
    parser.add_argument(
        "--figure",
        choices=["all", "figure9", "figure10", "logistic_t", "robit_t"],
        default="all",
    )
    parser.add_argument("--dfs", type=float, nargs="+", default=DEFAULT_DFS)
    parser.add_argument("--prior_scale", type=float, default=DEFAULT_PRIOR_SCALE)
    parser.add_argument("--coord", type=int, default=BETA_COORD)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--uncertainty", choices=["sd", "se"], default="sd")
    parser.add_argument("--axis_source", choices=["reference", "all"], default="reference")
    args = parser.parse_args()

    methods = normalize_methods(args.methods)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    for plot_key in selected_plot_keys(args.figure):
        date = args.date if args.date is not None else PLOT_SPECS[plot_key]["date"]
        plot_student_t_grid(
            args.rootdir,
            args.plotdir,
            plot_key,
            date,
            args.dfs,
            args.prior_scale,
            args.coord,
            methods,
            args.uncertainty,
            args.axis_source,
        )


if __name__ == "__main__":
    main()
