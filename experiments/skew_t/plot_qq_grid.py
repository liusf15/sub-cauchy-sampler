import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


METHODS = [
    ('SCP', 'SCS', 'orangered', 'o', '-.'),
    ('HMC', 'HMC', 'steelblue', 's', '-'),
]

FILENAME_RE = re.compile(
    r"skewt_df(?P<df>[-+0-9.eE]+)_d(?P<d>\d+)_lat(?P<latitude>[-+0-9.eE]+)"
    r"_nsample(?P<nsample>\d+)_burnin(?P<burnin>\d+)_init(?P<init>[^_]+)"
    r"_stepsize(?P<stepsize>[-+0-9.eE]+)_(?P<algo>[^_]+)_affine(?P<affine>[^_]+)"
    r"_seed(?P<seed>\d+)\.csv$"
)


def parse_metadata(path):
    match = FILENAME_RE.match(path.name)
    if match is None:
        return None
    metadata = match.groupdict()
    metadata['df'] = float(metadata['df'])
    metadata['d'] = int(metadata['d'])
    metadata['latitude'] = float(metadata['latitude'])
    metadata['nsample'] = int(metadata['nsample'])
    metadata['burnin'] = int(metadata['burnin'])
    metadata['stepsize'] = float(metadata['stepsize'])
    metadata['seed'] = int(metadata['seed'])
    return metadata


def matches(metadata, args, df):
    if metadata is None:
        return False
    checks = [
        metadata['affine'] == args.affine,
        metadata['d'] == args.d,
        metadata['nsample'] == args.nsample,
        metadata['burnin'] == args.burnin,
        metadata['init'] == args.init,
        metadata['algo'] == args.algo,
        np.isclose(metadata['df'], df),
        np.isclose(metadata['latitude'], args.latitude),
        np.isclose(metadata['stepsize'], args.stepsize),
    ]
    return all(checks)


def collect_paths(result_dir, args, df):
    paths = []
    for path in sorted(result_dir.glob(f"skewt_*_affine{args.affine}_seed*.csv")):
        metadata = parse_metadata(path)
        if matches(metadata, args, df):
            paths.append((metadata['seed'], path))
    return [path for _, path in sorted(paths)]


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


def summarize_quantiles(paths, method, coord):
    x_values = []
    y_values = []
    for path in paths:
        df = pd.read_csv(path, index_col=0)
        exact_col = f'Exact{coord}'
        method_col = f'{method}{coord}'
        missing = [col for col in (exact_col, method_col) if col not in df.columns]
        if missing:
            raise KeyError(f'{path} is missing column(s): {missing}')
        x_values.append(df[exact_col].to_numpy(dtype=float))
        y_values.append(df[method_col].to_numpy(dtype=float))

    x_values = np.vstack(x_values)
    y_values = np.vstack(y_values)
    y_sd = np.zeros(y_values.shape[1])
    if len(paths) > 1:
        y_sd = np.nanstd(y_values, axis=0, ddof=1)

    return {
        'x_mean': finite_mean(x_values, axis=0),
        'y_mean': finite_mean(y_values, axis=0),
        'y_sd': y_sd,
    }


def set_equal_limits(ax, values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    lower = values.min()
    upper = values.max()
    pad = 0.05 * (upper - lower) if upper > lower else 1.0
    ax.set_xlim(lower - pad, upper + pad)
    ax.set_ylim(lower - pad, upper + pad)


def plot_coordinate(ax, paths, coord):
    exact_values = None
    for method, label, color, marker, linestyle in METHODS:
        summary = summarize_quantiles(paths, method, coord)
        x = summary['x_mean']
        y = summary['y_mean']
        y_sd = summary['y_sd']
        if exact_values is None:
            exact_values = x
        ax.plot(
            x,
            y,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.5,
            markersize=4,
            label=label,
        )
        if len(paths) > 1:
            ax.fill_between(x, y - y_sd, y + y_sd, color=color, alpha=0.18, linewidth=0)

    set_equal_limits(ax, exact_values)
    left, right = ax.get_xlim()
    ax.plot([left, right], [left, right], color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('')
    ax.set_ylabel('')


def make_output_name(args, nseeds_by_df):
    dfs = '_'.join(f'{df:g}' for df in args.dfs)
    nseeds = '_'.join(str(nseeds_by_df[df]) for df in args.dfs)
    return (
        f'figure5_skewt_qq_scs_hmc_dfs{dfs}_d{args.d}_lat{args.latitude:g}'
        f'_nsample{args.nsample}_burnin{args.burnin}_init{args.init}'
        f'_stepsize{args.stepsize:g}_{args.algo}_affine{args.affine}_nseeds{nseeds}.pdf'
    )


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate skew-t per-seed quantiles into a Figure 5-style QQ plot.'
    )
    parser.add_argument('--date', type=str, default='skewt_qq')
    parser.add_argument('--rootdir', type=str, default='results')
    parser.add_argument('--plotdir', type=str, default='plots')
    parser.add_argument('--affine', choices=['scalar', 'covariance'], default='scalar')
    parser.add_argument('--coords', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--d', type=int, default=100)
    parser.add_argument('--df', type=float, default=1)
    parser.add_argument('--dfs', type=float, nargs='+', default=[2.0, 1.0])
    parser.add_argument('--latitude', type=float, default=1.1)
    parser.add_argument('--nsample', type=int, default=500_000)
    parser.add_argument('--burnin', type=int, default=100)
    parser.add_argument('--stepsize', type=float, default=.1)
    parser.add_argument('--init', choices=['warm', 'cold'], default='warm')
    parser.add_argument('--algo', choices=['stepout', 'reject'], default='stepout')
    args = parser.parse_args()
    if args.dfs is None:
        args.dfs = [args.df]

    result_dir = Path(args.rootdir) / 'skew_t' / args.date
    plot_dir = Path(args.plotdir) / 'skew_t' / args.date
    plot_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context='paper', font_scale=1.2)
    paths_by_df = {df: collect_paths(result_dir, args, df) for df in args.dfs}
    missing = [f'{df:g}' for df, paths in paths_by_df.items() if not paths]
    if missing:
        raise FileNotFoundError(
            f'No matching skew-t quantile CSVs found for df(s): {", ".join(missing)} in {result_dir}'
        )

    fig, axes = plt.subplots(
        len(args.dfs),
        len(args.coords),
        figsize=(2.7 * len(args.coords), 2.7 * len(args.dfs)),
        squeeze=False,
    )
    for row, df in enumerate(args.dfs):
        paths = paths_by_df[df]
        for col, coord in enumerate(args.coords):
            ax = axes[row, col]
            plot_coordinate(ax, paths, coord)
            ax.set_title(rf'$x_{{{coord + 1}}}$ ($\nu={df:g}$)')

    axes.reshape(-1)[0].legend()
    plt.tight_layout()

    nseeds_by_df = {df: len(paths) for df, paths in paths_by_df.items()}
    output_path = plot_dir / make_output_name(args, nseeds_by_df)
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {output_path}')


if __name__ == '__main__':
    main()
