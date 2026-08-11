import argparse
import os
from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from numpyro.infer import HMC, MCMC

from src.scp_core import SCP
from experiments.io import make_output_dirs
from experiments.targets import SkewMultivariateStudentT


def run_scp_single_seed(
    target,
    scp_model,
    opt_params,
    d,
    nsample,
    burnin,
    stepsize,
    thinning,
    seed,
    algo,
    init,
):
    """Run SCP for a single seed."""
    if init == 'warm':
        x0 = jnp.ones(d)
    else:
        x0 = jnp.zeros(d) + 100.0

    start = time.time()
    scp_samples, scp_accept_prob = scp_model.rwm_bright_side(
        target.log_prob,
        opt_params,
        seed=seed,
        x0=scp_model.inverse_projection(opt_params, x0),
        stepsize=stepsize,
        nsample=nsample,
        burnin=burnin,
        thinning=thinning,
        algo=algo,
    )
    scp_time = time.time() - start

    return scp_samples, scp_accept_prob, scp_time


def run_hmc_single_seed(
    target,
    d,
    nsample,
    burnin,
    thinning,
    seed,
    init,
    step_size,
    num_steps,
):
    """Run fixed-step HMC for a single seed."""
    if init == 'warm':
        x0 = jnp.ones(d)
    else:
        x0 = jnp.zeros(d) + 100.0

    start = time.time()
    hmc_kernel = HMC(
        potential_fn=lambda z: -target.log_prob(z),
        step_size=step_size,
        adapt_step_size=False,
        adapt_mass_matrix=False,
        num_steps=num_steps,
        trajectory_length=None,
    )
    mcmc = MCMC(
        hmc_kernel,
        num_warmup=burnin,
        num_samples=nsample,
        thinning=thinning,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(jax.random.key(seed), init_params=x0, extra_fields=("accept_prob",))
    hmc_samples = mcmc.get_samples()
    hmc_time = time.time() - start
    hmc_accept_rate = jnp.mean(mcmc.get_extra_fields()['accept_prob'])
    return hmc_samples, hmc_accept_rate, hmc_time


def run_multiple_seeds(
    d,
    latitude,
    nsample,
    burnin,
    stepsize,
    thinning,
    nseeds,
    algo,
    df,
    alpha_scale,
    init,
    affine,
    nexact,
    include_hmc,
    hmc_step_size,
    hmc_num_steps,
):
    """Run SCP with multiple random seeds and collect tail probability results."""
    # Setup target
    loc = jnp.zeros(d)
    scale_tril = jnp.eye(d)
    alpha = jnp.zeros(d)
    alpha = alpha.at[0].set(alpha_scale)
    alpha = alpha.at[1].set(-alpha_scale)

    target = SkewMultivariateStudentT(loc, scale_tril, df, alpha)
    scp_model = SCP(d=d, latitude=latitude, affine=affine)

    print("Initializing SCP parameters...")
    opt_params, _ = scp_model.minimize_reverse_kl(
        target.log_prob,
        seed=0,
        ntrain=512,
        max_iter=1000,
        learning_rate=0.01,
    )

    # Generate exact samples once
    print(f"Generating {nexact} exact samples...")
    exact_samples = target.sample(seed=jax.random.key(0), sample_shape=(nexact,))
    exact_norms = jnp.linalg.norm(exact_samples, axis=1)

    # Define c values
    c_values = 2**np.arange(4, 16)

    # Compute exact tail probabilities and their variances
    exact_tail_probs = []
    exact_tail_vars = []
    for c in c_values:
        exact_indicators = (exact_norms > c).astype(float)
        exact_tail_prob = jnp.mean(exact_indicators)
        exact_tail_var = jnp.var(exact_indicators) / len(exact_indicators)
        exact_tail_probs.append(float(exact_tail_prob))
        exact_tail_vars.append(float(exact_tail_var))

    # Run SCP for multiple seeds
    all_scp_tail_probs = []
    all_hmc_tail_probs = []
    all_accept_rates = []
    all_hmc_accept_rates = []
    all_times = []
    all_hmc_times = []

    for seed in range(nseeds):
        print(f"\nRunning SCP with seed {seed}/{nseeds-1}...")
        scp_samples, scp_accept_prob, scp_time = run_scp_single_seed(
            target=target,
            scp_model=scp_model,
            opt_params=opt_params,
            d=d,
            nsample=nsample,
            burnin=burnin,
            stepsize=stepsize,
            thinning=thinning,
            seed=seed,
            algo=algo,
            init=init,
        )

        print(f"  SCP acceptance rate: {scp_accept_prob:.4f}, time: {scp_time:.2f}s")
        all_accept_rates.append(float(scp_accept_prob))
        all_times.append(scp_time)

        # Compute tail probabilities for this seed
        scp_norms = jnp.linalg.norm(scp_samples, axis=1)
        scp_tail_probs_seed = []
        for c in c_values:
            scp_tail_prob = jnp.mean(scp_norms > c)
            scp_tail_probs_seed.append(float(scp_tail_prob))

        all_scp_tail_probs.append(scp_tail_probs_seed)

        if include_hmc:
            print(f"Running HMC with seed {seed}/{nseeds-1}...")
            hmc_samples, hmc_accept_rate, hmc_time = run_hmc_single_seed(
                target=target,
                d=d,
                nsample=nsample,
                burnin=burnin,
                thinning=thinning,
                seed=seed,
                init=init,
                step_size=hmc_step_size,
                num_steps=hmc_num_steps,
            )

            print(f"  HMC acceptance rate: {hmc_accept_rate:.4f}, time: {hmc_time:.2f}s")
            all_hmc_accept_rates.append(float(hmc_accept_rate))
            all_hmc_times.append(hmc_time)

            hmc_norms = jnp.linalg.norm(hmc_samples, axis=1)
            hmc_tail_probs_seed = []
            for c in c_values:
                hmc_tail_prob = jnp.mean(hmc_norms > c)
                hmc_tail_probs_seed.append(float(hmc_tail_prob))

            all_hmc_tail_probs.append(hmc_tail_probs_seed)

    # Convert to numpy array for easier manipulation
    all_scp_tail_probs = np.array(all_scp_tail_probs)  # shape: (nseeds, n_c_values)
    all_hmc_tail_probs = np.array(all_hmc_tail_probs) if include_hmc else None

    return {
        'c_values': c_values,
        'exact_tail_probs': exact_tail_probs,
        'exact_tail_vars': exact_tail_vars,
        'scp_tail_probs': all_scp_tail_probs,
        'hmc_tail_probs': all_hmc_tail_probs,
        'accept_rates': all_accept_rates,
        'hmc_accept_rates': all_hmc_accept_rates,
        'times': all_times,
        'hmc_times': all_hmc_times,
    }


def estimate_variance(values):
    if len(values) < 2:
        return np.nan
    return np.var(values, ddof=1)


def compute_variance_statistics(results):
    """Compute variance statistics across chains for each c value."""
    c_values = results['c_values']
    exact_tail_probs = results['exact_tail_probs']
    exact_tail_vars = results['exact_tail_vars']
    scp_tail_probs = results['scp_tail_probs']  # shape: (nseeds, n_c_values)
    hmc_tail_probs = results.get('hmc_tail_probs')

    stats = []
    for i, c in enumerate(c_values):
        exact_tail_prob = exact_tail_probs[i]
        exact_tail_var = exact_tail_vars[i]

        # Get SCP tail probabilities across all seeds for this c
        scp_probs_c = scp_tail_probs[:, i]

        # Compute variance and relative variance (CV) for SCP
        scp_var = estimate_variance(scp_probs_c)
        scp_mean = np.mean(scp_probs_c)
        scp_re = np.sqrt(scp_var) / exact_tail_prob if exact_tail_prob > 0 else np.nan

        row = {
            'c': float(c),
            'exact_tail_prob': exact_tail_prob,
            'exact_tail_var': exact_tail_var,
            'scp_mean': scp_mean,
            'scp_var': scp_var,
            'scp_re': scp_re,
            'scp_bias': scp_mean - exact_tail_prob,
        }

        if hmc_tail_probs is not None:
            hmc_probs_c = hmc_tail_probs[:, i]
            hmc_var = estimate_variance(hmc_probs_c)
            hmc_mean = np.mean(hmc_probs_c)
            hmc_re = np.sqrt(hmc_var) / exact_tail_prob if exact_tail_prob > 0 else np.nan
            row.update({
                'hmc_mean': hmc_mean,
                'hmc_var': hmc_var,
                'hmc_re': hmc_re,
                'hmc_bias': hmc_mean - exact_tail_prob,
            })

        stats.append(row)

    return pd.DataFrame(stats)


def plot_results(
    stats_df,
    plotpath,
    d,
    latitude,
    nsample,
    burnin,
    stepsize,
    df,
    nseeds,
    affine,
    uncertainty,
    min_tail_ratio,
):
    sns.set_theme(style="whitegrid", context='paper', font_scale=1.2)

    def positive(values):
        values = np.asarray(values, dtype=float)
        values[values <= 0.0] = np.nan
        return values

    c_values = stats_df['c'].to_numpy()
    exact = positive(stats_df['exact_tail_prob'])
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3), squeeze=False)
    tail_ax, re_ax = axes.reshape(-1)

    tail_ax.plot(c_values, exact, color='black', linestyle='--', linewidth=1.5, label='Exact')
    tail_has_positive = np.any(np.isfinite(exact))
    re_has_positive = False

    methods = [
        ('scp', 'SCS', 'orangered', 'o', '-.'),
    ]
    if 'hmc_mean' in stats_df.columns:
        methods.append(('hmc', 'HMC', 'steelblue', 's', '-'))

    for prefix, label, color, marker, linestyle in methods:
        mean = positive(stats_df[f'{prefix}_mean'])
        var = stats_df[f'{prefix}_var'].to_numpy(dtype=float)
        sd = np.sqrt(var)
        uncertainty_width = sd if uncertainty == 'sd' else sd / np.sqrt(nseeds)
        lower = np.maximum(mean - uncertainty_width, np.finfo(float).tiny)
        upper = mean + uncertainty_width
        tail_ax.plot(
            c_values,
            mean,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2,
            markersize=5,
            label=label,
        )
        tail_ax.fill_between(c_values, lower, upper, color=color, alpha=0.18, linewidth=0)
        tail_has_positive = tail_has_positive or np.any(np.isfinite(mean))
        re_values = mask_unreliable_relative_error(
            positive(stats_df[f'{prefix}_re']),
            mean,
            exact,
            min_tail_ratio,
        )
        re_has_positive = re_has_positive or np.any(np.isfinite(re_values))
        re_ax.plot(
            c_values,
            re_values,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2,
            markersize=5,
            label=label,
        )

    add_sqrt_reference_line(re_ax, c_values, positive(stats_df['scp_re']))

    tail_ax.set_xlabel('Threshold c')
    tail_ax.set_ylabel('')
    tail_ax.set_xscale('log', base=2)
    if tail_has_positive:
        tail_ax.set_yscale('log', base=2)
        set_log_ylim_from_reference(tail_ax, exact)
    tail_ax.set_title(r'Tail probability $P(||X|| > c)$ estimates')
    tail_ax.legend()

    re_ax.set_xlabel('Threshold c')
    re_ax.set_ylabel('')
    re_ax.set_xscale('log', base=2)
    if re_has_positive:
        re_ax.set_yscale('log', base=2)
    else:
        re_ax.text(
            0.5,
            0.5,
            'Relative error requires at least two seeds',
            ha='center',
            va='center',
            transform=re_ax.transAxes,
        )
    re_ax.set_title('Relative error (SD / exact probability)')
    re_ax.legend()

    plt.tight_layout()

    # Save plot
    plot_filename = (
        f'{plotpath}/figure6_skewt_tailprob_scs_hmc_df{df}_d{d}_lat{latitude}_nsample{nsample}_burnin{burnin}'
        f'_stepsize{stepsize}_affine{affine}_{uncertainty}_minratio{min_tail_ratio}_nseeds{nseeds}.pdf'
    )
    plt.savefig(plot_filename, bbox_inches='tight')
    print(f"\nSaved plot to {plot_filename}")
    plt.close()


def set_log_ylim_from_reference(ax, reference, pad_factor=1.25):
    reference = np.asarray(reference, dtype=float)
    reference = reference[np.isfinite(reference) & (reference > 0.0)]
    if reference.size == 0:
        return
    ymin = reference.min() / pad_factor
    ymax = reference.max() * pad_factor
    if ymin < ymax:
        ax.set_ylim(ymin, ymax)


def mask_unreliable_relative_error(re_values, mean, exact, min_tail_ratio):
    if min_tail_ratio <= 0.0:
        return re_values
    re_values = np.asarray(re_values, dtype=float).copy()
    mean = np.asarray(mean, dtype=float)
    exact = np.asarray(exact, dtype=float)
    ratio = np.divide(
        mean,
        exact,
        out=np.full_like(mean, np.nan, dtype=float),
        where=np.isfinite(exact) & (exact > 0.0),
    )
    re_values[ratio < min_tail_ratio] = np.nan
    return re_values


def add_sqrt_reference_line(ax, c_values, reference_values):
    c_values = np.asarray(c_values, dtype=float)
    reference_values = np.asarray(reference_values, dtype=float)
    mask = np.isfinite(c_values) & np.isfinite(reference_values) & (c_values > 0.0) & (reference_values > 0.0)
    if not np.any(mask):
        return
    c_ref = c_values[mask][len(c_values[mask]) // 2]
    y_ref = reference_values[mask][len(reference_values[mask]) // 2]
    y_values = y_ref * np.sqrt(c_values[mask] / c_ref)
    ax.plot(
        c_values[mask],
        y_values,
        color='0.5',
        linestyle=':',
        linewidth=1.5,
        label=r'$\propto c^{1/2}$',
    )


def build_stats_filename(savepath, args):
    return (
        f'{savepath}/skewt_df{args.df}_d{args.d}_lat{args.latitude}_nsample{args.nsample}_burnin{args.burnin}'
        f'_init{args.init}_stepsize{args.stepsize}_{args.algo}_affine{args.affine}_scp_variance_stats_nseeds{args.nseeds}.csv'
    )


def main(args):
    if args.thinning > args.nsample:
        print(f"Reducing thinning from {args.thinning} to 1 because nsample={args.nsample}.")
        args.thinning = 1

    # Setup output paths
    savepath, plotpath = make_output_dirs(args.rootdir, args.plotdir, 'skew_t', args.date)

    if args.plot_only:
        stats_filename = args.stats_csv or build_stats_filename(savepath, args)
        print(f"Loading existing statistics from {stats_filename}")
        stats_df = pd.read_csv(stats_filename)
        print("Creating plots...")
        plot_results(
            stats_df, plotpath, args.d, args.latitude, args.nsample, args.burnin,
            args.stepsize, args.df, args.nseeds, args.affine, args.uncertainty, args.min_tail_ratio
        )
        print("\nDone!")
        return

    # Run multiple seeds
    print(f"Running SCP with {args.nseeds} different seeds...")
    results = run_multiple_seeds(
        d=args.d,
        latitude=args.latitude,
        nsample=args.nsample,
        burnin=args.burnin,
        stepsize=args.stepsize,
        thinning=args.thinning,
        nseeds=args.nseeds,
        algo=args.algo,
        df=args.df,
        alpha_scale=args.alpha_scale,
        init=args.init,
        affine=args.affine,
        nexact=args.nexact,
        include_hmc=args.include_hmc,
        hmc_step_size=args.hmc_step_size,
        hmc_num_steps=args.hmc_num_steps,
    )

    # Compute variance statistics
    print("\nComputing variance statistics...")
    stats_df = compute_variance_statistics(results)

    # Save statistics
    stats_filename = build_stats_filename(savepath, args)
    stats_df.to_csv(stats_filename, index=False)
    print(f"Saved statistics to {stats_filename}")

    # Print summary
    print("\nVariance Statistics Summary:")
    print(stats_df.to_string())

    # Print acceptance rate statistics
    print(f"\nSCP Acceptance Rate: {np.mean(results['accept_rates']):.4f} ± {np.std(results['accept_rates']):.4f}")
    print(f"SCP Time per chain: {np.mean(results['times']):.2f} ± {np.std(results['times']):.2f} seconds")
    if args.include_hmc:
        print(f"HMC Acceptance Rate: {np.mean(results['hmc_accept_rates']):.4f} ± {np.std(results['hmc_accept_rates']):.4f}")
        print(f"HMC Time per chain: {np.mean(results['hmc_times']):.2f} ± {np.std(results['hmc_times']):.2f} seconds")

    # Create plots
    print("\nCreating plots...")
    plot_results(
        stats_df, plotpath, args.d, args.latitude, args.nsample, args.burnin,
        args.stepsize, args.df, args.nseeds, args.affine, args.uncertainty, args.min_tail_ratio
    )

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run SCP with multiple seeds and analyze variance')
    parser.add_argument('--nseeds', type=int, default=20, help='Number of random seeds to run')
    parser.add_argument('--date', type=str, default='20260507')
    parser.add_argument('--rootdir', type=str, default='results')
    parser.add_argument('--plotdir', type=str, default='plots')
    parser.add_argument('--d', type=int, default=100)
    parser.add_argument('--latitude', type=float, default=1.1)
    parser.add_argument('--affine', type=str, default='scalar', choices=['scalar', 'covariance'])
    parser.add_argument('--nsample', type=int, default=500_000)
    parser.add_argument('--burnin', type=int, default=100)
    parser.add_argument('--stepsize', type=float, default=.1)
    parser.add_argument('--thinning', type=int, default=50)
    parser.add_argument('--nexact', type=int, default=10_000_000)
    parser.add_argument('--algo', type=str, default='stepout', choices=['stepout', 'reject'])
    parser.add_argument('--df', type=float, default=1)
    parser.add_argument('--alpha_scale', type=float, default=100.)
    parser.add_argument('--init', type=str, default='warm', choices=['warm', 'cold'])
    parser.set_defaults(include_hmc=True)
    parser.add_argument('--include_hmc', action='store_true')
    parser.add_argument('--no_hmc', action='store_false', dest='include_hmc')
    parser.add_argument('--hmc_step_size', type=float, default=.1)
    parser.add_argument('--hmc_num_steps', type=int, default=10)
    parser.add_argument('--plot_only', action='store_true', help='Read an existing stats CSV and regenerate only the plot.')
    parser.add_argument('--stats_csv', type=str, default=None, help='Optional path to an existing variance stats CSV.')
    parser.add_argument('--uncertainty', choices=['sd', 'se'], default='sd', help='Uncertainty band for tail-probability estimates.')
    parser.add_argument(
        '--min_tail_ratio',
        type=float,
        default=0.1,
        help='Omit relative-SD points when method mean is below this fraction of the exact tail probability.',
    )
    args = parser.parse_args()

    main(args)
