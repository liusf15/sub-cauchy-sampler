import argparse
import os
import time
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.scp_core import SCP
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
    all_accept_rates = []
    all_times = []

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

    # Convert to numpy array for easier manipulation
    all_scp_tail_probs = np.array(all_scp_tail_probs)  # shape: (nseeds, n_c_values)

    return {
        'c_values': c_values,
        'exact_tail_probs': exact_tail_probs,
        'exact_tail_vars': exact_tail_vars,
        'scp_tail_probs': all_scp_tail_probs,
        'accept_rates': all_accept_rates,
        'times': all_times,
    }


def compute_variance_statistics(results):
    """Compute variance statistics across chains for each c value."""
    c_values = results['c_values']
    exact_tail_probs = results['exact_tail_probs']
    exact_tail_vars = results['exact_tail_vars']
    scp_tail_probs = results['scp_tail_probs']  # shape: (nseeds, n_c_values)

    stats = []
    for i, c in enumerate(c_values):
        exact_tail_prob = exact_tail_probs[i]
        exact_tail_var = exact_tail_vars[i]

        # Get SCP tail probabilities across all seeds for this c
        scp_probs_c = scp_tail_probs[:, i]

        # Compute variance and relative variance (CV) for SCP
        scp_var = np.var(scp_probs_c, ddof=1)
        scp_mean = np.mean(scp_probs_c)
        scp_re = np.sqrt(scp_var) / exact_tail_prob if exact_tail_prob > 0 else np.nan

        stats.append({
            'c': float(c),
            'exact_tail_prob': exact_tail_prob,
            'exact_tail_var': exact_tail_var,
            'scp_mean': scp_mean,
            'scp_var': scp_var,
            'scp_re': scp_re,
            'scp_bias': scp_mean - exact_tail_prob,
        })

    return pd.DataFrame(stats)


def plot_results(stats_df, savepath, d, latitude, nsample, burnin, stepsize, df, nseeds):
    sns.set_theme(style="whitegrid", context='paper', font_scale=1.2)
    plt.figure(figsize=(4, 3))
    sns.lineplot(x='c', y='scp_re', data=stats_df, marker='o', linewidth=2, markersize=10, color='orangered')
    plt.xlabel('Threshold c')
    plt.ylabel('Relative error (SD / mean)')
    plt.xscale('log', base=2)
    plt.yscale('log', base=2)
    plt.title('Relative error of tail probability estimates')

    # Save plot
    plot_filename = (
        f'{savepath}/skewt_tailprob_re_df{df}_d{d}_lat{latitude}_nsample{nsample}_burnin{burnin}'
        f'_stepsize{stepsize}_nseeds{nseeds}.pdf'
    )
    plt.savefig(plot_filename, bbox_inches='tight')
    print(f"\nSaved plot to {plot_filename}")
    plt.close()


def main(args):
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
    )

    # Setup save path
    savepath = os.path.join(args.rootdir, args.date, 'skewt')
    os.makedirs(savepath, exist_ok=True)

    # Compute variance statistics
    print("\nComputing variance statistics...")
    stats_df = compute_variance_statistics(results)

    # Save statistics
    stats_filename = (
        f'{savepath}/skewt_df{args.df}_d{args.d}_lat{args.latitude}_nsample{args.nsample}_burnin{args.burnin}'
        f'_init{args.init}_stepsize{args.stepsize}_{args.algo}_affine{args.affine}_scp_variance_stats_nseeds{args.nseeds}.csv'
    )
    stats_df.to_csv(stats_filename, index=False)
    print(f"Saved statistics to {stats_filename}")

    # Print summary
    print("\nVariance Statistics Summary:")
    print(stats_df.to_string())

    # Print acceptance rate statistics
    print(f"\nSCP Acceptance Rate: {np.mean(results['accept_rates']):.4f} ± {np.std(results['accept_rates']):.4f}")
    print(f"SCP Time per chain: {np.mean(results['times']):.2f} ± {np.std(results['times']):.2f} seconds")

    # Create plots
    print("\nCreating plots...")
    plot_results(
        stats_df, savepath, args.d, args.latitude, args.nsample, args.burnin,
        args.stepsize, args.df, args.nseeds
    )

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run SCP with multiple seeds and analyze variance')
    parser.add_argument('--nseeds', type=int, default=10, help='Number of random seeds to run')
    parser.add_argument('--date', type=str, default='20260507')
    parser.add_argument('--rootdir', type=str, default='experiments/results')
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
    args = parser.parse_args()

    main(args)
