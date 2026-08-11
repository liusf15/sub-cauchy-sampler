import argparse
import os
from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jnp
from numpyro.infer import HMC, MCMC
import pandas as pd

from experiments.io import make_output_dirs
from experiments.run_replicates import parse_seed_spec
from experiments.targets import SkewMultivariateStudentT
from src.scp_core import SCP


PLOT_INDICES = [0, 1, 2, 3]
QUANTILE_PROBS = jnp.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99])


def build_target(d, df, alpha_scale):
    loc = jnp.zeros(d)
    scale_tril = jnp.eye(d)
    alpha = jnp.zeros(d)
    alpha = alpha.at[0].set(alpha_scale)
    alpha = alpha.at[1].set(-alpha_scale)
    return SkewMultivariateStudentT(loc, scale_tril, df, alpha)


def initial_point(d, init):
    if init == 'warm':
        return jnp.ones(d)
    return jnp.zeros(d) + 100.0


def quantile_frame(samples, ps, name, indices):
    return pd.DataFrame(
        jnp.quantile(samples[:, indices], ps, axis=0),
        index=ps,
        columns=[f'{name}{idx}' for idx in indices],
    )


def exact_quantiles(target, nexact, exact_seed, ps, indices):
    print(f"Generating {nexact} exact samples once for df={target.df}...")
    exact_samples = target.sample(seed=jax.random.key(exact_seed), sample_shape=(nexact,))
    return quantile_frame(exact_samples, ps, 'Exact', indices)


def run_scp(target, args, seed, x0):
    scp_model = SCP(d=args.d, latitude=args.latitude, affine=args.affine)
    print(f"Initializing SCS parameters with fit seed {seed}...")
    train_start = time.time()
    opt_params, losses = scp_model.minimize_reverse_kl(
        target.log_prob,
        seed=seed,
        ntrain=args.scp_fit_ntrain,
        max_iter=args.scp_fit_max_iter,
        learning_rate=args.scp_fit_learning_rate,
    )
    train_time = time.time() - train_start

    print(f"Running SCS sampling with seed {seed}...")
    start = time.time()
    samples, accept_prob = scp_model.rwm_bright_side(
        target.log_prob,
        opt_params,
        seed=seed,
        x0=scp_model.inverse_projection(opt_params, x0),
        stepsize=args.stepsize,
        nsample=args.nsample,
        burnin=args.burnin,
        thinning=args.thinning,
        algo=args.algo,
    )
    sample_time = time.time() - start
    return samples, {
        'scp_accept_rate': float(accept_prob),
        'scp_time': sample_time,
        'scp_train_time': train_time,
        'scp_final_loss': float(losses[-1]),
        'scp_fit_seed': seed,
    }


def run_hmc(target, args, seed, x0):
    print(f"Running HMC with seed {seed}...")
    start = time.time()
    hmc_kernel = HMC(
        potential_fn=lambda z: -target.log_prob(z),
        step_size=args.hmc_step_size,
        adapt_step_size=False,
        adapt_mass_matrix=False,
        num_steps=args.hmc_num_steps,
        trajectory_length=None,
    )
    mcmc = MCMC(
        hmc_kernel,
        num_warmup=args.burnin,
        num_samples=args.nsample,
        thinning=args.thinning,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(jax.random.key(seed), init_params=x0, extra_fields=("accept_prob",))
    elapsed = time.time() - start
    accept_rate = jnp.mean(mcmc.get_extra_fields()['accept_prob'])
    return mcmc.get_samples(), {
        'hmc_accept_rate': float(accept_rate),
        'hmc_time': elapsed,
        'hmc_step_size': args.hmc_step_size,
        'hmc_num_steps': args.hmc_num_steps,
    }


def output_basename(args, df, seed):
    return (
        f'skewt_df{df}_d{args.d}_lat{args.latitude}_nsample{args.nsample}_burnin{args.burnin}'
        f'_init{args.init}_stepsize{args.stepsize}_{args.algo}_affine{args.affine}_seed{seed}'
    )


def run_one_seed(target, exact_quantiles_df, args, save_dir, df, seed):
    basename = output_basename(args, df, seed)
    result_base = os.path.join(save_dir, basename)
    result_csv = f'{result_base}.csv'
    meta_csv = f'{result_base}_meta.csv'
    if args.resume and os.path.exists(result_csv) and os.path.exists(meta_csv):
        print(f"Skipping existing seed {seed}: {result_csv}")
        return

    x0 = initial_point(args.d, args.init)
    scp_samples, scp_meta = run_scp(target, args, seed, x0)
    hmc_samples, hmc_meta = run_hmc(target, args, seed, x0)

    quantiles = pd.concat(
        [
            exact_quantiles_df,
            quantile_frame(scp_samples, QUANTILE_PROBS, 'SCP', PLOT_INDICES),
            quantile_frame(hmc_samples, QUANTILE_PROBS, 'HMC', PLOT_INDICES),
        ],
        axis=1,
    )
    quantiles.to_csv(result_csv)

    meta = {
        **scp_meta,
        **hmc_meta,
    }
    pd.DataFrame(meta, index=[0]).to_csv(meta_csv, index=False)
    print(f"Saved {result_csv}\n {meta_csv}")


def run(args):
    if args.thinning > args.nsample:
        print(f"Reducing thinning from {args.thinning} to 1 because nsample={args.nsample}.")
        args.thinning = 1
    if args.d < len(PLOT_INDICES):
        raise ValueError("Figure 5 quantile summaries require d >= 4.")

    seeds = parse_seed_spec(args.seeds)
    save_dir, _ = make_output_dirs(args.rootdir, args.plotdir, 'skew_t', args.date)

    for df in args.dfs:
        target = build_target(args.d, df, args.alpha_scale)
        exact_quantiles_df = exact_quantiles(target, args.nexact, args.exact_seed, QUANTILE_PROBS, PLOT_INDICES)
        for seed in seeds:
            print(f"\nRunning df={df:g}, seed={seed}")
            run_one_seed(target, exact_quantiles_df, args, save_dir, df, seed)


def main():
    parser = argparse.ArgumentParser(
        description='Run the skew-t Figure 5 SCS/HMC comparison efficiently over multiple seeds.'
    )
    parser.add_argument('--date', type=str, default='skewt_qq')
    parser.add_argument('--rootdir', type=str, default='results')
    parser.add_argument('--plotdir', type=str, default='plots')
    parser.add_argument('--seeds', type=str, default='0:20')
    parser.add_argument('--dfs', type=float, nargs='+', default=[2.0, 1.0])
    parser.add_argument('--d', type=int, default=100)
    parser.add_argument('--latitude', type=float, default=1.1)
    parser.add_argument('--affine', type=str, default='scalar', choices=['scalar', 'covariance'])
    parser.add_argument('--nsample', type=int, default=500_000)
    parser.add_argument('--burnin', type=int, default=100)
    parser.add_argument('--stepsize', type=float, default=.1)
    parser.add_argument('--thinning', type=int, default=50)
    parser.add_argument('--nexact', type=int, default=10_000_000)
    parser.add_argument('--exact_seed', type=int, default=0)
    parser.add_argument('--algo', type=str, default='stepout', choices=['stepout', 'reject'])
    parser.add_argument('--alpha_scale', type=float, default=100.)
    parser.add_argument('--init', type=str, default='warm', choices=['warm', 'cold'])
    parser.add_argument('--hmc_step_size', type=float, default=.1)
    parser.add_argument('--hmc_num_steps', type=int, default=10)
    parser.add_argument('--scp_fit_ntrain', type=int, default=512)
    parser.add_argument('--scp_fit_max_iter', type=int, default=1000)
    parser.add_argument('--scp_fit_learning_rate', type=float, default=.01)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
