import argparse
import gc
from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
from tqdm import trange 
import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import MCMC, NUTS, HMC
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from polyagamma import random_polyagamma

from src.cauchy_mh import independent_cauchy_mh
from experiments.io import make_output_dirs
from experiments.run_replicates import parse_seed_spec
from src.scp_core import SCP
from experiments.targets import LogisticRegression

numpyro.set_host_device_count(20)

def run_nuts(target, seed, nsample, burnin, thinning, nchains, progress_bar=True):
    d = target.d

    print("Running NUTS......")
    start = time.time()
    kernel = NUTS(potential_fn=lambda z: -target.log_prob(z))

    mcmc = MCMC(
        kernel,
        num_warmup=burnin,
        num_samples=nsample,
        num_chains=nchains,
        thinning=thinning,
        progress_bar=progress_bar,
    )
    init_params = jnp.zeros(d) if nchains == 1 else jnp.zeros((nchains, d))
    mcmc.run(jax.random.key(seed), init_params=init_params, extra_fields=("accept_prob",))
    nuts_samples = mcmc.get_samples()
    elapsed = time.time() - start
    accept_rate = jnp.mean(mcmc.get_extra_fields()['accept_prob'])
    print("NUTS acceptance rate:", accept_rate, 'Time:', elapsed)
    return nuts_samples, {
        'accept_rate': float(accept_rate),
        'time': elapsed,
    }

def run_scp(target, latitude, affine, seed, stepsize, nsample, burnin, thinning):
    d = target.d
    scp_model = SCP(d=d, latitude=latitude, affine=affine)
    print("Initializing SCP parameters......")
    train_start = time.time()
    opt_params, losses = scp_model.minimize_reverse_kl(
        target.log_prob,
        seed=seed,
        ntrain=256,
        max_iter=1000,
        learning_rate=0.01,
        clip_value=100.,
    )
    train_time = time.time() - train_start

    print("Running RWM on the bright side......")
    start = time.time()
    scp_samples, scp_accept_prob = scp_model.rwm_bright_side(target.log_prob,
                                                                opt_params, 
                                                                seed=seed, 
                                                                stepsize=stepsize,
                                                                nsample=nsample, 
                                                                burnin=burnin,
                                                                thinning=thinning,
                                                                algo='stepout')
    scp_time = time.time() - start
    print('SCP acceptance rate:', scp_accept_prob, 'Time:', scp_time)
    return scp_samples, {
        'accept_rate': float(scp_accept_prob),
        'time': scp_time,
        'train_time': train_time,
        'final_loss': float(losses[-1]),
    }

def run_hmc(target, seed, nsample, burnin, thinning):
    print("Running HMC......")
    start = time.time()
    hmc_kernel = HMC(potential_fn=lambda z: -target.log_prob(z), step_size=0.5, adapt_step_size=True, adapt_mass_matrix=False, num_steps=10, trajectory_length=None)
    mcmc = MCMC(hmc_kernel, num_warmup=burnin, num_samples=nsample, thinning=thinning, num_chains=1)
    mcmc.run(jax.random.key(seed), init_params=jnp.zeros(target.d), extra_fields=("accept_prob",))
    hmc_samples = mcmc.get_samples()
    hmc_time = time.time() - start
    accept_rate = jnp.mean(mcmc.get_extra_fields()['accept_prob'])
    print("HMC acceptance rate:", accept_rate, 'Time:', hmc_time)
    return hmc_samples, {
        'accept_rate': float(accept_rate),
        'time': hmc_time,
    }


def run_imh(target, seed, stepsize, nsample, burnin, thinning):
    print("Running independent Cauchy MH......")
    start = time.time()
    samples, accept_rate = independent_cauchy_mh(
        target.log_prob,
        jnp.zeros(target.d),
        jax.random.key(seed),
        nsample=nsample,
        burnin=burnin,
        thinning=thinning,
        stepsize=stepsize,
    )
    elapsed = time.time() - start
    print("Independent Cauchy MH acceptance rate:", accept_rate, "Time:", elapsed)
    return samples, {
        'accept_rate': float(accept_rate),
        'time': elapsed,
        'stepsize': stepsize,
    }


def check_finite(name, values, iteration):
    values = np.asarray(values)
    if not np.all(np.isfinite(values)):
        finite_values = values[np.isfinite(values)]
        max_abs = np.max(np.abs(finite_values)) if finite_values.size else np.nan
        raise FloatingPointError(
            f'Non-finite {name} at Gibbs iteration {iteration}; '
            f'max finite absolute value before failure: {max_abs:.6g}'
        )


def run_gibbs(target, seed, nsample, burnin, thinning, pg_method='alternate'):
    print("Running Gibbs......")
    rng = np.random.default_rng(seed)
    X = np.asarray(target.X, dtype=float)
    y = np.asarray(target.y, dtype=float)
    tau = target.prior_scale
    nu = target.prior_df

    n, p = X.shape
    kappa = y - 0.5

    beta = np.zeros(p)
    lam = np.ones(p)

    total_iters = burnin + nsample
    beta_samples = np.zeros((nsample // thinning, p))

    start = time.time()
    sample_idx = 0
    max_abs_eta = 0.0
    min_omega = np.inf
    max_omega = 0.0
    for t in trange(total_iters):
        # 1. Sample omega | beta
        eta = X @ beta              # (n,)
        check_finite('eta', eta, t)
        max_abs_eta = max(max_abs_eta, float(np.max(np.abs(eta))))
        # polyagamma expects shape and "z", here shape=1
        omega = random_polyagamma(
            np.ones(n),
            eta,
            random_state=rng,
            method=pg_method,
        )  # omega ~ PG(1, eta)
        check_finite('Polya-Gamma omega', omega, t)
        min_omega = min(min_omega, float(np.min(omega)))
        max_omega = max(max_omega, float(np.max(omega)))

        # 2. Sample beta | omega, lambda, y
        # prior precision diag(lam / tau^2)
        prior_prec = lam / (tau ** 2)   # (p,)
        check_finite('prior precision', prior_prec, t)

        # X^T Omega X
        WX = X * omega[:, None]        # (n, p)
        XtWX = X.T @ WX                # (p, p)
        check_finite('X^T Omega X', XtWX, t)

        P = XtWX + np.diag(prior_prec)  # precision matrix
        check_finite('Gibbs precision matrix', P, t)
        L = np.linalg.cholesky(P)

        rhs = X.T @ kappa

        # solve P^{-1} rhs via Cholesky
        m = np.linalg.solve(L.T, np.linalg.solve(L, rhs))

        # sample from N(m, P^{-1})
        z = rng.normal(size=p)
        v = np.linalg.solve(L.T, z)
        beta = m + v
        check_finite('beta', beta, t)

        # 3. Sample lambda_j | beta_j
        shape_post = 0.5 * (nu + 1.0)
        rate_post = 0.5 * (nu + (beta ** 2) / (tau ** 2))
        check_finite('lambda rate', rate_post, t)
        lam = rng.gamma(shape=shape_post) / rate_post
        check_finite('lambda', lam, t)

        if t >= burnin and (t - burnin) % thinning == 0:
            beta_samples[sample_idx] = beta
            sample_idx += 1

    print(jnp.median(beta_samples, axis=0))
    
    time_elapsed = time.time() - start
    print("Gibbs sampling time:", time_elapsed)
    return jnp.asarray(beta_samples), {
        'accept_rate': 1.0,
        'time': time_elapsed,
        'pg_method': pg_method,
        'max_abs_eta': max_abs_eta,
        'min_omega': min_omega,
        'max_omega': max_omega,
    }


PLOT_COORD = 1
PLOT_METHODS = [
    ('Gibbs', 'Gibbs'),
    ('HMC', 'HMC'),
    ('IMH', 'IS'),
    ('SCP', 'SCS'),
]
METHOD_ALIASES = {
    'gibbs': 'Gibbs',
    'hmc': 'HMC',
    'imh': 'IMH',
    'is': 'IMH',
    'scp': 'SCP',
    'scs': 'SCP',
}
DEFAULT_METHODS = ['Gibbs', 'HMC', 'IMH', 'SCP']
PS = jnp.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99])


def normalize_methods(methods):
    if not methods:
        return DEFAULT_METHODS
    normalized = []
    for method in methods:
        key = method.lower()
        if key == 'all':
            return DEFAULT_METHODS
        if key not in METHOD_ALIASES:
            raise ValueError(f'Unknown method {method!r}. Use Gibbs, HMC, IMH/IS, SCP/SCS, or all.')
        canonical = METHOD_ALIASES[key]
        if canonical not in normalized:
            normalized.append(canonical)
    return normalized


def build_design(n, d, data_seed):
    X = jax.random.normal(jax.random.PRNGKey(data_seed), (n, d-1))
    X -= jnp.mean(X, axis=0)
    X /= jnp.std(X, axis=0)
    X *= 0.5
    X = jnp.hstack([jnp.ones((n, 1)), X])
    y = X[:, 1] > 0.
    return X, y


def make_quantiles(samples, ps, name, coord=None):
    samples = jnp.asarray(samples)
    ps = jnp.asarray(ps)
    index = np.round(np.asarray(ps, dtype=float), 6)
    if coord is not None:
        samples = samples[:, coord]
        values = jnp.quantile(samples, ps, axis=0)
        columns = [f'{name}{coord}']
    else:
        values = jnp.quantile(samples, ps, axis=0)
        columns = [f'{name}{i}' for i in range(samples.shape[-1])]
    return pd.DataFrame(
        values,
        index=index,
        columns=columns,
    )


def save_comparison_plots(samples_by_method, quantiles_by_method, filename, d, args):
    coord = PLOT_COORD
    fig, ax = plt.subplots(1, len(PLOT_METHODS), figsize=(3 * len(PLOT_METHODS), 3), squeeze=False)
    flat_axes = ax.reshape(-1)
    nuts_quantiles = quantiles_by_method['NUTS']
    xref = nuts_quantiles.iloc[:, 0]
    for plot_ax, (method, label) in zip(flat_axes, PLOT_METHODS):
        quantiles = quantiles_by_method[method]
        plot_ax.plot(xref, quantiles.iloc[:, 0], marker='o')
        plot_ax.plot(xref, xref, 'r--')
        plot_ax.set_title(label)
        plot_ax.set_xlabel(f'NUTS beta{coord}')
        plot_ax.set_ylabel(f'{label} beta{coord}')
    fig.suptitle(
        f'QQ plot for beta{coord}: logistic d={d}, n={args.n}, stepsize={args.stepsize}, affine={args.affine}',
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(f'{filename}_qq.pdf')
    print(f"Saved QQ plot to {filename}_qq.pdf")
    plt.close()

    fig, ax = plt.subplots(1, len(PLOT_METHODS), figsize=(3 * len(PLOT_METHODS), 3), squeeze=False)
    flat_axes = ax.reshape(-1)
    for plot_ax, (method, label) in zip(flat_axes, PLOT_METHODS):
        samples = samples_by_method[method]
        plot_ax.plot(samples[:, coord])
        plot_ax.set_title(label)
        plot_ax.set_xlabel('iteration')
        plot_ax.set_ylabel(f'beta{coord}')
    fig.suptitle(f'Trace plot for beta{coord}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{filename}_trace.pdf')
    print(f"Saved trace plot to {filename}_trace.pdf")
    plt.close()


def print_acceptance_summary(meta_by_method):
    print("\nAcceptance rate summary:")
    labels = {
        'nuts': 'NUTS',
        'nuts_reference': 'NUTS',
        'gibbs': 'Gibbs',
        'hmc': 'HMC',
        'imh': 'IS',
        'scp': 'SCS',
    }
    for method in ['nuts_reference', 'nuts', 'gibbs', 'hmc', 'imh', 'scp']:
        if method not in meta_by_method:
            continue
        accept_rate = float(meta_by_method[method]['accept_rate'])
        print(f"  {labels[method]:<5} {accept_rate:.6g}")

def filename_prefix(args, prior_df):
    return (
        f'logistic_d{args.d}_n{args.n}_std_True_prior_{prior_df}_{args.prior_scale}'
        f'_affine{args.affine}'
    )


def reference_prefix(args, prior_df):
    return f'logistic_d{args.d}_n{args.n}_std_True_prior_{prior_df}_{args.prior_scale}'


def reference_base(result_dir, args, prior_df):
    return result_dir / (
        f'{reference_prefix(args, prior_df)}_reference_n{args.reference_nsample}'
        f'_thin{args.reference_thinning}_chains{args.reference_chains}'
    )


def legacy_reference_glob(result_dir, args, prior_df):
    return (
        f'{reference_prefix(args, prior_df)}_affine*_reference_n{args.reference_nsample}'
        f'_thin{args.reference_thinning}_chains{args.reference_chains}_nuts_reference.csv'
    )


def seed_base(result_dir, args, prior_df, seed):
    return result_dir / (
        f'{filename_prefix(args, prior_df)}_lat{args.latitude}_stepsize{args.stepsize}'
        f'_n{args.nsample}_hmc_n{args.hmc_nsample}_seed{seed}'
    )


def load_or_compute_reference(target, result_dir, args, prior_df):
    base = reference_base(result_dir, args, prior_df)
    quantiles_path = Path(f'{base}_nuts_reference.csv')
    meta_path = Path(f'{base}_nuts_reference_meta.csv')

    if args.resume and quantiles_path.exists():
        print(f'Using cached NUTS reference quantiles: {quantiles_path}')
        quantiles = pd.read_csv(quantiles_path, index_col=0)
        meta = {}
        if meta_path.exists():
            meta = pd.read_csv(meta_path, index_col=0).iloc[0].to_dict()
        return quantiles, meta

    if args.resume:
        legacy_paths = sorted(result_dir.glob(legacy_reference_glob(result_dir, args, prior_df)))
        if legacy_paths:
            legacy_quantiles_path = legacy_paths[0]
            legacy_meta_path = Path(str(legacy_quantiles_path).replace('_nuts_reference.csv', '_nuts_reference_meta.csv'))
            print(f'Using legacy affine-specific NUTS reference quantiles: {legacy_quantiles_path}')
            quantiles = pd.read_csv(legacy_quantiles_path, index_col=0)
            quantiles.to_csv(quantiles_path)
            meta = {}
            if legacy_meta_path.exists():
                meta_frame = pd.read_csv(legacy_meta_path, index_col=0)
                meta_frame.to_csv(meta_path)
                meta = meta_frame.iloc[0].to_dict()
            print(f'saved affine-free NUTS reference quantiles to {quantiles_path}')
            return quantiles, meta

    nuts_samples, nuts_meta = run_nuts(
        target,
        seed=args.reference_seed,
        nsample=args.reference_nsample,
        burnin=args.reference_burnin,
        thinning=args.reference_thinning,
        nchains=args.reference_chains,
        progress_bar=args.nuts_progress_bar,
    )
    quantiles = make_quantiles(nuts_samples, PS, 'NUTS', coord=PLOT_COORD)
    quantiles.to_csv(quantiles_path)
    pd.DataFrame([nuts_meta]).to_csv(meta_path)
    print("saved NUTS reference quantiles to", quantiles_path)
    del nuts_samples
    gc.collect()
    jax.clear_caches()
    return quantiles, nuts_meta


def run_method(target, method, seed, args):
    if method == 'Gibbs':
        return run_gibbs(
            target,
            seed=seed,
            nsample=args.nsample,
            burnin=args.burnin,
            thinning=args.thinning,
            pg_method=args.pg_method,
        )
    if method == 'HMC':
        return run_hmc(
            target,
            seed=seed,
            nsample=args.hmc_nsample,
            burnin=args.burnin,
            thinning=args.hmc_thinning,
        )
    if method == 'IMH':
        return run_imh(
            target,
            seed=seed,
            stepsize=args.is_stepsize,
            nsample=args.nsample,
            burnin=args.burnin,
            thinning=args.thinning,
        )
    if method == 'SCP':
        return run_scp(
            target,
            latitude=args.latitude,
            affine=args.affine,
            seed=seed,
            stepsize=args.stepsize,
            nsample=args.nsample,
            burnin=args.burnin,
            thinning=args.thinning,
        )
    raise ValueError(f'Unknown method: {method}')


def run_prior_df(args, result_dir, prior_df, seeds, methods):
    X, y = build_design(args.n, args.d, args.data_seed)
    target = LogisticRegression(X, y, prior_df=prior_df, prior_scale=args.prior_scale)
    nuts_quantiles, nuts_meta = load_or_compute_reference(target, result_dir, args, prior_df)

    for seed in seeds:
        base = seed_base(result_dir, args, prior_df, seed)
        quantiles_path = Path(f'{base}_quantiles.csv')
        meta_path = Path(f'{base}_meta.csv')
        if args.resume and quantiles_path.exists() and meta_path.exists():
            print(f'Skipping existing seed result: {quantiles_path}')
            continue

        print(f'Running logistic+t prior_df={prior_df}, seed={seed}, methods={methods}')
        quantiles_by_method = {'NUTS': nuts_quantiles}
        meta_by_method = {'nuts_reference': nuts_meta}
        if args.save_samples:
            sample_dir = Path(args.rootdir) / 'regression' / args.date / 'samples'
            sample_dir.mkdir(parents=True, exist_ok=True)

        for method in methods:
            samples, meta = run_method(target, method, seed, args)
            quantiles_by_method[method] = make_quantiles(samples, PS, method, coord=PLOT_COORD)
            meta_by_method[method.lower()] = meta
            if args.save_samples:
                sample_path = sample_dir / f'{base.name}_{method}_samples.csv'
                pd.DataFrame(np.asarray(samples)).to_csv(sample_path, index=False)
                print("saved samples to", sample_path)
            del samples
            gc.collect()
            jax.clear_caches()

        pd.concat(list(quantiles_by_method.values()), axis=1).to_csv(quantiles_path)
        pd.DataFrame(meta_by_method).T.to_csv(meta_path)
        print("saved quantiles to", quantiles_path)
        print("saved meta info to", meta_path)
        print_acceptance_summary(meta_by_method)


def run(args):
    result_dir, _ = make_output_dirs(args.rootdir, args.plotdir, 'regression', args.date)
    result_dir = Path(result_dir)
    methods = normalize_methods(args.methods)
    if args.dfs is None:
        dfs = [args.prior_df]
    else:
        dfs = args.dfs
    seeds = parse_seed_spec(args.seeds) if args.seeds is not None else [args.seed]

    for prior_df in dfs:
        run_prior_df(args, result_dir, prior_df, seeds, methods)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default='logistic_t')
    parser.add_argument('--rootdir', type=str, default='results')
    parser.add_argument('--plotdir', type=str, default='plots')
    parser.add_argument('--d', type=int, default=20)
    parser.add_argument('--n', type=int, default=50)
    parser.add_argument('--data_seed', type=int, default=2025)
    parser.add_argument('--prior_df', type=float, default=2.0)
    parser.add_argument('--dfs', type=float, nargs='+', default=None)
    parser.add_argument('--prior_scale', type=float, default=2.5)
    parser.add_argument('--latitude', type=float, default=1.1)
    parser.add_argument('--affine', type=str, default='covariance', choices=['scalar', 'covariance'])
    parser.add_argument('--nsample', type=int, default=500_000)
    parser.add_argument('--burnin', type=int, default=100)
    parser.add_argument('--stepsize', type=float, default=.02)
    parser.add_argument('--is_stepsize', type=float, default=.01)
    parser.add_argument('--thinning', type=int, default=50)
    parser.add_argument('--hmc_nsample', type=int, default=100_000)
    parser.add_argument('--hmc_thinning', type=int, default=10)
    parser.add_argument('--reference_nsample', type=int, default=5_000_000)
    parser.add_argument('--reference_burnin', type=int, default=100)
    parser.add_argument('--reference_thinning', type=int, default=500)
    parser.add_argument('--reference_chains', type=int, default=20)
    parser.add_argument('--reference_seed', type=int, default=0)
    parser.add_argument('--nuts_progress_bar', dest='nuts_progress_bar', action='store_true', default=True)
    parser.add_argument('--no_nuts_progress_bar', dest='nuts_progress_bar', action='store_false')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seeds', type=str, default=None)
    parser.add_argument('--methods', nargs='+', default=DEFAULT_METHODS)
    parser.add_argument(
        '--pg_method',
        choices=['alternate', 'devroye', 'gamma', 'saddle'],
        default='alternate',
        help='Polya-Gamma sampler used by the logistic Gibbs update.',
    )
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--save_samples', action='store_true')

    args = parser.parse_args()

    run(args)
