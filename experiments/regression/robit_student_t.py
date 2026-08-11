import argparse
import gc
from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr, ndtri
import numpyro
from numpyro.infer import MCMC, NUTS, HMC
import pandas as pd
import jax_tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.io import make_output_dirs
from experiments.run_replicates import parse_seed_spec
from src.cauchy_mh import independent_cauchy_mh
from src.scp_core import SCP, uniform_sample_bright_side
from experiments.targets import RobitRegression

numpyro.set_host_device_count(20)

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


def build_design(n, d, data_seed, standardize=True):
    X = jax.random.normal(jax.random.PRNGKey(data_seed), (n, d-1))
    X -= jnp.mean(X, axis=0)
    if standardize:
        X /= jnp.std(X, axis=0)
        X *= 0.5
    X = jnp.hstack([jnp.ones((n, 1)), X])
    y = X[:, 1] > 0.
    return X, y


def run_nuts(target, nsample, burnin, thinning, savepath=None, seed=0, nchains=20):
    d = target.d

    kernel = NUTS(potential_fn=lambda z: -target.log_prob(z))

    start = time.time()
    mcmc = MCMC(
        kernel,
        num_warmup=burnin,
        num_samples=nsample,
        num_chains=nchains,
        thinning=thinning,
        progress_bar=False,
    )
    init_params = jnp.zeros(d) if nchains == 1 else jnp.zeros((nchains, d))
    mcmc.run(jax.random.key(seed), init_params=init_params, extra_fields=("accept_prob",))
    nuts_samples = mcmc.get_samples()
    elapsed = time.time() - start
    accept_rate = jnp.mean(mcmc.get_extra_fields()['accept_prob'])
    meta_data = {
        'accept_rate': float(accept_rate),
        'time': elapsed,
    }
    if savepath is not None:
        pd.DataFrame(nuts_samples).to_csv(savepath)
        pd.DataFrame(meta_data, index=[0]).to_csv(savepath.replace('.csv', '_meta.csv'), index=False)
        print("saved to", savepath)
    return nuts_samples, meta_data


def run_scp(target, latitude, affine, seed, stepsize, nsample, burnin, thinning, savepath=None):
    d = target.d
    scp_model = SCP(d=d, latitude=latitude, affine=affine)
    print("Initializing SCP parameters......")
    train_start = time.time()
    opt_params, losses = scp_model.minimize_reverse_kl(
        target.log_prob,
        seed=seed,
        ntrain=2000,
        max_iter=1000,
        learning_rate=0.01,
        clip_value=None,
    )
    train_time = time.time() - train_start
    fit_diagnostics = scp_fit_diagnostics(scp_model, target, opt_params, seed)

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
    meta_data = {
        'accept_rate': float(scp_accept_prob),
        'time': scp_time,
        'train_time': train_time,
        'final_loss': float(losses[-1]),
        **fit_diagnostics,
    }
    if savepath is not None:
        pd.DataFrame(scp_samples).to_csv(savepath)
        print("saved to", savepath)
        pd.DataFrame(meta_data, index=[0]).to_csv(savepath.replace('.csv', '_meta.csv'), index=False)
    return scp_samples, meta_data


def scp_fit_diagnostics(scp_model, target, params, seed):
    observer, shift, scale = scp_model.transform_params(params)
    scale_np = np.asarray(scale)
    if scale_np.ndim == 0:
        scale_values = np.asarray([float(scale_np)])
        scale_diag = scale_values
        scale_singular_values = scale_values
    else:
        scale_diag = np.diag(scale_np)
        scale_singular_values = np.linalg.svd(scale_np, compute_uv=False)

    if isinstance(seed, int):
        key1, _ = jax.random.split(jax.random.key(seed))
    else:
        key1, _ = jax.random.split(seed)
    x0 = uniform_sample_bright_side(scp_model.d, scp_model.latitude, key1, n=1)[0]
    y0 = scp_model.projection(params, x0)
    y0_log_prob = target.log_prob(y0)

    return {
        'fit_shift_norm': float(jnp.linalg.norm(shift)),
        'fit_observer_norm': float(jnp.linalg.norm(observer)),
        'fit_scale_diag_min': float(np.min(scale_diag)),
        'fit_scale_diag_max': float(np.max(scale_diag)),
        'fit_scale_singular_min': float(np.min(scale_singular_values)),
        'fit_scale_singular_max': float(np.max(scale_singular_values)),
        'fit_scale_condition': float(np.max(scale_singular_values) / np.min(scale_singular_values)),
        'initial_projected_max_abs': float(jnp.max(jnp.abs(y0))),
        'initial_projected_norm': float(jnp.linalg.norm(y0)),
        'initial_projected_log_prob': float(y0_log_prob),
        'initial_projected_log_prob_finite': bool(jnp.isfinite(y0_log_prob)),
    }


def run_hmc(target, seed, nsample, burnin, thinning, savepath=None):
    start = time.time()
    hmc_kernel = HMC(potential_fn=lambda z: -target.log_prob(z), step_size=0.5, adapt_step_size=True, adapt_mass_matrix=False, num_steps=10, trajectory_length=None)
    mcmc = MCMC(hmc_kernel, num_warmup=burnin, num_samples=nsample, thinning=thinning, num_chains=1)
    mcmc.run(jax.random.key(seed), init_params=jnp.zeros(target.d), extra_fields=("accept_prob",))
    hmc_samples = mcmc.get_samples()
    hmc_time = time.time() - start
    accept_rate = jnp.mean(mcmc.get_extra_fields()['accept_prob'])
    print("HMC acceptance rate:", accept_rate, 'Time:', hmc_time)
    meta_data = {
        'accept_rate': float(accept_rate),
        'time': hmc_time
    }
    if savepath is not None:
        pd.DataFrame(hmc_samples).to_csv(savepath)
        print("saved to", savepath)
        pd.DataFrame(meta_data, index=[0]).to_csv(savepath.replace('.csv', '_meta.csv'), index=False)
    return hmc_samples, meta_data


def run_imh(target, seed, stepsize, nsample, burnin, thinning, savepath=None):
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
    meta_data = {
        'accept_rate': float(accept_rate),
        'time': elapsed,
        'stepsize': stepsize,
    }
    if savepath is not None:
        pd.DataFrame(samples).to_csv(savepath)
        print("saved to", savepath)
        pd.DataFrame(meta_data, index=[0]).to_csv(savepath.replace('.csv', '_meta.csv'), index=False)
    return samples, meta_data


def run_gibbs(target, seed, nsample, burnin, thinning, savepath=None):
    def sample_truncnorm_half(key, mu, sigma, side):
        alpha = (0.0 - mu) / sigma
        Phi_alpha = ndtr(alpha)
        lo = jnp.where(side > 0, Phi_alpha, 0.0)
        hi = jnp.where(side > 0, 1.0, Phi_alpha)
        u = jax.random.uniform(key, shape=mu.shape, minval=0.0, maxval=1.0)
        u = lo + (hi - lo) * u
        eps = 1e-6
        u = jnp.clip(u, eps, 1.0 - eps)
        z_std = ndtri(u)
        return mu + sigma * z_std

    def sym_psd_solve(prec, b):
        """
        Solve (prec) x = b for symmetric positive-definite 'prec'.
        """
        L = jnp.linalg.cholesky(prec)
        y = jax.scipy.linalg.solve_triangular(L, b, lower=True)
        x = jax.scipy.linalg.solve_triangular(L.T, y, lower=False)
        return x, L

    def mvn_sample_from_precision(key, prec, mean_prec):
        """
        Given precision matrix P and P*mu = mean_prec, sample x ~ N(mu, P^{-1}).
        Returns x.
        """
        mu, L = sym_psd_solve(prec, mean_prec)  # L is cholesky(prec)
        # If z ~ N(0, I), x = mu + P^{-1/2} z.
        # Since prec = L L^T, P^{-1/2} = L^{-T}
        z = jax.random.normal(key, shape=mu.shape)
        v = jax.scipy.linalg.solve_triangular(L.T, z, lower=False)
        return mu + v

    X, y = target.X, target.y
    n, d = target.n, target.d
    nu = target.link_df
    s_link = target.link_scale
    nu0 = target.prior_df
    s0 = target.prior_scale
    
    key = jax.random.key(seed)
    beta = jnp.zeros(d)
    lam = jnp.ones(n)
    tau = jnp.ones(d)

    eta = X @ beta
    side = 2 * y.astype(jnp.int32) - 1  # y in {0,1} -> side in {-1,+1}
    z = eta  # start at the mean (will be overwritten in first step)

    def one_step(carry):
        key, beta, z, lam, tau = carry
        key, k1, k2, k3, k4 = jax.random.split(key, 5)

        # --- Sample z | beta, lambda, y (truncated normals) ---
        eta = X @ beta
        sigma = s_link / jnp.sqrt(lam)  # elementwise
        z = sample_truncnorm_half(k1, eta, sigma, side)

        # --- Sample lambda | z, beta (Gamma((nu+1)/2, (nu + ((z-eta)/s)^2)/2)) ---
        resid = (z - eta) / s_link
        shape_lam = 0.5 * (nu + 1.0)
        rate_lam = 0.5 * (nu + resid**2)
        lam = jax.random.gamma(k2, shape_lam, shape=z.shape) / rate_lam

        # --- Sample tau | beta (per-coordinate Gamma((nu0+1)/2, (nu0 + (beta/s0)^2)/2)) ---
        shape_tau = 0.5 * (nu0 + 1.0)
        rate_tau = 0.5 * (nu0 + (beta / s0) ** 2)
        tau = jax.random.gamma(k3, shape_tau, shape=beta.shape) / rate_tau

        # --- Sample beta | z, lambda, tau (multivariate normal) ---
        # Likelihood precision: X^T diag(lam/s_link^2) X
        w = lam / (s_link**2)
        XtW = X.T * w  # each row j: w * X[:, j]
        lik_prec = XtW @ X  # (d,d)
        prior_prec = jnp.diag(tau / (s0**2))
        prec = lik_prec + prior_prec
        mean_prec = XtW @ z  # = X^T (W z)

        beta = mvn_sample_from_precision(k4, prec, mean_prec)

        return (key, beta, z, lam, tau)

    def run_steps(carry, nsteps):
        return jax.lax.fori_loop(0, nsteps, lambda _, current: one_step(current), carry)

    if nsample % thinning != 0:
        raise ValueError(f'nsample={nsample} must be divisible by thinning={thinning}')

    n_keep = nsample // thinning

    @jax_tqdm.scan_tqdm(n_keep)
    def sample_step(carry, _):
        carry = one_step(carry)
        beta_sample = carry[1]
        carry = run_steps(carry, thinning - 1)
        return carry, beta_sample

    start = time.time()
    carry = run_steps((key, beta, z, lam, tau), burnin)
    (key, beta, z, lam, tau), samples = jax.lax.scan(sample_step, carry, jnp.arange(n_keep))
    print(jnp.median(samples, axis=0))
    
    time_elapsed = time.time() - start
    print("Gibbs sampling time:", time_elapsed)
    meta_data = {
        'accept_rate': 1.0,
        'time': time_elapsed
    }
    if savepath is not None:
        pd.DataFrame(samples).to_csv(savepath)
        print("saved to", savepath)
        pd.DataFrame(meta_data, index=[0]).to_csv(savepath.replace('.csv', '_meta.csv'), index=False)
    return samples, meta_data


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
        f'QQ plot for beta{coord}: robit d={d}, n={args.n}, stepsize={args.stepsize}, affine={args.affine}',
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
        f'robit_d{args.d}_n{args.n}_std_{args.standardize}_prior_{prior_df}_{args.prior_scale}'
        f'_affine{args.affine}'
    )


def reference_prefix(args, prior_df):
    return f'robit_d{args.d}_n{args.n}_std_{args.standardize}_prior_{prior_df}_{args.prior_scale}'


def reference_base(result_dir, args, prior_df):
    return result_dir / (
        f'{reference_prefix(args, prior_df)}_reference_n{args.reference_nsample}'
        f'_thin{args.reference_thinning}_chains{args.reference_chains}'
    )


def legacy_reference_glob(args, prior_df):
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
        legacy_paths = sorted(result_dir.glob(legacy_reference_glob(args, prior_df)))
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
        nsample=args.reference_nsample,
        burnin=args.reference_burnin,
        thinning=args.reference_thinning,
        seed=args.reference_seed,
        nchains=args.reference_chains,
    )
    quantiles = make_quantiles(nuts_samples, PS, 'NUTS', coord=PLOT_COORD)
    quantiles.to_csv(quantiles_path)
    pd.DataFrame([nuts_meta]).to_csv(meta_path)
    print("saved NUTS reference quantiles to", quantiles_path)
    del nuts_samples
    gc.collect()
    jax.clear_caches()
    return quantiles, nuts_meta


def replace_method_results(quantiles_path, meta_path, quantiles_by_method, meta_by_method):
    if not quantiles_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f'Cannot replace method results because existing files were not found: '
            f'{quantiles_path} and {meta_path}'
        )

    existing_quantiles = pd.read_csv(quantiles_path, index_col=0)
    for method, quantiles in quantiles_by_method.items():
        for column in quantiles.columns:
            if len(existing_quantiles.index) != len(quantiles.index):
                raise ValueError(
                    f'Cannot replace {column}: existing file has {len(existing_quantiles.index)} '
                    f'quantile rows, replacement has {len(quantiles.index)}'
                )
            existing_quantiles[column] = quantiles[column].to_numpy(dtype=float)
    existing_quantiles.to_csv(quantiles_path)

    existing_meta = pd.read_csv(meta_path, index_col=0)
    replacement_meta = pd.DataFrame(meta_by_method).T
    existing_meta = existing_meta.drop(index=replacement_meta.index, errors='ignore')
    pd.concat([existing_meta, replacement_meta], axis=0, sort=False).to_csv(meta_path)


def run_method(target, method, seed, args):
    if method == 'Gibbs':
        return run_gibbs(
            target,
            seed=seed,
            nsample=args.nsample,
            burnin=args.burnin,
            thinning=args.thinning,
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
    X, y = build_design(args.n, args.d, args.data_seed, standardize=args.standardize)
    target = RobitRegression(
        X,
        y,
        link_df=args.link_df,
        link_scale=1.0,
        prior_df=prior_df,
        prior_scale=args.prior_scale,
    )
    replacing_existing_methods = getattr(args, 'replace_existing_methods', False)
    if replacing_existing_methods:
        nuts_quantiles, nuts_meta = None, {}
    else:
        nuts_quantiles, nuts_meta = load_or_compute_reference(target, result_dir, args, prior_df)

    for seed in seeds:
        base = seed_base(result_dir, args, prior_df, seed)
        quantiles_path = Path(f'{base}_quantiles.csv')
        meta_path = Path(f'{base}_meta.csv')
        if args.resume and not replacing_existing_methods and quantiles_path.exists() and meta_path.exists():
            print(f'Skipping existing seed result: {quantiles_path}')
            continue

        print(f'Running robit+t prior_df={prior_df}, seed={seed}, methods={methods}')
        quantiles_by_method = {} if replacing_existing_methods else {'NUTS': nuts_quantiles}
        meta_by_method = {} if replacing_existing_methods else {'nuts_reference': nuts_meta}

        for method in methods:
            samples, meta = run_method(target, method, seed, args)
            quantiles_by_method[method] = make_quantiles(samples, PS, method, coord=PLOT_COORD)
            meta_by_method[method.lower()] = meta
            del samples
            gc.collect()
            jax.clear_caches()

        if replacing_existing_methods:
            replace_method_results(quantiles_path, meta_path, quantiles_by_method, meta_by_method)
            print("updated existing quantiles at", quantiles_path)
            print("updated existing meta info at", meta_path)
        else:
            pd.concat(list(quantiles_by_method.values()), axis=1).to_csv(quantiles_path)
            pd.DataFrame(meta_by_method).T.to_csv(meta_path)
            print("saved quantiles to", quantiles_path)
            print("saved meta info to", meta_path)
        print_acceptance_summary(meta_by_method)


def run(args):
    result_dir, _ = make_output_dirs(args.rootdir, args.plotdir, 'regression', args.date)
    result_dir = Path(result_dir)
    methods = normalize_methods(args.methods)
    dfs = [args.prior_df] if args.dfs is None else args.dfs
    seeds = parse_seed_spec(args.seeds) if args.seeds is not None else [args.seed]

    for prior_df in dfs:
        run_prior_df(args, result_dir, prior_df, seeds, methods)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default='robit_t')
    parser.add_argument('--rootdir', type=str, default='results')
    parser.add_argument('--plotdir', type=str, default='plots')
    parser.add_argument('--algo', type=str, default='scp', choices=['all', 'scp', 'nuts', 'hmc', 'is', 'imh', 'gibbs'])
    parser.add_argument('--d', type=int, default=20)
    parser.add_argument('--n', type=int, default=50)
    parser.add_argument('--data_seed', type=int, default=2025)
    parser.add_argument('--prior_df', type=float, default=2.0)
    parser.add_argument('--dfs', type=float, nargs='+', default=None)
    parser.add_argument('--prior_scale', type=float, default=2.5)
    parser.add_argument('--standardize', dest='standardize', action='store_true', default=True)
    parser.add_argument('--no_standardize', dest='standardize', action='store_false')
    parser.add_argument('--link_df', type=float, default=2.0)
    parser.add_argument('--latitude', type=float, default=1.1)
    parser.add_argument('--affine', type=str, default='covariance', choices=['scalar', 'covariance'])
    parser.add_argument('--nsample', type=int, default=500_000)
    parser.add_argument('--burnin', type=int, default=100)
    parser.add_argument('--stepsize', type=float, default=.1)
    parser.add_argument('--is_stepsize', type=float, default=.01)
    parser.add_argument('--thinning', type=int, default=50)
    parser.add_argument('--hmc_nsample', type=int, default=100_000)
    parser.add_argument('--hmc_thinning', type=int, default=10)
    parser.add_argument('--reference_nsample', type=int, default=500_000)
    parser.add_argument('--reference_burnin', type=int, default=100)
    parser.add_argument('--reference_thinning', type=int, default=50)
    parser.add_argument('--reference_chains', type=int, default=20)
    parser.add_argument('--reference_seed', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seeds', type=str, default=None)
    parser.add_argument('--methods', nargs='+', default=DEFAULT_METHODS)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument(
        '--replace_existing_methods',
        action='store_true',
        help='Rerun only --methods and replace their columns/metadata in existing seed CSVs.',
    )
    args = parser.parse_args()

    run(args)
