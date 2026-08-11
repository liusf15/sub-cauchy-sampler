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
from src.scp_core import SCP, uniform_sample_bright_side
from experiments.targets import LogisticRegressionHorseshoe, NonCenteredLogisticRegressionHorseshoe

numpyro.set_host_device_count(20)

PLOT_METHODS = [
    ('Gibbs', 'Gibbs'),
    ('HMC', 'HMC'),
    ('IMH', 'IS'),
    ('SCP', 'SCS'),
]
METHOD_ALIASES = {
    'nuts': 'NUTS',
    'nuts_reference': 'NUTS',
    'gibbs': 'Gibbs',
    'hmc': 'HMC',
    'imh': 'IMH',
    'is': 'IMH',
    'scp': 'SCP',
    'scs': 'SCP',
}
ALL_METHODS = ['Gibbs', 'HMC', 'IMH', 'SCP']
DEFAULT_METHODS = ALL_METHODS
DEFAULT_METHOD_SEED_SPECS = {
    'Gibbs': '0:10',
    'HMC': '0:10',
    'IMH': '0:10',
    'SCP': '0:10',
}
PS = jnp.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99])


def none_or_float(value):
    if isinstance(value, str) and value.lower() in {'none', 'null'}:
        return None
    return float(value)


def normalize_methods(methods):
    if not methods:
        return DEFAULT_METHODS
    normalized = []
    for method in methods:
        key = method.lower()
        if key == 'all':
            return ALL_METHODS
        if key not in METHOD_ALIASES:
            raise ValueError(f'Unknown method {method!r}. Use NUTS, Gibbs, HMC, IMH/IS, SCP/SCS, or all.')
        canonical = METHOD_ALIASES[key]
        if canonical not in normalized:
            normalized.append(canonical)
    return normalized


def seed_lists_by_method(methods, seed, seeds_spec):
    if seeds_spec is not None:
        seeds = parse_seed_spec(seeds_spec)
        return {method: seeds for method in methods if method != 'NUTS'}
    return {
        method: parse_seed_spec(DEFAULT_METHOD_SEED_SPECS.get(method, str(seed)))
        for method in methods
        if method != 'NUTS'
    }


def build_design(n, d, data_seed):
    X = jax.random.normal(jax.random.PRNGKey(data_seed), (n, d-1))
    X -= jnp.mean(X, axis=0)
    X /= jnp.std(X, axis=0)
    X *= 0.5
    X = jnp.hstack([jnp.ones((n, 1)), X])
    y = X[:, 1] > 0.
    return X, y


def make_target(X, y, parametrization):
    if parametrization == 'centered':
        return LogisticRegressionHorseshoe(X, y)
    if parametrization == 'noncentered':
        return NonCenteredLogisticRegressionHorseshoe(X, y)
    raise ValueError(f'Unknown logistic horseshoe parametrization: {parametrization}')


def samples_for_output(target, method, samples):
    if method == 'Gibbs':
        return jnp.asarray(samples)
    if hasattr(target, 'to_centered'):
        return target.to_centered(samples)
    return jnp.asarray(samples)


def method_parametrization(target, method):
    if method == 'Gibbs':
        return 'centered'
    return getattr(target, 'parametrization', 'centered')

def run_nuts(target, seed, nsample, burnin, thinning, nchains):
    d = target.d

    print("Running NUTS......")
    start = time.time()
    kernel = NUTS(potential_fn=lambda z: -target.log_prob(z))

    mcmc = MCMC(kernel, num_warmup=burnin, num_samples=nsample, num_chains=nchains, thinning=thinning, progress_bar=True)
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

def tree_finite_stats(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return {
            'finite': True,
            'nan_count': 0,
            'inf_count': 0,
            'max_abs_finite': 0.0,
        }
    flat = np.concatenate([np.asarray(leaf).reshape(-1) for leaf in leaves])
    finite = np.isfinite(flat)
    return {
        'finite': bool(np.all(finite)),
        'nan_count': int(np.isnan(flat).sum()),
        'inf_count': int(np.isinf(flat).sum()),
        'max_abs_finite': float(np.max(np.abs(flat[finite]))) if finite.any() else np.nan,
    }


def scp_fit_diagnostics(scp_model, target, params, seed):
    raw_stats = tree_finite_stats(params)
    observer, shift, scale = scp_model.transform_params(params)
    scale_np = np.asarray(scale)
    scale_finite = bool(np.all(np.isfinite(scale_np)))
    diagnostics = {
        'fit_params_finite': raw_stats['finite'],
        'fit_params_nan_count': raw_stats['nan_count'],
        'fit_params_inf_count': raw_stats['inf_count'],
        'fit_params_max_abs_finite': raw_stats['max_abs_finite'],
        'fit_shift_norm': float(jnp.linalg.norm(shift)) if raw_stats['finite'] else np.nan,
        'fit_observer_norm': float(jnp.linalg.norm(observer)) if raw_stats['finite'] else np.nan,
        'fit_scale_finite': scale_finite,
        'fit_scale_nan_count': int(np.isnan(scale_np).sum()),
        'fit_scale_inf_count': int(np.isinf(scale_np).sum()),
    }
    if scale_finite:
        if scale_np.ndim == 0:
            scale_diag = np.asarray([float(scale_np)])
            scale_singular_values = scale_diag
        else:
            scale_diag = np.diag(scale_np)
            scale_singular_values = np.linalg.svd(scale_np, compute_uv=False)
        diagnostics.update({
            'fit_scale_diag_min': float(np.min(scale_diag)),
            'fit_scale_diag_max': float(np.max(scale_diag)),
            'fit_scale_singular_min': float(np.min(scale_singular_values)),
            'fit_scale_singular_max': float(np.max(scale_singular_values)),
            'fit_scale_condition': float(np.max(scale_singular_values) / np.min(scale_singular_values)),
        })
    else:
        diagnostics.update({
            'fit_scale_diag_min': np.nan,
            'fit_scale_diag_max': np.nan,
            'fit_scale_singular_min': np.nan,
            'fit_scale_singular_max': np.nan,
            'fit_scale_condition': np.nan,
        })

    if raw_stats['finite'] and scale_finite:
        if isinstance(seed, int):
            key1, _ = jax.random.split(jax.random.key(seed))
        else:
            key1, _ = jax.random.split(seed)
        x0 = uniform_sample_bright_side(scp_model.d, scp_model.latitude, key1, n=1)[0]
        y0 = scp_model.projection(params, x0)
        y0_log_prob = target.log_prob(y0)
        diagnostics.update({
            'fit_initial_projected_finite': bool(jnp.all(jnp.isfinite(y0))),
            'fit_initial_projected_max_abs': float(jnp.max(jnp.abs(y0))),
            'fit_initial_projected_log_prob': float(y0_log_prob),
            'fit_initial_projected_log_prob_finite': bool(jnp.isfinite(y0_log_prob)),
        })
    else:
        diagnostics.update({
            'fit_initial_projected_finite': False,
            'fit_initial_projected_max_abs': np.nan,
            'fit_initial_projected_log_prob': np.nan,
            'fit_initial_projected_log_prob_finite': False,
        })
    return diagnostics


def run_scp(
    target,
    latitude,
    affine,
    seed,
    stepsize,
    nsample,
    burnin,
    thinning,
    scp_ntrain,
    scp_max_iter,
    scp_learning_rate,
    scp_clip_value,
    scp_grad_clip_norm,
    scp_rwm_algo,
):
    d = target.d
    scp_model = SCP(d=d, latitude=latitude, affine=affine)
    print("Initializing SCP parameters......")
    train_start = time.time()
    opt_params, losses = scp_model.minimize_reverse_kl(
        target.log_prob,
        seed=seed,
        ntrain=scp_ntrain,
        max_iter=scp_max_iter,
        learning_rate=scp_learning_rate,
        clip_value=scp_clip_value,
        grad_clip_norm=scp_grad_clip_norm,
    )
    train_time = time.time() - train_start
    final_loss = float(losses[-1])
    fit_diagnostics = scp_fit_diagnostics(scp_model, target, opt_params, seed)
    print("Final loss", final_loss, "Time:", train_time)
    if (
        not np.isfinite(final_loss)
        or final_loss >= 1e19
        or not fit_diagnostics['fit_params_finite']
        or not fit_diagnostics['fit_scale_finite']
    ):
        hint = ''
        if getattr(target, 'parametrization', 'centered') == 'centered':
            hint = ' For the centered horseshoe target at latitude 1.1, use a smaller KL clip such as --scp_clip_value 5.'
        raise FloatingPointError(
            'SCP reverse-KL fit failed before sampling; '
            f'latitude={latitude}, affine={affine}, clip_value={scp_clip_value}, '
            f'final_loss={final_loss:.6g}, '
            f'params_finite={fit_diagnostics["fit_params_finite"]}, '
            f'scale_finite={fit_diagnostics["fit_scale_finite"]}.'
            f'{hint}'
        )

    print("Running RWM on the bright side......")
    start = time.time()
    scp_samples, scp_accept_prob = scp_model.rwm_bright_side(target.log_prob,
                                                                opt_params,
                                                                seed=seed,
                                                                stepsize=stepsize,
                                                                nsample=nsample,
                                                                burnin=burnin,
                                                                thinning=thinning,
                                                                algo=scp_rwm_algo)
    scp_time = time.time() - start
    print('SCP acceptance rate:', scp_accept_prob, 'Time:', scp_time)
    return scp_samples, {
        'accept_rate': float(scp_accept_prob),
        'time': scp_time,
        'nsample': nsample,
        'burnin': burnin,
        'thinning': thinning,
        'stepsize': stepsize,
        'train_time': train_time,
        'final_loss': final_loss,
        'scp_ntrain': scp_ntrain,
        'scp_max_iter': scp_max_iter,
        'scp_learning_rate': scp_learning_rate,
        'scp_clip_value': scp_clip_value,
        'scp_grad_clip_norm': scp_grad_clip_norm,
        'scp_rwm_algo': scp_rwm_algo,
        **fit_diagnostics,
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
        'nsample': nsample,
        'burnin': burnin,
        'thinning': thinning,
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
        'nsample': nsample,
        'burnin': burnin,
        'thinning': thinning,
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
    """Horseshoe Gibbs sampler using Polya-Gamma augmentation for the likelihood
    and inverse-gamma augmentation for half-Cauchy priors.

    Augmentation scheme:
        lambda_j^2 | nu_j ~ IG(1/2, 1/nu_j),  nu_j ~ IG(1/2, 1)
        tau^2     | xi    ~ IG(1/2, 1/xi),      xi   ~ IG(1/2, 1)
    """
    print("Running Gibbs......")
    rng = np.random.default_rng(seed)
    X = np.asarray(target.X, dtype=float)
    y = np.asarray(target.y, dtype=float)

    n, p = X.shape
    kappa = y - 0.5

    # Initialize parameters
    beta = np.zeros(p)
    lambda_sq = np.ones(p)   # lambda_j^2
    tau_sq = 1.0             # tau^2
    nu = np.ones(p)          # auxiliary for lambda
    xi = 1.0                 # auxiliary for tau

    total_iters = burnin + nsample
    gibbs_samples = np.zeros((nsample // thinning, target.d))

    start = time.time()
    sample_idx = 0
    max_abs_eta = 0.0
    min_omega = np.inf
    max_omega = 0.0
    for t in trange(total_iters):
        # 1. Sample omega_i | beta  ~  PG(1, x_i^T beta)
        eta = X @ beta
        check_finite('eta', eta, t)
        max_abs_eta = max(max_abs_eta, float(np.max(np.abs(eta))))
        omega = random_polyagamma(np.ones(n), eta, random_state=rng, method=pg_method)
        check_finite('Polya-Gamma omega', omega, t)
        min_omega = min(min_omega, float(np.min(omega)))
        max_omega = max(max_omega, float(np.max(omega)))

        # 2. Sample beta | omega, lambda, tau, y
        prior_prec = 1.0 / (tau_sq * lambda_sq)  # (p,)
        check_finite('prior precision', prior_prec, t)
        WX = X * omega[:, None]
        XtWX = X.T @ WX
        check_finite('X^T Omega X', XtWX, t)
        P = XtWX + np.diag(prior_prec)
        check_finite('Gibbs precision matrix', P, t)
        L = np.linalg.cholesky(P)
        rhs = X.T @ kappa
        m = np.linalg.solve(L.T, np.linalg.solve(L, rhs))
        z = rng.normal(size=p)
        v = np.linalg.solve(L.T, z)
        beta = m + v
        check_finite('beta', beta, t)

        # 3. Sample lambda_j^2 | beta_j, tau, nu_j  ~  IG(1, 1/nu_j + beta_j^2 / (2*tau^2))
        rate_lambda = 1.0 / nu + beta**2 / (2.0 * tau_sq)
        check_finite('lambda rate', rate_lambda, t)
        lambda_sq = rate_lambda / rng.gamma(shape=1.0, size=p)
        check_finite('lambda_sq', lambda_sq, t)

        # 4. Sample nu_j | lambda_j  ~  IG(1, 1 + 1/lambda_j^2)
        rate_nu = 1.0 + 1.0 / lambda_sq
        check_finite('nu rate', rate_nu, t)
        nu = rate_nu / rng.gamma(shape=1.0, size=p)
        check_finite('nu', nu, t)

        # 5. Sample tau^2 | beta, lambda, xi  ~  IG((p+1)/2, 1/xi + sum(beta_j^2 / (2*lambda_j^2)))
        shape_tau = 0.5 * (p + 1)
        rate_tau = 1.0 / xi + 0.5 * np.sum(beta**2 / lambda_sq)
        check_finite('tau rate', rate_tau, t)
        tau_sq = rate_tau / rng.gamma(shape=shape_tau)
        check_finite('tau_sq', tau_sq, t)

        # 6. Sample xi | tau  ~  IG(1, 1 + 1/tau^2)
        rate_xi = 1.0 + 1.0 / tau_sq
        check_finite('xi rate', rate_xi, t)
        xi = rate_xi / rng.gamma(shape=1.0)
        check_finite('xi', xi, t)

        if t >= burnin and (t - burnin) % thinning == 0:
            gibbs_samples[sample_idx, :p] = beta
            gibbs_samples[sample_idx, p:(2*p)] = .5 * jnp.log(lambda_sq)
            gibbs_samples[sample_idx, -1] = .5 * jnp.log(tau_sq)
            sample_idx += 1


    time_elapsed = time.time() - start
    print("Gibbs sampling time:", time_elapsed)
    return jnp.asarray(gibbs_samples), {
        'accept_rate': 1.0,
        'time': time_elapsed,
        'nsample': nsample,
        'burnin': burnin,
        'thinning': thinning,
        'pg_method': pg_method,
        'max_abs_eta': max_abs_eta,
        'min_omega': min_omega,
        'max_omega': max_omega,
    }


def make_quantiles(samples, ps, name, coord=None):
    samples = jnp.asarray(samples)
    if not bool(jnp.all(jnp.isfinite(samples))):
        raise FloatingPointError(f'Non-finite samples for {name}; refusing to save quantiles.')
    ps = jnp.asarray(ps)
    index = np.round(np.asarray(ps, dtype=float), 6)
    if coord is not None:
        samples = samples[:, coord]
        values = jnp.quantile(samples, ps, axis=0)
        columns = [f'{name}{coord}']
    else:
        values = jnp.quantile(samples, ps, axis=0)
        columns = [f'{name}{i}' for i in range(samples.shape[-1])]
    if not bool(jnp.all(jnp.isfinite(values))):
        raise FloatingPointError(f'Non-finite quantiles for {name}; refusing to save quantiles.')
    return pd.DataFrame(
        values,
        index=index,
        columns=columns,
    )


def coordinate_label(target, idx):
    if idx < target.p:
        return f'beta{idx}'
    if idx < 2 * target.p:
        return f'log_lambda{idx - target.p}'
    return 'log_tau'


def default_plot_indices(target):
    if target.p > 1:
        return [1, target.p + 1]
    return [0, target.p]


def save_figure9_style_plot(quantiles_by_method, filename, target, args):
    plot_indices = default_plot_indices(target)
    fig, axes = plt.subplots(
        len(plot_indices),
        len(PLOT_METHODS),
        figsize=(3 * len(PLOT_METHODS), 2.8 * len(plot_indices)),
        squeeze=False,
    )

    for row, idx in enumerate(plot_indices):
        coord_name = coordinate_label(target, idx)
        xref = quantiles_by_method['NUTS'][f'NUTS{idx}']
        for col, (method, label) in enumerate(PLOT_METHODS):
            ax = axes[row, col]
            y = quantiles_by_method[method][f'{method}{idx}']
            ax.plot(xref, y, marker='o', linewidth=1.5, markersize=4)
            ax.plot(xref, xref, 'r--', linewidth=1)
            if row == 0:
                ax.set_title(label)
            if col == 0:
                ax.set_ylabel(f'{coord_name} quantile')
            ax.set_xlabel('NUTS quantile')

    fig.suptitle(
        f'Logistic horseshoe Q-Q plot, d={args.d}, n={args.n}, stepsize={args.stepsize}, affine={args.affine}, parametrization={args.parametrization}',
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(f'{filename}_qq.pdf')
    print(f"Saved Figure 9-style horseshoe QQ plot to {filename}_qq.pdf")
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


def filename_prefix(args):
    suffix = '' if getattr(args, 'parametrization', 'centered') == 'centered' else f'_param{args.parametrization}'
    return f'logistic_horseshoe_d{args.d}_n{args.n}_std_True{suffix}_affine{args.affine}'


def reference_prefix_for_parametrization(args, parametrization):
    suffix = '' if parametrization == 'centered' else f'_param{parametrization}'
    return f'logistic_horseshoe_d{args.d}_n{args.n}_std_True{suffix}'


def reference_prefix(args):
    return reference_prefix_for_parametrization(
        args,
        getattr(args, 'parametrization', 'centered'),
    )


def method_reference_prefix(args, method):
    if method == 'Gibbs':
        return reference_prefix_for_parametrization(args, 'centered')
    return reference_prefix(args)


def reference_base(result_dir, args):
    return result_dir / (
        f'{reference_prefix(args)}_reference_n{args.reference_nsample}'
        f'_thin{args.reference_thinning}_chains{args.reference_chains}'
    )


def legacy_reference_glob(args):
    return (
        f'{reference_prefix(args)}_affine*_reference_n{args.reference_nsample}'
        f'_thin{args.reference_thinning}_chains{args.reference_chains}_nuts_reference.csv'
    )


def seed_base(result_dir, args, seed):
    return result_dir / (
        f'{filename_prefix(args)}_lat{args.latitude}_stepsize{args.stepsize}'
        f'_n{args.nsample}_hmc_n{args.hmc_nsample}_seed{seed}'
    )


def format_tag_value(value):
    if value is None:
        return 'none'
    if isinstance(value, float):
        return f'{value:g}'
    return str(value)


def method_run_config(method, args):
    if method == 'SCP':
        return {
            'method_tag': 'scp',
            'nsample': args.nsample,
            'burnin': args.burnin,
            'thinning': args.thinning,
            'stepsize': args.stepsize,
            'latitude': args.latitude,
            'affine': args.affine,
            'scp_ntrain': args.scp_ntrain,
            'scp_max_iter': args.scp_max_iter,
            'scp_learning_rate': args.scp_learning_rate,
            'scp_clip_value': args.scp_clip_value,
            'scp_grad_clip_norm': args.scp_grad_clip_norm,
            'scp_rwm_algo': args.scp_rwm_algo,
        }
    if method == 'IMH':
        return {
            'method_tag': 'imh',
            'nsample': args.nsample,
            'burnin': args.burnin,
            'thinning': args.thinning,
            'stepsize': args.is_stepsize,
        }
    if method == 'Gibbs':
        return {
            'method_tag': 'gibbs',
            'nsample': args.gibbs_nsample if args.gibbs_nsample is not None else args.nsample,
            'burnin': args.gibbs_burnin if args.gibbs_burnin is not None else args.burnin,
            'thinning': args.gibbs_thinning if args.gibbs_thinning is not None else args.thinning,
            'pg_method': args.pg_method,
        }
    if method == 'HMC':
        return {
            'method_tag': 'hmc',
            'nsample': args.hmc_nsample,
            'burnin': args.burnin,
            'thinning': args.hmc_thinning,
        }
    raise ValueError(f'No run configuration for method {method!r}')


def method_base(result_dir, args, seed, method):
    cfg = method_run_config(method, args)
    parts = [
        method_reference_prefix(args, method),
        f'seed{seed}',
        cfg['method_tag'],
        f'n{cfg["nsample"]}',
        f'thin{cfg["thinning"]}',
        f'burnin{cfg["burnin"]}',
    ]
    if method in {'SCP', 'IMH'}:
        parts.append(f'stepsize{format_tag_value(cfg["stepsize"])}')
    if method == 'SCP':
        parts.extend([
            f'lat{format_tag_value(cfg["latitude"])}',
            f'affine{cfg["affine"]}',
            f'clip{format_tag_value(cfg["scp_clip_value"])}',
            f'algo{cfg["scp_rwm_algo"]}',
            f'ntrain{cfg["scp_ntrain"]}',
        ])
    if method == 'Gibbs':
        parts.append(f'pg{cfg["pg_method"]}')
    return result_dir / '_'.join(parts)


def method_paths(result_dir, args, seed, method):
    base = method_base(result_dir, args, seed, method)
    return Path(f'{base}_quantiles.csv'), Path(f'{base}_meta.csv')


def save_method_result(result_dir, args, seed, method, samples, meta, target):
    output_samples = samples_for_output(target, method, samples)
    quantiles = make_quantiles(output_samples, PS, method, coord=None)
    meta = dict(meta)
    meta.update(method_run_config(method, args))
    meta['method'] = method
    meta['seed'] = seed
    meta['data_seed'] = args.data_seed
    meta['d'] = args.d
    meta['n'] = args.n
    meta['parametrization'] = method_parametrization(target, method)
    meta['output_parametrization'] = 'centered'

    quantiles_path, meta_path = method_paths(result_dir, args, seed, method)
    quantiles.to_csv(quantiles_path)
    pd.DataFrame([meta]).to_csv(meta_path)
    print("saved method quantiles to", quantiles_path)
    print("saved method meta info to", meta_path)
    return quantiles_path, meta_path


def load_or_compute_reference(target, result_dir, args):
    base = reference_base(result_dir, args)
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
        legacy_paths = sorted(result_dir.glob(legacy_reference_glob(args)))
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
    )
    nuts_output_samples = samples_for_output(target, 'NUTS', nuts_samples)
    quantiles = make_quantiles(nuts_output_samples, PS, 'NUTS', coord=None)
    nuts_meta = dict(nuts_meta)
    nuts_meta['seed'] = args.reference_seed
    nuts_meta['nsample'] = args.reference_nsample
    nuts_meta['burnin'] = args.reference_burnin
    nuts_meta['thinning'] = args.reference_thinning
    nuts_meta['chains'] = args.reference_chains
    nuts_meta['data_seed'] = args.data_seed
    nuts_meta['d'] = args.d
    nuts_meta['n'] = args.n
    nuts_meta['parametrization'] = method_parametrization(target, 'NUTS')
    nuts_meta['output_parametrization'] = 'centered'
    quantiles.to_csv(quantiles_path)
    pd.DataFrame([nuts_meta]).to_csv(meta_path)
    print("saved NUTS reference quantiles to", quantiles_path)
    del nuts_samples, nuts_output_samples
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
            nsample=args.gibbs_nsample if args.gibbs_nsample is not None else args.nsample,
            burnin=args.gibbs_burnin if args.gibbs_burnin is not None else args.burnin,
            thinning=args.gibbs_thinning if args.gibbs_thinning is not None else args.thinning,
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
            scp_ntrain=args.scp_ntrain,
            scp_max_iter=args.scp_max_iter,
            scp_learning_rate=args.scp_learning_rate,
            scp_clip_value=args.scp_clip_value,
            scp_grad_clip_norm=args.scp_grad_clip_norm,
            scp_rwm_algo=args.scp_rwm_algo,
        )
    raise ValueError(f'Unknown method: {method}')


def run_legacy_combined(args, result_dir, target, methods, seeds):
    replacing_existing_methods = getattr(args, 'replace_existing_methods', False)
    nuts_quantiles, nuts_meta = (None, {}) if replacing_existing_methods else load_or_compute_reference(target, result_dir, args)
    for seed in seeds:
        base = seed_base(result_dir, args, seed)
        quantiles_path = Path(f'{base}_quantiles.csv')
        meta_path = Path(f'{base}_meta.csv')
        if args.resume and not replacing_existing_methods and quantiles_path.exists() and meta_path.exists():
            print(f'Skipping existing seed result: {quantiles_path}')
            continue

        print(f'Running logistic horseshoe, seed={seed}, methods={methods}')
        quantiles_by_method = {} if replacing_existing_methods else {'NUTS': nuts_quantiles}
        meta_by_method = {} if replacing_existing_methods else {'nuts_reference': nuts_meta}

        for method in methods:
            samples, meta = run_method(target, method, seed, args)
            output_samples = samples_for_output(target, method, samples)
            quantiles_by_method[method] = make_quantiles(output_samples, PS, method, coord=None)
            meta = dict(meta)
            meta['parametrization'] = method_parametrization(target, method)
            meta['output_parametrization'] = 'centered'
            meta_by_method[method.lower()] = meta
            del samples, output_samples
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


def run_separate_methods(args, result_dir, target, methods, seeds_by_method):
    if 'NUTS' in methods:
        load_or_compute_reference(target, result_dir, args)
        methods = [method for method in methods if method != 'NUTS']
    if not methods:
        return

    for method in methods:
        method_seeds = seeds_by_method.get(method, [])
        for seed in method_seeds:
            print(f'Running logistic horseshoe, seed={seed}, methods={[method]}')
            quantiles_path, meta_path = method_paths(result_dir, args, seed, method)
            if args.resume and quantiles_path.exists() and meta_path.exists():
                print(f'Skipping existing method result: {quantiles_path}')
                continue

            samples, meta = run_method(target, method, seed, args)
            _, _ = save_method_result(result_dir, args, seed, method, samples, meta, target)
            print_acceptance_summary({method.lower(): meta})
            del samples
            gc.collect()
            jax.clear_caches()


def run(args):
    result_dir, _ = make_output_dirs(args.rootdir, args.plotdir, 'regression', args.date)
    result_dir = Path(result_dir)
    X, y = build_design(args.n, args.d, args.data_seed)
    target = make_target(X, y, args.parametrization)
    methods = normalize_methods(args.methods)

    if args.legacy_combined_output or args.replace_existing_methods:
        seeds = parse_seed_spec(args.seeds) if args.seeds is not None else [args.seed]
        if 'NUTS' in methods:
            methods = [method for method in methods if method != 'NUTS']
        run_legacy_combined(args, result_dir, target, methods, seeds)
        return

    run_separate_methods(
        args,
        result_dir,
        target,
        methods,
        seed_lists_by_method(methods, args.seed, args.seeds),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default='logistic_horseshoe_noncentered')
    parser.add_argument('--rootdir', type=str, default='results')
    parser.add_argument('--plotdir', type=str, default='plots')
    parser.add_argument('--d', type=int, default=20)
    parser.add_argument('--n', type=int, default=50)
    parser.add_argument('--data_seed', type=int, default=2025)
    parser.add_argument(
        '--parametrization',
        choices=['centered', 'noncentered'],
        default='noncentered',
        help='Horseshoe parametrization used by NUTS, HMC, IMH, and SCS; saved quantiles remain centered.',
    )
    parser.add_argument('--latitude', type=float, default=1.7)
    parser.add_argument('--affine', type=str, default='covariance', choices=['scalar', 'covariance'])
    parser.add_argument('--nsample', type=int, default=500_000)
    parser.add_argument('--thinning', type=int, default=50)
    parser.add_argument('--burnin', type=int, default=100)
    parser.add_argument(
        '--gibbs_nsample',
        type=int,
        default=500_000,
        help='Gibbs iterations after warmup.',
    )
    parser.add_argument(
        '--gibbs_thinning',
        type=int,
        default=50,
        help='Gibbs thinning.',
    )
    parser.add_argument(
        '--gibbs_burnin',
        type=int,
        default=100,
        help='Gibbs warmup iterations.',
    )
    parser.add_argument('--stepsize', type=float, default=.05)
    parser.add_argument('--is_stepsize', type=float, default=.01)
    parser.add_argument('--scp_ntrain', type=int, default=256)
    parser.add_argument('--scp_max_iter', type=int, default=1000)
    parser.add_argument('--scp_learning_rate', type=float, default=.01)
    parser.add_argument('--scp_clip_value', type=none_or_float, default=None)
    parser.add_argument('--scp_grad_clip_norm', type=float, default=10.)
    parser.add_argument(
        '--scp_rwm_algo',
        choices=['stepout', 'reject'],
        default='stepout',
        help='Boundary handling for SCS random walk on the bright side.',
    )
    parser.add_argument('--hmc_nsample', type=int, default=100_000)
    parser.add_argument('--hmc_thinning', type=int, default=10)
    parser.add_argument('--reference_nsample', type=int, default=500_000)
    parser.add_argument('--reference_burnin', type=int, default=100)
    parser.add_argument('--reference_thinning', type=int, default=500)
    parser.add_argument('--reference_chains', type=int, default=20)
    parser.add_argument('--reference_seed', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--seeds',
        type=str,
        default=None,
        help='Seed list applied to every selected method. Defaults to 0:10 for Gibbs, HMC, IMH, and SCS.',
    )
    parser.add_argument(
        '--methods',
        nargs='+',
        default=DEFAULT_METHODS,
        help='Methods to run. Defaults to Gibbs HMC IMH SCS. Use NUTS, Gibbs, HMC, IMH/IS, SCP/SCS, or all. NUTS only builds/uses the cached reference.',
    )
    parser.add_argument(
        '--pg_method',
        choices=['alternate', 'devroye', 'gamma', 'saddle'],
        default='alternate',
        help='Polya-Gamma sampler used by the logistic Gibbs update.',
    )
    parser.add_argument('--resume', action='store_true')
    parser.add_argument(
        '--legacy_combined_output',
        action='store_true',
        help='Write the historical per-seed combined quantile/meta files. By default each method is saved separately.',
    )
    parser.add_argument(
        '--replace_existing_methods',
        action='store_true',
        help='Compatibility mode: rerun only --methods and replace those columns/metadata in existing combined seed files.',
    )

    args = parser.parse_args()

    run(args)
