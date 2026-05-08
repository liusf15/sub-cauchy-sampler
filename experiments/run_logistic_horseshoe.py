import numpy as np
from tqdm import trange
import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import MCMC, NUTS, HMC
import pandas as pd
import os
import argparse
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from polyagamma import random_polyagamma

from src.cauchy_mh import independent_cauchy_mh
from src.scp_core import SCP
from experiments.targets import LogisticRegressionHorseshoe

numpyro.set_host_device_count(20)

def run_nuts(target, seed, nsample, burnin, thinning, nchains):
    d = target.d

    print("Running NUTS......")
    start = time.time()
    kernel = NUTS(potential_fn=lambda z: -target.log_prob(z))

    mcmc = MCMC(kernel, num_warmup=burnin, num_samples=nsample, num_chains=nchains, thinning=thinning, progress_bar=False)
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
        seed=1,
        ntrain=256,
        max_iter=1000,
        learning_rate=0.01,
        clip_value=100.,
    )
    train_time = time.time() - train_start
    print("Final loss", losses[-1], "Time:", train_time)

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


def run_gibbs(target, seed, nsample, burnin, thinning):
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
    for t in trange(total_iters):
        # 1. Sample omega_i | beta  ~  PG(1, x_i^T beta)
        eta = X @ beta
        omega = random_polyagamma(np.ones(n), eta, random_state=rng)

        # 2. Sample beta | omega, lambda, tau, y
        prior_prec = 1.0 / (tau_sq * lambda_sq)  # (p,)
        WX = X * omega[:, None]
        XtWX = X.T @ WX
        P = XtWX + np.diag(prior_prec)
        L = np.linalg.cholesky(P)
        rhs = X.T @ kappa
        m = np.linalg.solve(L.T, np.linalg.solve(L, rhs))
        z = rng.normal(size=p)
        v = np.linalg.solve(L.T, z)
        beta = m + v

        # 3. Sample lambda_j^2 | beta_j, tau, nu_j  ~  IG(1, 1/nu_j + beta_j^2 / (2*tau^2))
        rate_lambda = 1.0 / nu + beta**2 / (2.0 * tau_sq)
        lambda_sq = rate_lambda / rng.gamma(shape=1.0, size=p)

        # 4. Sample nu_j | lambda_j  ~  IG(1, 1 + 1/lambda_j^2)
        rate_nu = 1.0 + 1.0 / lambda_sq
        nu = rate_nu / rng.gamma(shape=1.0, size=p)

        # 5. Sample tau^2 | beta, lambda, xi  ~  IG((p+1)/2, 1/xi + sum(beta_j^2 / (2*lambda_j^2)))
        shape_tau = 0.5 * (p + 1)
        rate_tau = 1.0 / xi + 0.5 * np.sum(beta**2 / lambda_sq)
        tau_sq = rate_tau / rng.gamma(shape=shape_tau)

        # 6. Sample xi | tau  ~  IG(1, 1 + 1/tau^2)
        rate_xi = 1.0 + 1.0 / tau_sq
        xi = rate_xi / rng.gamma(shape=1.0)

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
    }


def make_quantiles(samples, ps, name, coord=None):
    samples = jnp.asarray(samples)
    if coord is not None:
        samples = samples[:, coord]
        values = jnp.quantile(samples, ps, axis=0)
        columns = [f'{name}{coord}']
    else:
        values = jnp.quantile(samples, ps, axis=0)
        columns = [f'{name}{i}' for i in range(samples.shape[-1])]
    return pd.DataFrame(
        values,
        index=ps,
        columns=columns,
    )


def print_acceptance_summary(meta_by_method):
    print("\nAcceptance rate summary:")
    for method in ['nuts', 'gibbs', 'hmc', 'imh', 'scp']:
        accept_rate = float(meta_by_method[method]['accept_rate'])
        print(f"  {method.upper():<5} {accept_rate:.6g}")

def run(args):
    d = args.d
    n = args.n
    X = jax.random.normal(jax.random.PRNGKey(2025), (n, d-1))
    X -= jnp.mean(X, axis=0)
    X /= jnp.std(X, axis=0)
    X *= 0.5

    X = jnp.hstack([jnp.ones((n, 1)), X])
    y = X[:, 1] > 0.

    target = LogisticRegressionHorseshoe(X, y)
    savepath = os.path.join(args.rootdir, args.date, 'logistic_horseshoe')
    filename_prec = f"logistic_horseshoe_d{d}_n{n}_affine{args.affine}"
    os.makedirs(savepath, exist_ok=True)

    nuts_samples, nuts_meta = run_nuts(
        target,
        seed=args.seed,
        nsample=10000,
        burnin=100,
        thinning=1,
        nchains=20,
    )
    scp_samples, scp_meta = run_scp(
        target,
        latitude=args.latitude,
        affine=args.affine,
        seed=args.seed,
        stepsize=args.stepsize,
        nsample=args.nsample,
        burnin=args.burnin,
        thinning=args.thinning,
    )
    hmc_samples, hmc_meta = run_hmc(
        target,
        seed=args.seed,
        nsample=args.nsample//10,
        burnin=args.burnin ,
        thinning=args.thinning//10,
    )
    imh_samples, imh_meta = run_imh(
        target,
        seed=args.seed,
        stepsize=0.01,
        nsample=args.nsample//10,
        burnin=args.burnin,
        thinning=args.thinning//10,
    )
    gibbs_samples, gibbs_meta = run_gibbs(
        target,
        seed=args.seed,
        nsample=args.nsample//10,
        burnin=args.burnin,
        thinning=args.thinning//10,
    )
    print(gibbs_samples.shape)

    filename_base = os.path.join(
        savepath,
        f'{filename_prec}_lat{args.latitude}_stepsize{args.stepsize}_n{args.nsample}_seed{args.seed}',
    )
    samples_by_method = {
        'NUTS': nuts_samples,
        'SCP': scp_samples,
        'HMC': hmc_samples,
        'IMH': imh_samples,
        'Gibbs': gibbs_samples,  # already beta-only
    }
    meta_by_method = {
        'nuts': nuts_meta,
        'scp': scp_meta,
        'hmc': hmc_meta,
        'imh': imh_meta,
        'gibbs': gibbs_meta,
    }

    ps = jnp.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99])
    quantiles_by_method = {
        method: make_quantiles(samples, ps, method, coord=None)
        for method, samples in samples_by_method.items()
    }
    quantiles = pd.concat(list(quantiles_by_method.values()), axis=1)
    quantiles_path = f'{filename_base}_quantiles.csv'
    quantiles.to_csv(quantiles_path)
    print("saved quantiles to", quantiles_path)

    print_acceptance_summary(meta_by_method)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default='20260508')
    parser.add_argument('--rootdir', type=str, default='experiments/results')
    parser.add_argument('--d', type=int, default=20)
    parser.add_argument('--n', type=int, default=50)
    parser.add_argument('--latitude', type=float, default=1.7)
    parser.add_argument('--affine', type=str, default='covariance', choices=['scalar', 'covariance'])
    parser.add_argument('--nsample', type=int, default=5_000_000)
    parser.add_argument('--thinning', type=int, default=500)
    parser.add_argument('--burnin', type=int, default=100)
    parser.add_argument('--stepsize', type=float, default=.003)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    run(args)
