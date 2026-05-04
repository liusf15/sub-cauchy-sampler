import numpy as np
from tqdm import trange 
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import MCMC, NUTS, HMC
import pandas as pd
import os
import argparse
import time
from polyagamma import random_polyagamma

from src.cauchy_mh import independent_cauchy_mh
from src.scp_core import SCP
from experiments.targets import LogisticRegression

numpyro.set_host_device_count(20)

def run_nuts(target, nsample, burnin, thinning, savepath):
    d = target.d
    nchains = 20

    kernel = NUTS(potential_fn=lambda z: -target.log_prob(z))

    mcmc = MCMC(kernel, num_warmup=burnin, num_samples=nsample, num_chains=nchains, thinning=thinning)
    mcmc.run(jax.random.key(0), init_params=jnp.zeros((nchains, d)))
    nuts_samples = mcmc.get_samples()
    pd.DataFrame(nuts_samples).to_csv(savepath)
    print("saved to", savepath)

def run_scp(target, latitude, affine, seed, stepsize, nsample, burnin, thinning, savepath):
    d = target.d
    scp_model = SCP(d=d, latitude=latitude, affine=affine)
    print("Initializing SCP parameters......")
    opt_params, losses = scp_model.minimize_reverse_kl(target.log_prob, 
                                                       seed=0, 
                                                       ntrain=2000,
                                                       max_iter=1000, 
                                                       learning_rate=0.1,
                                                       clip_value=1000.)

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
    pd.DataFrame(scp_samples).to_csv(savepath)
    print("saved to", savepath)
    meta_data = {
        'accept_rate': float(scp_accept_prob),
        'time': scp_time
    }
    pd.DataFrame(meta_data, index=[0]).to_csv(savepath.replace('.csv', '_meta.csv'))

def run_hmc(target, seed, nsample, burnin, thinning, savepath):
    start = time.time()
    hmc_kernel = HMC(potential_fn=lambda z: -target.log_prob(z), step_size=0.5, adapt_step_size=True, adapt_mass_matrix=False, num_steps=10, trajectory_length=None)
    mcmc = MCMC(hmc_kernel, num_warmup=burnin, num_samples=nsample, thinning=thinning, num_chains=1)
    mcmc.run(jax.random.key(seed), init_params=jnp.zeros(target.d), extra_fields=("accept_prob",))
    hmc_samples = mcmc.get_samples()
    hmc_time = time.time() - start
    accept_rate = jnp.mean(mcmc.get_extra_fields()['accept_prob'])
    print("HMC acceptance rate:", accept_rate, 'Time:', hmc_time)
    pd.DataFrame(hmc_samples).to_csv(savepath)
    print("saved to", savepath)
    meta_data = {
        'accept_rate': float(accept_rate),
        'time': hmc_time
    }
    pd.DataFrame(meta_data, index=[0]).to_csv(savepath.replace('.csv', '_meta.csv'))


def run_imh(target, seed, stepsize, nsample, burnin, thinning, savepath):
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
    pd.DataFrame(samples).to_csv(savepath)
    print("saved to", savepath)
    meta_data = {
        'accept_rate': float(accept_rate),
        'time': elapsed,
    }
    pd.DataFrame(meta_data, index=[0]).to_csv(savepath.replace('.csv', '_meta.csv'))


def run_gibbs(target, seed, nsample, burnin, thinning, savepath):
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

    beta_samples = np.zeros((nsample, p))

    start = time.time()
    for t in trange(total_iters):
        # 1. Sample omega | beta
        eta = X @ beta              # (n,)
        # polyagamma expects shape and "z", here shape=1
        omega = random_polyagamma(np.ones(n), eta, random_state=rng)  # omega ~ PG(1, eta)

        # 2. Sample beta | omega, lambda, y
        # prior precision diag(lam / tau^2)
        prior_prec = lam / (tau ** 2)   # (p,)

        # X^T Omega X
        WX = X * omega[:, None]        # (n, p)
        XtWX = X.T @ WX                # (p, p)

        P = XtWX + np.diag(prior_prec)  # precision matrix
        L = np.linalg.cholesky(P)

        rhs = X.T @ kappa

        # solve P^{-1} rhs via Cholesky
        m = np.linalg.solve(L.T, np.linalg.solve(L, rhs))

        # sample from N(m, P^{-1})
        z = rng.normal(size=p)
        v = np.linalg.solve(L.T, z)
        beta = m + v

        # 3. Sample lambda_j | beta_j
        shape_post = 0.5 * (nu + 1.0)
        rate_post = 0.5 * (nu + (beta ** 2) / (tau ** 2))
        lam = rng.gamma(shape=shape_post) / rate_post

        if t >= burnin:
            idx = t - burnin
            beta_samples[idx] = beta

    samples = beta_samples[burnin::thinning]
    print(jnp.median(samples, axis=0))
    
    time_elapsed = time.time() - start
    print("Gibbs sampling time:", time_elapsed)
    pd.DataFrame(samples).to_csv(savepath)
    print("saved to", savepath)
    meta_data = {
        'accept_rate': 1.0,
        'time': time_elapsed
    }
    pd.DataFrame(meta_data, index=[0]).to_csv(savepath.replace('.csv', '_meta.csv'))

def run(args):
    d = args.d
    n = args.n
    X = jax.random.normal(jax.random.PRNGKey(2025), (n, d-1))
    X -= jnp.mean(X, axis=0)
    if args.standardize:
        X /= jnp.std(X, axis=0) 
        X *= 0.5
    X = jnp.hstack([jnp.ones((n, 1)), X])
    y = X[:, 1] > 0.

    prior_df = args.prior_df
    prior_scale = args.prior_scale

    target = LogisticRegression(X, y, prior_df=prior_df, prior_scale=prior_scale)
    savepath = os.path.join(args.rootdir, args.date, 'logistic')
    filename_prec = f"logistic_d{d}_n{n}_std_{args.standardize}_prior_{prior_df}_{prior_scale}_affine{args.affine}"
    os.makedirs(savepath, exist_ok=True)
    if args.algo == 'nuts':
        savepath = os.path.join(savepath, f'{filename_prec}_nuts_n{args.nsample}.csv')
        run_nuts(target, args.nsample, args.burnin, args.thinning, savepath)
    elif args.algo == 'scp':
        savepath = os.path.join(savepath, f'{filename_prec}_scp_lat{args.latitude}_stepsize{args.stepsize}_n{args.nsample}_seed{args.seed}.csv')
        run_scp(target, latitude=args.latitude, affine=args.affine, seed=args.seed, stepsize=args.stepsize, nsample=args.nsample, burnin=args.burnin, thinning=args.thinning, savepath=savepath)
    elif args.algo == 'hmc':
        savepath = os.path.join(savepath, f'{filename_prec}_hmc_n{args.nsample}_seed{args.seed}.csv')
        run_hmc(target, seed=args.seed, nsample=args.nsample, burnin=args.burnin, thinning=args.thinning, savepath=savepath)
    elif args.algo == 'imh':
        savepath = os.path.join(savepath, f'{filename_prec}_imh_stepsize{args.stepsize}_n{args.nsample}_seed{args.seed}.csv')
        run_imh(target, seed=args.seed, stepsize=args.stepsize, nsample=args.nsample, burnin=args.burnin, thinning=args.thinning, savepath=savepath)
    elif args.algo == 'gibbs':
        savepath = os.path.join(savepath, f'{filename_prec}_gibbs_n{args.nsample}_seed{args.seed}.csv')
        run_gibbs(target, seed=args.seed, nsample=args.nsample, burnin=args.burnin, thinning=args.thinning, savepath=savepath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default='20251201')
    parser.add_argument('--rootdir', type=str, default='experiments/results')
    parser.add_argument('--algo', type=str, default='scp', choices=['scp', 'nuts', 'hmc', 'imh', 'gibbs'])
    parser.add_argument('--d', type=int, default=20)
    parser.add_argument('--n', type=int, default=50)
    parser.add_argument('--prior_df', type=float, default=2.0)
    parser.add_argument('--prior_scale', type=float, default=2.5)
    parser.add_argument('--standardize', action='store_true', default=False)
    parser.add_argument('--latitude', type=float, default=1.5)
    parser.add_argument('--affine', type=str, default='scalar', choices=['scalar', 'covariance'])
    parser.add_argument('--nsample', type=int, default=10000)
    parser.add_argument('--burnin', type=int, default=100)
    parser.add_argument('--stepsize', type=float, default=.1)
    parser.add_argument('--thinning', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    run(args)
