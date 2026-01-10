import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr, ndtri
import numpyro
from numpyro.infer import MCMC, NUTS, HMC
import pandas as pd
import os
import argparse
import time
import jax_tqdm

from src.scp_core import SCP
from experiments.targets import RobitRegression

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

def run_scp(target, latitude, seed, stepsize, nsample, burnin, thinning, savepath):
    d = target.d
    scp_model = SCP(d=d, latitude=latitude) 
    print("Initializing SCP parameters......")
    opt_params, losses = scp_model.minimize_reverse_kl(target.log_prob, 
                                                       seed=0, 
                                                       ntrain=2000,
                                                       max_iter=1000, 
                                                       learning_rate=0.1,
                                                       clip_value=200.)

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

def run_gibbs(target, seed, nsample, burnin, thinning, savepath):
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

    @jax_tqdm.scan_tqdm(burnin + nsample)
    def one_step(carry, t):
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

        return (key, beta, z, lam, tau), (beta, z, lam, tau)

    # Run warmup + sampling in a single scan (discard warmup later)
    total_iters = burnin + nsample
    start = time.time()
    (key, beta, z, lam, tau), traj = jax.lax.scan(one_step, (key, beta, z, lam, tau), jnp.arange(total_iters))
    beta_samples = traj[0]
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

    target = RobitRegression(X, y, link_df=2., link_scale=1.0, prior_df=prior_df, prior_scale=prior_scale)
    savepath = os.path.join(args.rootdir, args.date, 'robit')
    filename_prec = f"robit_d{d}_n{n}_std_{args.standardize}_prior_{prior_df}_{prior_scale}"
    os.makedirs(savepath, exist_ok=True)
    if args.algo == 'nuts':
        savepath = os.path.join(savepath, f'{filename_prec}_nuts_n{args.nsample}.csv')
        run_nuts(target, args.nsample, args.burnin, args.thinning, savepath)
    elif args.algo == 'scp':
        savepath = os.path.join(savepath, f'{filename_prec}_scp_lat{args.latitude}_stepsize{args.stepsize}_n{args.nsample}_seed{args.seed}.csv')
        run_scp(target, latitude=args.latitude, seed=args.seed, stepsize=args.stepsize, nsample=args.nsample, burnin=args.burnin, thinning=args.thinning, savepath=savepath)
    elif args.algo == 'hmc':
        savepath = os.path.join(savepath, f'{filename_prec}_hmc_n{args.nsample}_seed{args.seed}.csv')
        run_hmc(target, seed=args.seed, nsample=args.nsample, burnin=args.burnin, thinning=args.thinning, savepath=savepath)
    elif args.algo == 'gibbs':
        savepath = os.path.join(savepath, f'{filename_prec}_gibbs_n{args.nsample}_seed{args.seed}.csv')
        run_gibbs(target, seed=args.seed, nsample=args.nsample, burnin=args.burnin, thinning=args.thinning, savepath=savepath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default='20251016')
    parser.add_argument('--rootdir', type=str, default='experiments/results')
    parser.add_argument('--algo', type=str, default='scp', choices=['scp', 'nuts', 'hmc', 'gibbs'])
    parser.add_argument('--d', type=int, default=20)
    parser.add_argument('--n', type=int, default=50)
    parser.add_argument('--prior_df', type=float, default=2.0)
    parser.add_argument('--prior_scale', type=float, default=2.5)
    parser.add_argument('--standardize', action='store_true', default=False)
    parser.add_argument('--latitude', type=float, default=1.5)
    parser.add_argument('--nsample', type=int, default=10000)
    parser.add_argument('--burnin', type=int, default=100)
    parser.add_argument('--stepsize', type=float, default=.1)
    parser.add_argument('--thinning', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    run(args)
