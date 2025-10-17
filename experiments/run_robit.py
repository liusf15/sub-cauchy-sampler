import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, HMC
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
import time

from src.scp_core import SCP
from experiments.targets import RobitRegression

numpyro.set_host_device_count(20)

def run_nuts(target, savepath):
    d = target.d
    nchains = 10

    kernel = NUTS(potential_fn=lambda z: -target.log_prob(z))

    mcmc = MCMC(kernel, num_warmup=1_000, num_samples=100_000, num_chains=nchains, thinning=10)
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
                                                       max_iter=200, 
                                                       learning_rate=0.1,
                                                       clip_value=500.)

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

def run(args):
    d = args.d
    n = 10
    X = jax.random.normal(jax.random.PRNGKey(2025), (n, d-1))
    X -= jnp.mean(X, axis=0)
    X = jnp.hstack([jnp.ones((n, 1)), X])
    y = X[:, 1] > 0.
    
    target = RobitRegression(X, y, link_df=2., link_scale=1.0, prior_df=2., prior_scale=1.)
    savepath = os.path.join(args.rootdir, args.date, 'robit')
    os.makedirs(savepath, exist_ok=True)
    if args.algo == 'nuts':
        savepath = os.path.join(savepath, f'robit_d{d}_n{n}_nuts.csv')
        run_nuts(target, savepath)
    elif args.algo == 'scp':
        savepath = os.path.join(savepath, f'robit_d{d}_n{n}_scp_stepsize{args.stepsize}_n{args.nsample}_seed{args.seed}.csv')
        run_scp(target, latitude=args.latitude, seed=args.seed, stepsize=args.stepsize, nsample=args.nsample, burnin=args.burnin, thinning=args.thinning, savepath=savepath)
    elif args.algo == 'hmc':
        savepath = os.path.join(savepath, f'robit_d{d}_n{n}_hmc_n{args.nsample}_seed{args.seed}.csv')
        run_hmc(target, seed=args.seed, nsample=args.nsample, burnin=args.burnin, thinning=args.thinning, savepath=savepath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default='20251016')
    parser.add_argument('--rootdir', type=str, default='experiments/results')
    parser.add_argument('--algo', type=str, default='scp', choices=['scp', 'nuts', 'hmc'])
    parser.add_argument('--d', type=int, default=10)
    parser.add_argument('--latitude', type=float, default=1.5)
    parser.add_argument('--nsample', type=int, default=10000)
    parser.add_argument('--burnin', type=int, default=100)
    parser.add_argument('--stepsize', type=float, default=.1)
    parser.add_argument('--thinning', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    run(args)
