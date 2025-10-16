import os
import argparse
import jax
import jax.numpy as jnp
import pandas as pd
from numpyro.infer import HMC, MCMC
import matplotlib.pyplot as plt
import time

from src.scp_core import SCP
from experiments.targets import Banana_t, skewt

def run(target_name, d, latitude, nsample, burnin, stepsize, thinning=1, seed=0, algo='stepout', df=5, df1=2, df2=3, savepath=None):
    if target_name == 'Banana_t':
        target = Banana_t(d=d, df=df)
    elif target_name == 'skewt':
        target = skewt(a=jnp.linspace(df1/2, df2/2, d), 
                       b=jnp.linspace(df2/2, df1/2, d))
    else:
        raise ValueError("Unknown target")
    
    scp_model = SCP(d=d, latitude=latitude) 
    print("Initializing SCP parameters......")
    opt_params, losses = scp_model.minimize_reverse_kl(target.log_prob, 
                                                       seed=0, 
                                                       ntrain=2000, 
                                                       learning_rate=0.01)

    print("Running RWM on the bright side......")
    start = time.time()
    scp_samples, scp_accept_prob = scp_model.rwm_bright_side(target.log_prob,
                                                                opt_params, 
                                                                seed=seed, 
                                                                stepsize=stepsize,
                                                                nsample=nsample, 
                                                                burnin=burnin,
                                                                thinning=thinning,
                                                                algo=algo)
    scp_time = time.time() - start
    print('SCP acceptance rate:', scp_accept_prob, 'Time:', scp_time)

    print("Running HMC......")
    start = time.time()
    hmc_kernel = HMC(potential_fn=lambda z: -target.log_prob(z), 
                     step_size=0.5, 
                     adapt_step_size=False, 
                     adapt_mass_matrix=False, 
                     num_steps=10, 
                     trajectory_length=None)
    mcmc = MCMC(hmc_kernel, 
                num_warmup=0, 
                num_samples=nsample, 
                thinning=thinning, 
                num_chains=1)
    # x0 = jnp.zeros(d)
    key1, key2 = jax.random.split(jax.random.key(seed))
    x0 = jax.random.normal(key1, (d,))
    mcmc.run(key2, init_params=x0, extra_fields=("accept_prob",))
    hmc_samples = mcmc.get_samples()
    hmc_time = time.time() - start
    hmc_accept_prob = jnp.mean(mcmc.get_extra_fields()['accept_prob'])
    print('HMC acceptance rate:', hmc_accept_prob, 'Time:', hmc_time)

    ps = jnp.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99])

    exact_samples = target.sample(0, n=10_000_000)
    # if target_name == 'Banana_t':
    #     exact_quantiles = t.ppf(ps, df)
    #     exact_quantiles = jnp.tile(exact_quantiles, (d, 1)).T
    #     exact_quantiles = pd.DataFrame(exact_quantiles, index=ps, columns=[f'Exact{i}' for i in range(d)])
    #     _z = jax.random.normal(jax.random.key(0), (1000000, 2))
    #     exact_quantiles.iloc[:, 1] = jnp.quantile(_z[:, 0]**2 + _z[:, 1], ps)

    exact_quantiles = pd.DataFrame(jnp.quantile(exact_samples, ps, axis=0), index=ps, columns=[f'Exact{i}' for i in range(d)])
    scp_quantiles = pd.DataFrame(jnp.quantile(scp_samples, ps, axis=0), index=ps, columns=[f'SCP{i}' for i in range(d)])
    hmc_quantiles = pd.DataFrame(jnp.quantile(hmc_samples, ps, axis=0), index=ps, columns=[f'HMC{i}' for i in range(d)])

    if savepath is not None:
        if target_name == 'Banana_t':
            filename = f'{savepath}/{target_name}_df{df}_d{d}_lat{latitude}_nsample{nsample}_burnin{burnin}_stepsize{stepsize}_{algo}_initrandom_seed{seed}'
        elif target_name == 'skewt':
            filename = f'{savepath}/{target_name}_df{df1}_{df2}_d{d}_lat{latitude}_nsample{nsample}_burnin{burnin}_stepsize{stepsize}_{algo}_seed{seed}'

        quantiles = pd.concat([exact_quantiles, scp_quantiles, hmc_quantiles], axis=1)
        quantiles.to_csv(f'{filename}.csv')

        meta_data = {'scp_accept_prob': scp_accept_prob,
                     'scp_time': scp_time,
                     'hmc_accept_prob': hmc_accept_prob,
                     'hmc_time': hmc_time}
        meta_df = pd.DataFrame(meta_data, index=[0])
        meta_df.to_csv(f'{filename}_meta.csv', index=False)

        idx_to_plot = jnp.arange(0, d, d//10)
        fig, ax = plt.subplots(2, 5, figsize=(10, 5))
        for i in range(2):
            for j in range(5):
                idx = int(idx_to_plot[i * 5 + j])
                if idx == 10:
                    idx = 1
                ax[i, j].plot(exact_quantiles.iloc[:, idx], scp_quantiles.iloc[:, idx], marker='o')
                ax[i, j].plot(exact_quantiles.iloc[:, idx], hmc_quantiles.iloc[:, idx], marker='x')
                ax[i, j].set_title(f'x{idx}')
                ax[i, j].plot(exact_quantiles.iloc[:, idx], exact_quantiles.iloc[:, idx], 'r--')
        fig.suptitle('d={}, latitude={}, stepsize={}, acceptance rate={:.4f}'.format(d, latitude, stepsize, scp_accept_prob), fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{filename}.pdf')
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default='20251015')
    parser.add_argument('--rootdir', type=str, default='experiments/results')
    parser.add_argument('--target', type=str, default='skewt', choices=['Banana_t', 'skewt'])
    parser.add_argument('--d', type=int, default=10)
    parser.add_argument('--latitude', type=float, default=1.5)
    parser.add_argument('--nsample', type=int, default=10000)
    parser.add_argument('--burnin', type=int, default=100)
    parser.add_argument('--stepsize', type=float, default=1.)
    parser.add_argument('--thinning', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--algo', type=str, default='stepout', choices=['stepout', 'reject'])
    parser.add_argument('--df', type=float, default=5)
    parser.add_argument('--df1', type=float, default=2)
    parser.add_argument('--df2', type=float, default=3)
    args = parser.parse_args()
    
    savepath = os.path.join(args.rootdir, args.date, args.target)
    os.makedirs(savepath, exist_ok=True)

    run(target_name=args.target,
        d=args.d,
        latitude=args.latitude,
        nsample=args.nsample,
        burnin=args.burnin,
        stepsize=args.stepsize,
        thinning=args.thinning,
        seed=args.seed,
        algo=args.algo,
        df=args.df,
        df1=args.df1,
        df2=args.df2,
        savepath=savepath)
