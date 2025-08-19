import os
import argparse
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt

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
                                                       seed=0, ntrain=2000, learning_rate=0.01)

    print("Running RWM on the bright side......")
    mcmc_samples, accept_prob = scp_model.rwm_bright_side(target.log_prob,
                                                          opt_params, seed=seed, stepsize=stepsize,nsample=nsample, 
                                                          burnin=burnin,
                                                          thinning=thinning,algo=algo)
    print('Acceptance rate:', accept_prob)

    exact_samples = target.sample(seed, n=10_000_000)

    ps = jnp.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99])
    exact_quantiles = pd.DataFrame(jnp.quantile(exact_samples, ps, axis=0), index=ps, columns=[f'Exact{i}' for i in range(d)])
    mcmc_quantiles = pd.DataFrame(jnp.quantile(mcmc_samples, ps, axis=0), index=ps, columns=[f'MCMC{i}' for i in range(d)])

    if savepath is not None:
        if target_name == 'Banana_t':
            filename = f'{savepath}/{target_name}_df{df}_d{d}_lat{latitude}_nsample{nsample}_burnin{burnin}_stepsize{stepsize}_{algo}_seed{seed}'
        elif target_name == 'skewt':
            filename = f'{savepath}/{target_name}_df{df1}_{df2}_d{d}_lat{latitude}_nsample{nsample}_burnin{burnin}_stepsize{stepsize}_{algo}_seed{seed}'

        quantiles = pd.concat([exact_quantiles, mcmc_quantiles], axis=1)
        quantiles.to_csv(f'{filename}.csv')

        idx_to_plot = jnp.arange(0, d, d//10)
        fig, ax = plt.subplots(2, 5, figsize=(10, 5))
        for i in range(2):
            for j in range(5):
                idx = int(idx_to_plot[i * 5 + j])
                if idx == 10:
                    idx = 1
                ax[i, j].plot(exact_quantiles.iloc[:, idx], mcmc_quantiles.iloc[:, idx], marker='o')
                ax[i, j].set_title(f'x{idx}')
                ax[i, j].plot(exact_quantiles.iloc[:, idx], exact_quantiles.iloc[:, idx], 'r--')
        fig.suptitle('d={}, latitude={}, stepsize={}, acceptance rate={:.4f}'.format(d, latitude, stepsize, accept_prob), fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{filename}.pdf')
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default='20250807')
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
