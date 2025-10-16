import os
import argparse
import jax
import distrax
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt
from numpyro.infer import HMC, MCMC
from src.scp_core import SCP
from experiments.targets import SkewMultivariateStudentT

def run(d, latitude, nsample, burnin, stepsize, thinning=1, seed=0, algo='stepout', df=5, alpha_scale=100., init='warm', savepath=None):
    loc = jnp.zeros(d)
    scale_tril = jnp.eye(d)
    alpha = jnp.zeros(d)
    alpha = alpha.at[0].set(alpha_scale)
    alpha = alpha.at[1].set(-alpha_scale)

    target = SkewMultivariateStudentT(loc, scale_tril, df, alpha)
    exact_samples = target.sample(seed=jax.random.key(0), sample_shape=(1000,))
    
    scp_model = SCP(d=d, latitude=latitude) 
    print("Initializing SCP parameters......")
    opt_params, losses = scp_model.minimize_reverse_kl(target.log_prob, 
                                                       seed=0, 
                                                       ntrain=2000, 
                                                       learning_rate=0.01)

    print("Running RWM on the bright side......")
    if init == 'warm':
        x0 = jnp.zeros(d) + 1.
    else:
        x0 = jnp.zeros(d) + 100.
    u0 = scp_model.inverse_projection(opt_params, x0)
    scp_samples, scp_accept_prob = scp_model.rwm_bright_side(target.log_prob,
                                                          opt_params, 
                                                          seed=seed, 
                                                          x0=u0,
                                                          stepsize=stepsize,
                                                          nsample=nsample, 
                                                          burnin=burnin,
                                                          thinning=thinning,
                                                          algo=algo)
    print('SCP acceptance rate:', scp_accept_prob)

    print("Running HMC......")
    hmc_kernel = HMC(potential_fn=lambda z: -target.log_prob(z), 
                     step_size=0.1, 
                     adapt_step_size=False, 
                     adapt_mass_matrix=False, 
                     num_steps=10, 
                     trajectory_length=None)
    mcmc = MCMC(hmc_kernel, 
                num_warmup=0, 
                num_samples=nsample, 
                thinning=thinning, 
                num_chains=1)
    mcmc.run(jax.random.key(seed), init_params=x0, extra_fields=("accept_prob",))
    hmc_samples = mcmc.get_samples()
    print("HMC acceptance rate:", jnp.mean(mcmc.get_extra_fields()['accept_prob']))

    exact_samples = target.sample(seed=jax.random.key(0), sample_shape=(10_000_000,))

    ps = jnp.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99])
    exact_quantiles = pd.DataFrame(jnp.quantile(exact_samples, ps, axis=0), index=ps, columns=[f'Exact{i}' for i in range(d)])
    scp_quantiles = pd.DataFrame(jnp.quantile(scp_samples, ps, axis=0), index=ps, columns=[f'SCP{i}' for i in range(d)])
    hmc_quantiles = pd.DataFrame(jnp.quantile(hmc_samples, ps, axis=0), index=ps, columns=[f'HMC{i}' for i in range(d)])

    if savepath is not None:

        filename = f'{savepath}/skewt_df{df}_d{d}_lat{latitude}_nsample{nsample}_burnin{burnin}_init{init}_stepsize{stepsize}_{algo}_seed{seed}'

        # save quantiles
        quantiles = pd.concat([exact_quantiles, scp_quantiles, hmc_quantiles], axis=1)
        quantiles.to_csv(f'{filename}.csv')

        # qq plot
        idx_to_plot = jnp.arange(0, d, d//10)
        fig, ax = plt.subplots(2, 5, figsize=(10, 5))
        for i in range(2):
            for j in range(5):
                idx = int(idx_to_plot[i * 5 + j])
                ax[i, j].plot(exact_quantiles.iloc[:, idx], hmc_quantiles.iloc[:, idx], marker='o', label='HMC')
                ax[i, j].plot(exact_quantiles.iloc[:, idx], scp_quantiles.iloc[:, idx], marker='*', label='SCP')
                ax[i, j].set_title(f'x{idx}')
                ax[i, j].plot(exact_quantiles.iloc[:, idx], exact_quantiles.iloc[:, idx], 'r--')
                if i == 0 and j == 0:
                    ax[i, j].legend()
        fig.suptitle('d={}, latitude={}, stepsize={}, acceptance rate={:.4f}'.format(d, latitude, stepsize, scp_accept_prob), fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{filename}.pdf')
        plt.close()

        # trace plot
        fig, ax = plt.subplots(1, 2, figsize=(6, 3))
        for j in range(2):
            ax[0].plot(scp_samples[:, j], label=rf'$x_{{{j+1}}}$')
            ax[1].plot(hmc_samples[:, j])
        ax[0].set_title('SCP')
        ax[0].legend()
        ax[1].set_title('HMC')
        plt.tight_layout()
        plt.savefig(f'{filename}_trace.pdf')
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default='20251015')
    parser.add_argument('--rootdir', type=str, default='experiments/results')
    parser.add_argument('--d', type=int, default=10)
    parser.add_argument('--latitude', type=float, default=1.5)
    parser.add_argument('--nsample', type=int, default=10000)
    parser.add_argument('--burnin', type=int, default=100)
    parser.add_argument('--stepsize', type=float, default=.1)
    parser.add_argument('--thinning', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--algo', type=str, default='stepout', choices=['stepout', 'reject'])
    parser.add_argument('--df', type=float, default=2)
    parser.add_argument('--alpha_scale', type=float, default=100.)
    parser.add_argument('--init', type=str, default='warm', choices=['warm', 'cold'])
    args = parser.parse_args()
    
    savepath = os.path.join(args.rootdir, args.date, 'skewt')
    os.makedirs(savepath, exist_ok=True)

    run(d=args.d,
        latitude=args.latitude,
        nsample=args.nsample,
        burnin=args.burnin,
        stepsize=args.stepsize,
        thinning=args.thinning,
        seed=args.seed,
        algo=args.algo,
        df=args.df,
        alpha_scale=args.alpha_scale,
        init=args.init,
        savepath=savepath)
