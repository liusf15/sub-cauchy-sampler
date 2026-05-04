import argparse
import os
import time
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpyro.infer import HMC, MCMC
import pandas as pd

from src.cauchy_mh import independent_cauchy_mh
from src.scp_core import SCP
from experiments.targets import SkewMultivariateStudentT


def run(
    d,
    latitude,
    nsample,
    burnin,
    stepsize,
    thinning=1,
    seed=0,
    algo='stepout',
    df=5,
    alpha_scale=100.0,
    init='warm',
    affine='scalar',
    nexact=10_000_000,
    savepath=None,
):
    loc = jnp.zeros(d)
    scale_tril = jnp.eye(d)
    alpha = jnp.zeros(d)
    alpha = alpha.at[0].set(alpha_scale)
    alpha = alpha.at[1].set(-alpha_scale)

    target = SkewMultivariateStudentT(loc, scale_tril, df, alpha)
    scp_model = SCP(d=d, latitude=latitude, affine=affine)

    print("Initializing SCP parameters......")
    opt_params, _ = scp_model.minimize_reverse_kl(
        target.log_prob,
        seed=0,
        ntrain=512,
        max_iter=1000,
        learning_rate=0.01,
    )

    if init == 'warm':
        x0 = jnp.ones(d)
    else:
        x0 = jnp.zeros(d) + 100.0

    print("Running RWM on the bright side......")
    start = time.time()
    scp_samples, scp_accept_prob = scp_model.rwm_bright_side(
        target.log_prob,
        opt_params,
        seed=seed,
        x0=scp_model.inverse_projection(opt_params, x0),
        stepsize=stepsize,
        nsample=nsample,
        burnin=burnin,
        thinning=thinning,
        algo=algo,
    )
    scp_time = time.time() - start
    print('SCP acceptance rate:', scp_accept_prob)

    print("Running HMC......")
    start = time.time()
    hmc_kernel = HMC(
        potential_fn=lambda z: -target.log_prob(z),
        step_size=0.1,
        adapt_step_size=False,
        adapt_mass_matrix=False,
        num_steps=10,
        trajectory_length=None,
    )
    mcmc = MCMC(
        hmc_kernel,
        num_warmup=0,
        num_samples=nsample,
        thinning=thinning,
        num_chains=1,
    )
    mcmc.run(jax.random.key(seed), init_params=x0, extra_fields=("accept_prob",))
    hmc_samples = mcmc.get_samples()
    hmc_time = time.time() - start
    hmc_accept_rate = jnp.mean(mcmc.get_extra_fields()['accept_prob'])
    print("HMC acceptance rate:", hmc_accept_rate)

    print("Running independent Cauchy MH......")
    start = time.time()
    imh_samples, imh_accept_rate = independent_cauchy_mh(
        target.log_prob,
        x0,
        jax.random.key(seed + 1),
        nsample=nsample,
        burnin=burnin,
        thinning=thinning,
    )
    imh_time = time.time() - start
    print("IMH acceptance rate:", imh_accept_rate)

    plot_indices = [0, 1, 2, 3]
    if d < len(plot_indices):
        raise ValueError("Quantile summaries and QQ plots require d >= 4 because indices are fixed to 0, 1, 2, 3.")

    exact_samples = target.sample(seed=jax.random.key(0), sample_shape=(nexact,))

    ps = jnp.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99])
    exact_quantiles = pd.DataFrame(
        jnp.quantile(exact_samples[:, plot_indices], ps, axis=0),
        index=ps,
        columns=[f'Exact{i}' for i in plot_indices],
    )
    scp_quantiles = pd.DataFrame(
        jnp.quantile(scp_samples[:, plot_indices], ps, axis=0),
        index=ps,
        columns=[f'SCP{i}' for i in plot_indices],
    )
    hmc_quantiles = pd.DataFrame(
        jnp.quantile(hmc_samples[:, plot_indices], ps, axis=0),
        index=ps,
        columns=[f'HMC{i}' for i in plot_indices],
    )
    imh_quantiles = pd.DataFrame(
        jnp.quantile(imh_samples[:, plot_indices], ps, axis=0),
        index=ps,
        columns=[f'IMH{i}' for i in plot_indices],
    )

    meta_data = {
        'scp_accept_rate': float(scp_accept_prob),
        'scp_time': scp_time,
        'hmc_accept_rate': float(hmc_accept_rate),
        'hmc_time': hmc_time,
        'imh_accept_rate': float(imh_accept_rate),
        'imh_time': imh_time,
    }

    if savepath is None:
        return

    filename = (
        f'{savepath}/skewt_df{df}_d{d}_lat{latitude}_nsample{nsample}_burnin{burnin}'
        f'_init{init}_stepsize{stepsize}_{algo}_affine{affine}_seed{seed}'
    )

    quantiles = pd.concat([exact_quantiles, scp_quantiles, hmc_quantiles, imh_quantiles], axis=1)
    quantiles.to_csv(f'{filename}.csv')
    pd.DataFrame(meta_data, index=[0]).to_csv(f'{filename}_meta.csv', index=False)

    fig, ax = plt.subplots(1, len(plot_indices), figsize=(2.8 * len(plot_indices), 2.8), squeeze=False)
    flat_axes = ax.reshape(-1)
    for local_idx, (plot_ax, idx) in enumerate(zip(flat_axes, plot_indices)):
        plot_ax.plot(exact_quantiles.iloc[:, local_idx], hmc_quantiles.iloc[:, local_idx], marker='o', label='HMC')
        plot_ax.plot(exact_quantiles.iloc[:, local_idx], scp_quantiles.iloc[:, local_idx], marker='*', label='SCP')
        plot_ax.plot(exact_quantiles.iloc[:, local_idx], imh_quantiles.iloc[:, local_idx], marker='s', label='IMH')
        plot_ax.plot(exact_quantiles.iloc[:, local_idx], exact_quantiles.iloc[:, local_idx], 'r--')
        plot_ax.set_title(f'x{idx}')
    flat_axes[0].legend()
    fig.suptitle(
        f'd={d}, latitude={latitude}, stepsize={stepsize}, affine={affine}, scp acc={float(scp_accept_prob):.4f}',
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(f'{filename}.pdf')
    plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    for j in range(min(2, d)):
        ax[0].plot(scp_samples[:, j], label=rf'$x_{{{j + 1}}}$')
        ax[1].plot(hmc_samples[:, j], label=rf'$x_{{{j + 1}}}$')
        ax[2].plot(imh_samples[:, j], label=rf'$x_{{{j + 1}}}$')
    ax[0].set_title('SCP')
    ax[1].set_title('HMC')
    ax[2].set_title('IMH')
    ax[0].legend()
    plt.tight_layout()
    plt.savefig(f'{filename}_trace.pdf')
    print(f"Saved results to {filename}.csv\n {filename}_meta.csv\n {filename}.pdf\n {filename}_trace.pdf")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default='20260504')
    parser.add_argument('--rootdir', type=str, default='experiments/results')
    parser.add_argument('--d', type=int, default=100)
    parser.add_argument('--latitude', type=float, default=1.1)
    parser.add_argument('--affine', type=str, default='scalar', choices=['scalar', 'covariance'])
    parser.add_argument('--nsample', type=int, default=500_000)
    parser.add_argument('--burnin', type=int, default=100)
    parser.add_argument('--stepsize', type=float, default=.1)
    parser.add_argument('--thinning', type=int, default=50)
    parser.add_argument('--nexact', type=int, default=10_000_000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--algo', type=str, default='stepout', choices=['stepout', 'reject'])
    parser.add_argument('--df', type=float, default=1)
    parser.add_argument('--alpha_scale', type=float, default=100.)
    parser.add_argument('--init', type=str, default='warm', choices=['warm', 'cold'])
    args = parser.parse_args()

    savepath = os.path.join(args.rootdir, args.date, 'skewt')
    os.makedirs(savepath, exist_ok=True)

    run(
        d=args.d,
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
        affine=args.affine,
        nexact=args.nexact,
        savepath=savepath,
    )
