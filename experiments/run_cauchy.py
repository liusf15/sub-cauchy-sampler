import argparse
import jax
import jax.numpy as jnp
from numpyro.infer import MCMC, HMC
from src.scp_core import SCP
from experiments.targets import MultivariateStudentT

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def RWM(log_prob, x0, key, nsamples, stepsize):
    def random_walk_step(carry, i):
        x, key = carry
        key, subkey = jax.random.split(key)
        x_proposal = x + stepsize * jax.random.normal(subkey, shape=x.shape)
        log_accept_ratio = log_prob(x_proposal) - log_prob(x)
        key, subkey = jax.random.split(key)
        accept = jnp.log(jax.random.uniform(subkey)) < log_accept_ratio
        x = jnp.where(accept, x_proposal, x)
        return (x, key), (x, accept)
    return jax.lax.scan(random_walk_step, (x0, key), jnp.arange(nsamples))[1]

def run_scp(target, lat, x0, key, nsamples, stepsize, optimize_params=True):
    d = target.d
    scp_model = SCP(d, latitude=lat)
    if not optimize_params:
        params = {
                    'observer': jnp.zeros(d),
                    'shift': jnp.zeros(d),
                    'scale': 0. # log scale
                }
    else:
        params = scp_model.minimize_reverse_kl(target.log_prob, seed=1, ntrain=2000, learning_rate=0.01, max_iter=2000)[0]
    # params_rootd = {
    #         'observer': jnp.zeros(d),
    #         'shift': jnp.zeros(d),
    #         'scale': jnp.log(jnp.sqrt(d) * (1 - 1 / lat) )
    #     }

    u0 = scp_model.inverse_projection(params, x0)
    samples, accept_prob = scp_model.rwm_bright_side(target.log_prob, params, seed=key, x0=u0, stepsize=stepsize, nsample=nsamples, burnin=0, thinning=1)
    avg_latitude = jax.vmap(scp_model.inverse_projection, in_axes=(None, 0))(params, samples)[:, -1].mean()

    return jnp.sum(samples**2, 1), accept_prob, avg_latitude

def run(d, latitude=1.1, init='warm', seed=0, nsamples=10_000, savepath='experiments/plots'):
    target = MultivariateStudentT(d=d, df=1.0)
    exact_samples = target.sample(seed=jax.random.key(0), sample_shape=(50000,))

    key = jax.random.key(seed)
    key, subkey = jax.random.split(key)
    if init == 'warm':
        x0 = target.sample(seed=subkey, sample_shape=())
    else:
        x0 = jnp.zeros(d) + 1000.

    all_samples = {}
    accept_prob = {}
    print("Running RWM......")
    rwm_samples, rwm_accept_prob = RWM(target.log_prob, x0, key, nsamples=nsamples, stepsize=0.25)
    all_samples['rwm'] = jnp.sum(rwm_samples**2, 1)
    accept_prob['rwm'] = rwm_accept_prob.mean()

    print("Running HMC......")
    hmc_kernel = HMC(potential_fn=lambda z: -target.log_prob(z), step_size=1., adapt_step_size=False, adapt_mass_matrix=False, num_steps=5, trajectory_length=None)
    mcmc = MCMC(hmc_kernel, num_warmup=0, num_samples=nsamples, thinning=1, num_chains=1)
    mcmc.run(key, init_params=x0, extra_fields=("accept_prob", ))
    hmc_samples = mcmc.get_samples()
    all_samples['hmc'] = jnp.sum(hmc_samples**2, 1)
    accept_prob['hmc'] = mcmc.get_extra_fields()['accept_prob'].mean()

    print("Running SPS......")
    all_samples['sps'], accept_prob['sps'], sps_avg_lat = run_scp(target, 2., x0, key, nsamples, 0.02, optimize_params=True)
    print('avg latitude (SPS):', sps_avg_lat)

    print("Running SCP......")
    all_samples['scp'], accept_prob['scp'], scp_avg_lat = run_scp(target, latitude, x0, key, nsamples, 10., optimize_params=True)
    print('avg latitude (SCP):', scp_avg_lat)

    print(pd.DataFrame(accept_prob, index=[0]))
    pd.DataFrame(accept_prob, index=[0]).to_csv(f'{savepath}/cauchy_accept_prob_d{d}_lat{latitude}_init{init}_seed{seed}.csv', index=False)
    pd.DataFrame(all_samples).to_csv(f'{savepath}/cauchy_samples_d{d}_lat{latitude}_init{init}_seed{seed}.csv', index=False)

    plt.figure(figsize=(6, 4))
    sns.set_theme(context='paper', style="whitegrid", font_scale=1.5)
    plt.plot(all_samples['scp'], label='SCP', alpha=.6, c='orangered')
    plt.plot(all_samples['hmc'], label='HMC', c='deepskyblue')
    plt.plot(all_samples['sps'], label='SPS', c='green')
    plt.plot(all_samples['rwm'], label='RWM', c='yellow')
    plt.yscale('log', base=10)
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel(r'$\|X\|^2$')
    plt.tight_layout()
    plt.savefig(f'{savepath}/cauchy_trace_d{d}_lat{latitude}_init{init}_seed{seed}.pdf')
    plt.close()

    plt.figure(figsize=(4, 3))
    sns.set_theme(context='paper', style="whitegrid", font_scale=1.5)
    sns.kdeplot(jnp.log(jnp.sum(exact_samples**2, 1)), c='deepskyblue', lw=3, label='Exact')
    sns.kdeplot(jnp.log(all_samples['scp']), c='orangered', ls='--', lw=3, label='SCP')
    plt.legend()
    plt.xlabel(r'$\log_{10}\|X\|^2$')
    plt.tight_layout()
    plt.savefig(f'{savepath}/cauchy_kde_d{d}_lat{latitude}_init{init}_seed{seed}.pdf')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=100)
    parser.add_argument("--init", type=str, default='warm')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lat", type=float, default=1.1)
    args = parser.parse_args()
    
    run(d=args.d, latitude=args.lat, init=args.init, seed=args.seed, nsamples=10_000)
