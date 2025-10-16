import os
import argparse
import jax
import distrax
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt
from jax.scipy.special import gammaln, betainc
from numpyro.infer import HMC, MCMC
from src.scp_core import SCP
from experiments.targets import Banana_t, skewt

def student_t_pdf(x, df):
    # t-pdf: c * (1 + x^2/df)^(-(df+1)/2)
    c = jnp.exp(
        gammaln((df + 1.0) / 2.0) - gammaln(df / 2.0)
        - 0.5 * jnp.log(df * jnp.pi)
    )
    return c * jnp.power(1.0 + (x * x) / df, -(df + 1.0) / 2.0)

def student_t_cdf(x, df):
    """
    JAX-compatible Student-t CDF using the regularized incomplete beta function.
    """
    x = jnp.asarray(x)
    t2 = x**2
    a = 0.5 * df
    b = 0.5
    ib = betainc(a, b, df / (df + t2))  # regularized incomplete beta
    cdf = jnp.where(x > 0, 1.0 - 0.5 * ib, 0.5 * ib)
    return cdf

@jax.custom_jvp
def student_t_cdf(x, df):
    # primal value via incomplete beta, but clamp z away from 1 to avoid NaNs
    v = df
    z = v / (v + x * x)
    # keep strictly < 1 in float64; this avoids hitting the singular endpoint
    eps = jnp.finfo(x.dtype).eps
    z = jnp.clip(z, 0.0, 1.0 - eps)
    a = 0.5 * v
    b = 0.5
    ib = betainc(a, b, z)
    # piecewise for sign; this is safe since z is now < 1
    return jnp.where(x >= 0.0, 1.0 - 0.5 * ib, 0.5 * ib)

# exact jvp: d/dx CDF = PDF
@student_t_cdf.defjvp
def _student_t_cdf_jvp(primals, tangents):
    x, df = primals
    xdot, dfdot = tangents
    y = student_t_cdf(x, df)
    # we ignore gradients w.r.t. df (NumPyro only needs grads in x)
    ydot = student_t_pdf(x, df) * (xdot if isinstance(xdot, jax.Array) or jnp.ndim(xdot) else xdot)
    return y, ydot

class SkewMultivariateStudentT(distrax.Distribution):
    def __init__(self, loc, scale_tril, df, alpha):
        """
        Multivariate skew-t distribution (Azzalini-Capitanio form).

        Y ~ ST_d(loc, scale_tril @ scale_tril^T, alpha, df)

        Args:
            loc: Location vector of shape (d,)
            scale_tril: Lower-triangular Cholesky factor of scale matrix, shape (d,d)
            df: Degrees of freedom (>0)
            alpha: Shape (skewness) vector of shape (d,)
        """
        self.loc = jnp.asarray(loc)
        self.scale_tril = jnp.asarray(scale_tril)
        self.df = df
        self.alpha = jnp.asarray(alpha)
        self.d = self.loc.shape[0]
        assert self.scale_tril.shape == (self.d, self.d)
        assert self.alpha.shape == (self.d,)

        # Precompute inverse and log|det|
        self.scale_inv = jnp.linalg.inv(self.scale_tril)
        self.log_det = jnp.sum(jnp.log(jnp.diag(self.scale_tril)))

        # Normalization constant for symmetric t kernel
        self._log_norm_const = (
            gammaln((df + self.d) / 2.0)
            - gammaln(df / 2.0)
            - 0.5 * self.d * jnp.log(df * jnp.pi)
            - self.log_det
        )

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        log f_Y(x) = log[2 * t_d(z; 0, I, df) * T_{df+d}( alpha^T z * sqrt((df+d)/(df+||z||^2)) )]
        where z = scale_inv @ (x - loc).
        """
        z = jnp.einsum("ij,...j->...i", self.scale_inv, x - self.loc)
        norm2 = jnp.sum(z**2, axis=-1)
        log_kernel = -0.5 * (self.df + self.d) * jnp.log1p(norm2 / self.df)
        log_t_d = self._log_norm_const + log_kernel

        # Skewing term
        az = jnp.dot(z, self.alpha)
        scale = jnp.sqrt((self.df + self.d) / (self.df + norm2))
        arg = az * scale
        cdf = student_t_cdf(arg, df=self.df + self.d)
        cdf = jnp.clip(cdf, jnp.finfo(x.dtype).tiny, 1.0)

        return jnp.log(2.0) + log_t_d + jnp.log(cdf)

    def _sample_n(self, seed: jax.random.PRNGKey, sample_shape=()):
        """
        Stochastic representation:
            W ~ ChiSq(df)/df
            Z ~ SkewNormal_d(0, I, alpha)
            X = loc + scale_tril @ (Z / sqrt(W))
        """
        key_u0, key_v, key_g = jax.random.split(seed, 3)
        shape = (sample_shape,) + (self.d,)

        # Generate skew-normal Z
        alpha = self.alpha
        alpha_sq = jnp.dot(alpha, alpha)
        delta = alpha / jnp.sqrt(1.0 + alpha_sq)
        dn2 = jnp.dot(delta, delta)
        dn = jnp.sqrt(dn2)
        u = jnp.where(dn > 0, delta / dn, delta)

        U0 = jax.random.normal(key_u0, sample_shape)
        V = jax.random.normal(key_v, shape)

        coeff = (1.0 - jnp.sqrt(1.0 - dn2))
        uTv = jnp.einsum("d,...d->...", u, V)
        SV = V - coeff * u[None, ...] * uTv[..., None]
        Z = delta[None, ...] * jnp.abs(U0)[..., None] + SV

        # Student-t scaling
        chi2 = 2.0 * jax.random.gamma(key_g, self.df / 2.0, shape=sample_shape)
        W = chi2 / self.df
        Y = self.loc + jnp.einsum("ij,...j->...i", self.scale_tril, Z / jnp.sqrt(W)[..., None])
        return Y

    @property
    def event_shape(self):
        return (self.d,)

    @property
    def batch_shape(self):
        return ()
    
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
    mcmc.run(jax.random.key(1), init_params=x0, extra_fields=("accept_prob",))
    hmc_samples = mcmc.get_samples()
    print("HMC acceptance rate:", jnp.mean(mcmc.get_extra_fields()['accept_prob']))

    exact_samples = target.sample(seed=jax.random.key(0), sample_shape=(1_000_000,))

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
