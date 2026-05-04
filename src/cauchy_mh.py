import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
import jax_tqdm

def standard_multivariate_student_t_logpdf(x, df):
    x = jnp.asarray(x)
    d = x.shape[-1]
    norm2 = jnp.sum(x**2, axis=-1)
    log_norm = (
        gammaln((df + d) / 2.0)
        - gammaln(df / 2.0)
        - 0.5 * d * jnp.log(df * jnp.pi)
    )
    log_kernel = -0.5 * (df + d) * jnp.log1p(norm2 / df)
    return log_norm + log_kernel


def standard_multivariate_cauchy_logpdf(x, scale=1.0):
    x = jnp.asarray(x)
    scale = jnp.asarray(scale)
    d = x.shape[-1]
    return standard_multivariate_student_t_logpdf(x / scale, df=1.0) - d * jnp.log(scale)


def standard_multivariate_cauchy_sample(key, d, n=1, scale=1.0):
    key1, key2 = jax.random.split(key)
    z = jax.random.normal(key1, shape=(n, d))
    g = 2.0 * jax.random.gamma(key2, 0.5, shape=(n,))
    return scale * z / jnp.sqrt(g[:, None])


def independent_cauchy_mh(logp_fn, x0, key, nsample, burnin=0, thinning=1, stepsize=1.0):
    """
    Independent Metropolis-Hastings with a scaled multivariate Cauchy proposal.
    """
    total_steps = nsample + burnin
    x0 = jnp.asarray(x0)
    logp0 = logp_fn(x0)
    logq0 = standard_multivariate_cauchy_logpdf(x0, scale=stepsize)

    @jax_tqdm.scan_tqdm(total_steps)
    def mh_step(carry, i):
        x, logp_x, logq_x, key = carry
        key, proposal_key, accept_key = jax.random.split(key, 3)
        y = standard_multivariate_cauchy_sample(proposal_key, x.shape[0], n=1, scale=stepsize)[0]
        logp_y = logp_fn(y)
        logq_y = standard_multivariate_cauchy_logpdf(y, scale=stepsize)
        log_accept = logp_y + logq_x - logp_x - logq_y
        accept = (jnp.log(jax.random.uniform(accept_key)) < log_accept) * 1.0
        x = x * (1.0 - accept) + y * accept
        logp_x = logp_x * (1.0 - accept) + logp_y * accept
        logq_x = logq_x * (1.0 - accept) + logq_y * accept
        return (x, logp_x, logq_x, key), (x, accept)

    _, (samples, accepts) = jax.lax.scan(
        mh_step,
        (x0, logp0, logq0, key),
        jnp.arange(total_steps),
    )
    return samples[burnin::thinning], jnp.mean(accepts[burnin::thinning])
