import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
import distrax
from typing import Union
from scipy.stats import jf_skew_t, skewcauchy, cauchy

class StudentT(distrax.Distribution):
    """Student's t-distribution: location-scale version."""

    def __init__(self, df: float, loc: Union[float, jnp.ndarray], scale: Union[float, jnp.ndarray]):
        self.df = jnp.asarray(df)
        self.loc = jnp.asarray(loc)
        self.scale = jnp.asarray(scale)

    def log_prob(self, x):
        df = self.df
        z = (x - self.loc) / self.scale
        log_norm = (
            gammaln((df + 1) / 2)
            - gammaln(df / 2)
            - 0.5 * jnp.log(df * jnp.pi)
            - jnp.log(self.scale)
        )
        log_kernel = -0.5 * (df + 1) * jnp.log1p((z ** 2) / df)
        return log_norm + log_kernel

    def prob(self, x):
        return jnp.exp(self.log_prob(x))

    def _sample_n(self, seed, sample_shape=()):
        """Sample using standard method: Z = loc + scale * T, where T ~ t(df)."""
        key1, key2 = jax.random.split(seed)
        g = jax.random.gamma(key1, self.df / 2, shape=sample_shape) * 2
        z = jax.random.normal(key2, shape=sample_shape)
        t = z / jnp.sqrt(g / self.df)
        return self.loc + self.scale * t

    @property
    def event_shape(self):
        return ()

    @property
    def batch_shape(self):
        return jnp.broadcast_shapes(
            jnp.shape(self.df), jnp.shape(self.loc), jnp.shape(self.scale)
        )


class MultivariateStudentT(distrax.Distribution):
    def __init__(self, df: float, dim: int):
        """
        Standard multivariate Student's t-distribution with zero mean and identity covariance.

        Args:
            df: Degrees of freedom (> 0), scalar
            dim: Dimension (d >= 1)
        """
        self.df = df
        self.dim = dim

        # Precompute log normalization constant
        self._log_norm_const = (
            gammaln((df + dim) / 2)
            - gammaln(df / 2)
            - 0.5 * dim * jnp.log(df * jnp.pi)
        )

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        norm2 = jnp.sum(x**2, axis=-1)
        log_kernel = -0.5 * (self.df + self.dim) * jnp.log1p(norm2 / self.df)
        return self._log_norm_const + log_kernel

    def _sample_n(self, seed: jax.random.PRNGKey, sample_shape=()) -> jnp.ndarray:
        """
        Samples from the standard multivariate t distribution:
        X = Z / sqrt(G / df), where
        - Z ~ N(0, I)
        - G ~ Gamma(df / 2, 1/2)
        """
        key1, key2 = jax.random.split(seed)
        shape = (sample_shape, ) + (self.dim,)

        z = jax.random.normal(key1, shape)  # Z ~ N(0, I)
        g = jax.random.gamma(key2, self.df / 2, shape=sample_shape) * 2  # chi2 ~ Gamma(df/2, 1/2), so chi2 ~ 2 * gamma
        return z / jnp.sqrt(g[..., None] / self.df)

    @property
    def event_shape(self):
        return (self.dim,)

    @property
    def batch_shape(self):
        return ()
    

class Banana_t:
    def __init__(self, df: float, dim: int):
        assert dim >= 2, "Dimension must be at least 2"
        self.df = df
        self.dim = dim
        self.base_dist = MultivariateStudentT(df, dim)

    def sample(self, seed, n=1):
        samples = self.base_dist.sample(seed=seed, sample_shape=(n, ))
        samples = samples.at[..., 1].set(samples[..., 0] ** 2 * .5 + samples[..., 1])
        return samples

    def log_prob(self, x):
        z = x.copy()
        z = z.at[..., 1].set(z[..., 1] - z[..., 0] ** 2 * .5)
        return self.base_dist.log_prob(z)


class skewt:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.d = len(a)
        self._skewt_dist = jf_skew_t(a, b)
    
    def log_prob(self, x):
        logp = (self.a + 0.5) * jnp.log(1 + x / jnp.sqrt(self.a + self.b + x**2)) + (self.b + .5) * jnp.log(1 - x / jnp.sqrt(self.a + self.b + x**2))
        return logp.sum(-1)

    def sample(self, seed, n=1):
        return self._skewt_dist.rvs((n, self.d), random_state=seed)


class skewCauchy:
    def __init__(self, a):
        self.a = a
        self.d = len(a)
        self._cauchy_dist = skewcauchy(a)
    
    def log_prob(self, x):
        logp = -jnp.log(1 + x ** 2 / (self.a * jnp.sign(x) + 1)**2)
        return logp.sum(-1)
    
    def sample(self, seed, n=1):
        return self._cauchy_dist.rvs((n, self.d), random_state=seed)


class CauchyDifference:
    """
    Definition 2.1 of https://arxiv.org/pdf/2105.12488 
    """
    def __init__(self, d, gamma=1., lbd=1.):
        self.d = d
        self.gamma = gamma
        self.lbd = lbd
    
    def log_prob(self, x):
        z = jnp.diff(x, axis=-1)
        logp1 = -jnp.log(self.gamma + z[..., 0]**2)
        logp2 = jnp.sum(-jnp.log(self.lbd + z[..., 1:]**2), axis=-1)
        return logp1 + logp2
    
    def sample(self, seed, n=1):
        z = cauchy.rvs(size=(n, self.d), random_state=seed)
        z[:, 0] /= self.gamma
        z[:, 1:] /= self.lbd
        return jnp.cumsum(z, axis=-1)
