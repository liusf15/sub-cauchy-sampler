import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
import distrax
from typing import Union

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
        # x1 = x[..., 0]
        # x2 = x[..., 1]
        # rest = x[..., 2:]
        # logp_x1 = self.base_dist.log_prob(x1)
        # logp_eps = self.base_dist.log_prob(x2 - x1 ** 2 * .5)
        # logp_rest = jnp.sum(self.base_dist.log_prob(rest), axis=-1)
        # return logp_x1 + logp_eps + logp_rest
        z = x.copy()
        z = z.at[..., 1].set(z[..., 1] - z[..., 0] ** 2 * .5)
        return self.base_dist.log_prob(z)
    