import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, betainc
import distrax
from typing import Union
from scipy.stats import jf_skew_t, skewcauchy, cauchy

class StudentT(distrax.Distribution):
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
    def __init__(self, d, df):
        """
        Standard multivariate Student's t-distribution with zero mean and identity covariance.

        Args:
            d: Dimension (d >= 1)
            df: Degrees of freedom (> 0), scalar
        """
        self.df = df
        self.d = d

        self._log_norm_const = (
            gammaln((df + d) / 2)
            - gammaln(df / 2)
            - 0.5 * d * jnp.log(df * jnp.pi)
        )

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        norm2 = jnp.sum(x**2, axis=-1)
        log_kernel = -0.5 * (self.df + self.d) * jnp.log1p(norm2 / self.df)
        return self._log_norm_const + log_kernel

    def _sample_n(self, seed: jax.random.PRNGKey, sample_shape=()) -> jnp.ndarray:
        key1, key2 = jax.random.split(seed)
        shape = (sample_shape, ) + (self.d,)

        z = jax.random.normal(key1, shape)  # Z ~ N(0, I)
        g = jax.random.gamma(key2, self.df / 2, shape=sample_shape) * 2  # chi2 ~ Gamma(df/2, 1/2), so chi2 ~ 2 * gamma
        return z / jnp.sqrt(g[..., None] / self.df)

    @property
    def event_shape(self):
        return (self.d,)

    @property
    def batch_shape(self):
        return ()
    
class Banana_t:
    def __init__(self, d, df):
        self.df = df
        self.d = d
        self.base_dist = MultivariateStudentT(d, df)

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

class funnel_t:
    def __init__(self, d, df):
        self.df = df
        self.d = d
        self.base_dist_x0 = StudentT(df, loc=0., scale=1.)

    def sample(self, seed, n=1):
        key1, key2 = jax.random.split(jax.random.key(seed))
        x0 = self.base_dist_x0.sample(seed=key1, sample_shape=(n, 1))
        x = jax.random.normal(key=key2, shape=(n, self.d-1)) * jnp.exp(x0 / 2)
        samples = jnp.concatenate([x0, x], axis=-1)
        return samples

    def log_prob(self, x):
        logp_0 = self.base_dist_x0.log_prob(x[..., 0])
        scale = jnp.exp(x[..., 0] / 2)
        logp_rest = jnp.sum(-0.5 * (x[..., 1:] / scale)** 2 - jnp.log(scale), axis=-1)
        return logp_0 + logp_rest


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

class RobitRegression:
    """
    Posterior (unnormalized) for robit regression:
        y_i ~ Bernoulli( F_t( (x_i^T beta)/scale, df=df_link ) ),
      with student-t prior on beta
    """
    def __init__(self, X, y, link_df=2., link_scale=1.0, prior_df=2., prior_scale=1.):
        self.X = jnp.asarray(X)
        self.y = jnp.asarray(y)
        self.n, self.d = self.X.shape
        self.link_df = float(link_df)
        self.link_scale = float(link_scale)
        self.prior_df = float(prior_df)
        self.prior_scale = float(prior_scale)

    def log_prob(self, beta):
        beta = jnp.asarray(beta)
        # prior
        log_prior = (-self.prior_df - 1) / 2 * jnp.sum(jnp.log1p((beta / self.prior_scale) ** 2 / self.prior_df))
        
        # likelihood
        eta = self.X @ beta
        z = eta / self.link_scale
        p = student_t_cdf(z, self.link_df)
        p = jnp.clip(p, 1e-8, 1.0 - 1e-8)

        y = jnp.broadcast_to(self.y, p.shape)
        ll = jnp.sum(y * jnp.log(p) + (1.0 - y) * jnp.log1p(-p), axis=-1)

        return log_prior + ll

class LogisticRegression:
    def __init__(self, X, y, prior_df=2., prior_scale=1.):
        self.X = jnp.asarray(X)
        self.y = jnp.asarray(y)
        self.n, self.d = self.X.shape
        self.prior_df = float(prior_df)
        self.prior_scale = float(prior_scale)

    def log_prob(self, beta):
        beta = jnp.asarray(beta)
        # prior
        log_prior = (-self.prior_df - 1) / 2 * jnp.sum(
            jnp.log1p((beta / self.prior_scale) ** 2 / self.prior_df),
            axis=-1,
        )
        
        # likelihood
        eta = jnp.einsum("np,...p->...n", self.X, beta)
        p = jax.nn.sigmoid(eta)
        p = jnp.clip(p, 1e-6, 1.0 - 1e-6)

        y = jnp.broadcast_to(self.y, p.shape)
        ll = jnp.sum(y * jnp.log(p) + (1.0 - y) * jnp.log1p(-p), axis=-1)

        return log_prior + ll


class LogisticRegressionHorseshoe:
    def __init__(self, X, y):
        self.X = jnp.asarray(X)
        self.y = jnp.asarray(y)
        self.n, self.p = self.X.shape
        self.d = 2 * self.p + 1

    def unpack(self, state):
        state = jnp.asarray(state)
        beta = state[..., :self.p]
        log_lambda_local = state[..., self.p:2 * self.p]
        log_tau_global = state[..., -1]
        return beta, log_lambda_local, log_tau_global

    def extract_beta(self, samples):
        return jnp.asarray(samples)[..., :self.p]

    def initial_state(self):
        return jnp.zeros(self.d)

    def log_prob(self, state):
        beta, log_lambda_local, log_tau_global = self.unpack(state)
        lambda_local = jnp.exp(log_lambda_local)
        tau_global = jnp.exp(log_tau_global)

        eta = jnp.einsum("np,...p->...n", self.X, beta)
        y = jnp.broadcast_to(self.y, eta.shape)
        log_likelihood = jnp.sum(y * eta - jnp.logaddexp(0.0, eta), axis=-1)

        scale = tau_global[..., None] * lambda_local
        beta_prior = -0.5 * jnp.sum((beta / scale) ** 2, axis=-1) - jnp.sum(log_lambda_local, axis=-1) - self.p * log_tau_global
        lambda_prior = jnp.sum(log_lambda_local - jnp.log1p(lambda_local**2), axis=-1)
        tau_prior = log_tau_global - jnp.log1p(tau_global**2)
        return log_likelihood + beta_prior + lambda_prior + tau_prior


def ar1_correlation_matrix(n, rho):
    idx = jnp.arange(n)
    return rho ** jnp.abs(idx[:, None] - idx[None, :])


class MultivariateProbitPosterior:
    def __init__(self, X, y, rho=0.7, cov=None, prior_df=2.0, prior_scale=1.0):
        """
        Latent Gaussian posterior for multivariate binary outcomes:
            Y_ij = 1{Z_ij > 0}
            Z_i ~ N(mu_i, Sigma),  mu_ij = X_ij^T beta.

        X has shape (n, J, p), y has shape (n, J), and Sigma is a fixed
        J x J correlation matrix. If cov is not supplied, an AR(1)
        correlation matrix with parameter rho is used.
        """
        self.X = jnp.asarray(X)
        self.y = jnp.asarray(y)
        if self.X.ndim != 3:
            raise ValueError("X must have shape (n, J, p)")
        if self.y.shape != self.X.shape[:2]:
            raise ValueError("y must have shape (n, J), matching X.shape[:2]")

        self.n, self.J, self.p = self.X.shape
        self.d = self.p + self.n * self.J
        self.side = 2.0 * self.y - 1.0
        self.rho = float(rho)
        self.prior_df = float(prior_df)
        self.prior_scale = float(prior_scale)

        self.cov = ar1_correlation_matrix(self.J, self.rho) if cov is None else jnp.asarray(cov)
        if self.cov.shape != (self.J, self.J):
            raise ValueError("cov must have shape (J, J)")
        self.cov_tril = jnp.linalg.cholesky(self.cov)
        self.cov_tril_inv = jnp.linalg.inv(self.cov_tril)
        self.log_det = jnp.sum(jnp.log(jnp.diag(self.cov_tril)))

    def unpack(self, state):
        state = jnp.asarray(state)
        beta = state[..., :self.p]
        u = state[..., self.p:].reshape(state.shape[:-1] + (self.n, self.J))
        return beta, u

    def latent_from_unconstrained(self, u):
        return self.side * jnp.exp(u)

    def latent_from_state(self, state):
        _, u = self.unpack(state)
        return self.latent_from_unconstrained(u)

    def extract_beta(self, samples):
        return jnp.asarray(samples)[..., :self.p]

    def initial_state(self):
        beta = jnp.zeros(self.p)
        z = 0.5 * self.side
        u = jnp.log(self.side * z)
        return jnp.concatenate([beta, u.reshape(-1)])

    def log_prob(self, state):
        beta, u = self.unpack(state)
        z = self.latent_from_unconstrained(u)
        mean = jnp.einsum("ijp,...p->...ij", self.X, beta)
        resid = z - mean
        whitened = jnp.einsum("jk,...ik->...ij", self.cov_tril_inv, resid)
        log_latent = -0.5 * jnp.sum(whitened**2, axis=(-2, -1)) - self.n * self.log_det
        log_prior = (-self.prior_df - 1.0) / 2.0 * jnp.sum(
            jnp.log1p((beta / self.prior_scale) ** 2 / self.prior_df),
            axis=-1,
        )
        log_jacobian = jnp.sum(u, axis=(-2, -1))
        return log_prior + log_latent + log_jacobian


class HorseshoeRegressionPosterior:
    def __init__(self, X, y, sigma=1.0):
        self.X = jnp.asarray(X)
        self.y = jnp.asarray(y)
        self.n, self.p = self.X.shape
        self.d = 2 * self.p + 1
        self.sigma = float(sigma)

    def unpack(self, state):
        state = jnp.asarray(state)
        beta = state[..., :self.p]
        log_lambda_local = state[..., self.p:2 * self.p]
        log_tau_global = state[..., -1]
        return beta, log_lambda_local, log_tau_global

    def extract_beta(self, samples):
        return jnp.asarray(samples)[..., :self.p]

    def initial_state(self):
        return jnp.zeros(self.d)

    def log_prob(self, state):
        beta, log_lambda_local, log_tau_global = self.unpack(state)
        lambda_local = jnp.exp(log_lambda_local)
        tau_global = jnp.exp(log_tau_global)

        resid = (self.y - self.X @ beta) / self.sigma
        log_likelihood = -0.5 * jnp.sum(resid**2) - self.n * jnp.log(self.sigma)

        scale = tau_global * lambda_local
        beta_prior = -0.5 * jnp.sum((beta / scale) ** 2) - jnp.sum(log_lambda_local) - self.p * log_tau_global
        lambda_prior = jnp.sum(log_lambda_local - jnp.log1p(lambda_local**2))
        tau_prior = log_tau_global - jnp.log1p(tau_global**2)
        return log_likelihood + beta_prior + lambda_prior + tau_prior
