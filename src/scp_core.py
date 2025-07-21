import jax
import jax.numpy as jnp
from scipy.stats import beta

def uniform_sample_bright_side(d, latitude, key, n=1):
    """
    Sample uniformly from the bright side of a d-dimensional sphere.
    """
    key, subkey = jax.random.split(key)
    u = jax.random.uniform(subkey, shape=(n,))

    u = beta.cdf(latitude / 2, d/2, d/2) * u
    height = beta.ppf(u, d/2, d/2) * 2
    radius = jnp.sqrt(1 - (height - 1)**2)
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (n, d))
    x = x / jnp.linalg.norm(x, axis=-1, keepdims=True) * radius[:, None]
    return jnp.concatenate([x, height[:, None]], axis=-1)

def rwm_bright_side(logp_fn, x0, latitude, key, nsample, stepsize=0.1):
    """
    Random walk on the sphere centered at (0,...,0,1) with last coordinate < latitude.
    logp_fn: target log density function on this sphere.
    x0: initial point on the sphere
    nsample: number of MCMC iterations
    stepsize: step size for the random walk
    """
    center = jnp.eye(x0.shape[0])[-1]
    def random_walk_step(carry, i):
        x, key = carry
        key, subkey = jax.random.split(key)
        x_new = x + jax.random.normal(subkey, shape=x.shape) * stepsize
        x_new = (x_new - center) / jnp.linalg.norm(x_new - center) + center
        log_accept = jnp.where(x_new[-1] < latitude, logp_fn(x_new) - logp_fn(x), -jnp.inf)

        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey)
        accept = (jnp.log(u) < log_accept) * 1.
        x = x * (1 - accept) + x_new * accept

        return (x, key), (x, accept)

    return jax.lax.scan(random_walk_step, (x0, key), jnp.arange(nsample))[1]

class SCP:
    def __init__(self, d, latitude):
        self.d = d
        self.latitude = latitude

    def transform_params(self, params):
        observer = params['observer']
        observer = observer / jnp.sqrt(1 + jnp.sum(observer**2)) * jnp.sqrt(1 - (1 - self.latitude)**2)
        scale = jnp.exp(params['scale'])
        shift = params['shift']
        return observer, shift, scale
    
    def inverse_transform_params(self, observer, shift, scale):
        observer = observer / jnp.sqrt(1 - (1 - self.latitude)**2)
        observer = observer / jnp.sqrt(1 - jnp.sum(observer**2) )
        return {'observer': observer, 
                'shift': shift, 
                'scale': jnp.log(scale)}

    def projection(self, params, x):
        observer, shift, scale = self.transform_params(params)
        x_ = jnp.atleast_2d(x)
        y = (self.latitude * x_[:, :-1] - x_[:, -1:] * observer) / (self.latitude - x_[:, -1:])
        y = y * scale + shift
        return y[0] if x.ndim == 1 else y

    def log_jacobian(self, params, y, x=None):
        observer, shift, scale = self.transform_params(params)
        if y is None:
            y = self.projection(params, x)
        y_hat = (y - shift) / scale

        _a = jnp.sum((y_hat - observer) ** 2) + self.latitude ** 2
        
        _b = 2 * ((jnp.dot(y_hat - observer, observer) - self.latitude * (self.latitude - 1)))
        
        _c = jnp.sum(observer ** 2) + self.latitude ** 2 - 2 * self.latitude

        Delta = _b ** 2 - 4 * _a * _c
        M = (-_b + jnp.sqrt(Delta)) / (2 * _a)
        
        d = self.d
        
        return d * jnp.log(scale / M) - jnp.log(self.latitude) + jnp.log(M * jnp.sum((y_hat - observer) ** 2) + jnp.dot(y_hat - observer, observer) + self.latitude - self.latitude**2 * (1 - M))

    def log_prob(self, params, y):
         return -self.log_jacobian(params, y)

    def sample(self, params, key, n=1):
        X = uniform_sample_bright_side(self.d, self.latitude, key, n=n)
        return self.projection(params, X)
    
    def reverse_kl(self, params, logp_fn, X):
        Y = self.projection(params, X)
        logp = jax.vmap(logp_fn)(Y)
        logdet = jax.vmap(self.log_jacobian, in_axes=(None, 0))(params, Y)
        return -jnp.mean(logdet + logp)
    
    def forward_kl(self, params, Y):
        logq = -jax.vmap(self.log_jacobian, in_axes=(None, 0))(params, Y)
        return -jnp.mean(logq)
    
    def transform_target(self, logp_Rd, params):
        def logp_transformed(x):
            y = self.projection(params, x)
            logdet = self.log_jacobian(params, y)
            return logp_Rd(y) + logdet
        return logp_transformed