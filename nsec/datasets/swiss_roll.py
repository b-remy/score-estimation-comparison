import jax
import jax.numpy as np
import numpy as onp
import tensorflow_probability as tfp; tfp = tfp.experimental.substrates.jax
tfd = tfp.distributions
from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt

def get_swiss_roll(sigma, resolution=1024):

    n_samples = 2*resolution
    X, _ = make_swiss_roll(n_samples, noise=0)
    coords = np.vstack([X[:, 0], X[:, 2]])

    distribution = tfd.MixtureSameFamily(
    mixture_distribution=tfd.Categorical(probs=np.ones(2*resolution)/resolution/2),
    components_distribution=tfd.MultivariateNormalDiag(loc=coords.T, scale_identity_multiplier=sigma)
    )
    return distribution

"""
key = jax.random.PRNGKey(0)
swiss_roll = get_swiss_roll(.05)
data = swiss_roll.sample(1000, seed=key)
"""
