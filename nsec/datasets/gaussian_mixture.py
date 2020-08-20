import jax
import jax.numpy as np
import tensorflow_probability as tfp; tfp = tfp.experimental.substrates.jax
tfd = tfp.distributions

def get_gm(p, noise=1.):
    gm = tfd.MixtureSameFamily(
    mixture_distribution=tfd.Categorical(probs=[p, 1-p]),
    components_distribution=tfd.MultivariateNormalDiag(
        loc=[[-5., -5.],  # component 1
             [5., 5.]],  # component 2
        scale_identity_multiplier=[noise, noise])
    )

    return gm
