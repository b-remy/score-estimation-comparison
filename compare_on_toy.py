import jax
import jax.numpy as jnp
import numpy as onp
import haiku as hk
from jax.experimental import optix
from nsec.datasets.two_moons import get_two_moons
from nsec.datasets.gaussian_mixture import get_gm
from nsec.datasets.swiss_roll import get_swiss_roll
from nsec.utils import display_score_two_moons
from nsec.models.dae.ardae import ARDAE
from nsec.normalization import SNParamsTree as CustomSNParamsTree
from nsec.utils import display_score_error_gm, display_score_error_two_moons
from functools import partial
import matplotlib.pyplot as plt

# Tow moons dataset
#distribution, dist_label = get_two_moons(0.05), 'two_moons'
# Mixture 2 gaussian dataset
#distribution, dist_label = get_gm(0.5), 'two_gaussians'
# Swiss roll
distribution, dist_label = get_swiss_roll(0.05), 'swiss_roll'

# Computing the true score of data distribution
true_score = jax.vmap(jax.grad(distribution.log_prob))

"""Creates AR-DAE model
"""
def forward(x, sigma, is_training=False):
    denoiser = ARDAE(is_training=is_training)
    return denoiser(x, sigma)
model_train = hk.transform_with_state(partial(forward, is_training=True))

batch_size = 512
delta = 0.05
def get_batch(rng_key):
    y = distribution.sample(batch_size, seed=rng_key)
    u = onp.random.randn(batch_size, 2)
    s = delta * onp.random.randn(batch_size, 1)
    x = y + s * u
    # x is a noisy sample, y is a sample from the distribution
    # u is the random normal noise realisation
    return {'x':x, 'y':y, 'u':u, 's':s}

# Optimizer
optimizer = optix.adam(1e-3)
rng_seq = hk.PRNGSequence(42)

@jax.jit
def loss_fn(params, state, rng_key, batch):
    res, state = model_train.apply(params,  state, rng_key,
                                   batch['x'], batch['s'])
    loss = jnp.mean((batch['u'] + batch['s'] * res)**2)
    return loss, state

@jax.jit
def update(params, state, rng_key, opt_state, batch):
    (loss, state), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, state, rng_key, batch)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optix.apply_updates(params, updates)
    return loss, new_params, state, new_opt_state

params, state = model_train.init(next(rng_seq),
                                 jnp.zeros((1, 2)),
                                 jnp.ones((1, 1)))

opt_state = optimizer.init(params)
losses = []

for step in range(2000):
    batch = get_batch(next(rng_seq))
    loss, params, state, opt_state = update(params, state, next(rng_seq), opt_state, batch)
    losses.append(loss)
    if step%100==0:
        print(step, loss)

model = hk.transform_with_state(partial(forward, is_training=False))
dae_score = partial(model.apply, params, state, next(rng_seq))

"""Creates AR-DAE model with Lipschitz normalization
"""
def forward(x, sigma, is_training=False):
    denoiser = ARDAE(is_training=is_training)
    return denoiser(x, sigma)

model_train = hk.transform_with_state(partial(forward, is_training=True))
sn_fn = hk.transform_with_state(lambda x: CustomSNParamsTree(ignore_regex='[^?!.]*b$')(x))

batch_size = 512
delta = 0.05

def get_batch(rng_key):
    y = distribution.sample(batch_size, seed=rng_key)
    u = onp.random.randn(batch_size, 2)
    s = delta * onp.random.randn(batch_size, 1)
    x = y + s * u
    # x is a noisy sample, y is a sample from the distribution
    # u is the random normal noise realisation
    return {'x':x, 'y':y, 'u':u, 's':s}

optimizer = optix.adam(1e-3)
rng_seq = hk.PRNGSequence(42)

@jax.jit
def loss_fn(params, state, rng_key, batch):
    res, state = model_train.apply(params,  state, rng_key,
                                   batch['x'], batch['s'])
    loss = jnp.mean((batch['u'] + batch['s'] * res)**2)
    return loss, state

@jax.jit
def update(params, state, sn_state, rng_key, opt_state, batch):
    (loss, state), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, state, rng_key, batch)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optix.apply_updates(params, updates)
    new_params, new_sn_state = sn_fn.apply(None, sn_state, None, new_params)
    return loss, new_params, state, new_sn_state, new_opt_state

params, state = model_train.init(next(rng_seq),
                                 jnp.zeros((1, 2)),
                                 jnp.ones((1, 1)))

opt_state = optimizer.init(params)
_, sn_state = sn_fn.init(jax.random.PRNGKey(1), params)
losses = []

for step in range(2000):
    batch = get_batch(next(rng_seq))
    loss, params, state, sn_state, opt_state = update(params, state, sn_state, next(rng_seq), opt_state, batch)
    losses.append(loss)
    if step%100==0:
        print(step, loss)

model_sn = hk.transform_with_state(partial(forward, is_training=False))
score_sn = partial(model_sn.apply, params, state, next(rng_seq))

scores = [dae_score, score_sn]
labels = ['DAE', 'SN DAE']

plt.figure(dpi=100)

if dist_label=='two_moons':
    display_score_error_two_moons(true_score, scores, labels, distribution=distribution, is_amortized=True, is_reg=True, scale=4, offset=[0., 0.])
elif dist_label=='two_gaussians':
    display_score_error_gm(true_score, scores, labels, distribution=distribution, is_amortized=True, is_reg=True, scale=1, offset=[0., 0.])
elif dist_label=='swiss_roll':
    display_score_error_gm(true_score, scores, labels, distribution=distribution, is_amortized=True, is_reg=True, scale=4, offset=[0., 0.])

plt.savefig('error_comparison.png')
plt.show()
