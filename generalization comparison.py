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
from nsec.models.nflow.nsf import NeuralSplineCoupling, NeuralSplineFlow
from functools import partial
import matplotlib.pyplot as plt
import pickle
import os, os.path

# Tow moons dataset
distribution, dist_label = get_two_moons(0.05), 'two_moons'
# Mixture 2 gaussian dataset
#distribution, dist_label = get_gm(0.5), 'two_gaussians'
# Swiss roll
#distribution, dist_label = get_swiss_roll(0.5), 'swiss_roll'

delta = 0.05

# Computing the true score of data distribution
true_score = jax.vmap(jax.grad(distribution.log_prob))

"""Creates AR-DAE model
"""
def forward(x, sigma, is_training=False):
    denoiser = ARDAE(is_training=is_training)
    return denoiser(x, sigma)
model_train = hk.transform_with_state(partial(forward, is_training=True))

batch_size = 512
delta = delta
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

print("Let's learn a denoising auto encoder")
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
#lipschitz_constants = [0.01, 0.1, 0.5, 1, 2, 5, 10]
l = 2

def forward(x, sigma, is_training=False):
    denoiser = ARDAE(is_training=is_training)
    return denoiser(x, sigma)

lipschitz_constant = l

model_train = hk.transform_with_state(partial(forward, is_training=True))
sn_fn = hk.transform_with_state(lambda x: CustomSNParamsTree(ignore_regex='[^?!.]*b$', val=lipschitz_constant)(x))

batch_size = 512
delta = delta
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

print("Let's learn a denoising auto encoder with spectral normalization (cte = {})".format(lipschitz_constant))
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

print("Let's build a Neural Spline Flow")
"""Build a Normalizing follows
"""

def forwardNF(x):
    flow = NeuralSplineFlow()
    return flow(x)

optimizer = optix.adam(1e-4)
rng_seq = hk.PRNGSequence(42)

batch_size = 512

def make_samples(rng_seq, n_samples, gm):
    return gm.sample(n_samples, seed = next(rng_seq))

def get_batch():
    x = make_samples(rng_seq, batch_size,distribution)
    return {'x': x}
model_NF = hk.transform(forwardNF, apply_rng=True)


@jax.jit
def loss_fn(params, rng_key, batch):
    log_prob = model_NF.apply(params, rng_key, batch['x'])
    return -jnp.mean(log_prob)

@jax.jit
def update(params, rng_key, opt_state, batch):
    loss, grads = jax.value_and_grad(loss_fn)(params, rng_key, batch)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optix.apply_updates(params, updates)
    return loss, new_params, new_opt_state


params = model_NF.init(next(rng_seq), jnp.zeros((1, 2)))
opt_state = optimizer.init(params)

losses = []
for step in range(2000):
    batch = get_batch()
    loss, params, opt_state = update(params, next(rng_seq), opt_state, batch)
    losses.append(loss)
    if step%100==0:
        print(loss)

log_prob = partial(model_NF.apply, params, next(rng_seq))
log_prob(jnp.zeros(2).reshape(1,2)).shape

def log_prob_reshaped(x):
    x = x.reshape([1,-1])
    return jnp.reshape(log_prob(x), ())

score_NF = jax.vmap(jax.grad(log_prob_reshaped))

scale = 3
offset = jnp.array([0., 0.])

d_offset = jnp.array([.5, .25])
c1 = scale * (jnp.array([-.7, -0.5])) + d_offset + offset
c2 = scale * (jnp.array([.7, 0.5])) + d_offset + offset

#X = np.arange(c1[0], c2[0], 0.1)
#Y = np.arange(c1[1], c2[1], 0.1)
n = 100
X = jnp.linspace(c1[0], c2[0], int(n*7/5))
Y = jnp.linspace(c1[1], c2[1], n)

points = jnp.stack(jnp.meshgrid(X, Y), axis=-1).reshape((-1, 2))

estimated_sn, state = score_sn(points, 0.0*jnp.ones((len(points),1)))
estimated_sn = estimated_sn.reshape([len(Y), len(X),2])/jnp.linalg.norm(estimated_sn)

estimated_dae, state = dae_score(points, 0.0*jnp.ones((len(points),1)))
estimated_dae = estimated_dae.reshape([len(Y), len(X),2])/jnp.linalg.norm(estimated_dae)

estimated_nf = score_NF(points)
estimated_nf = estimated_nf.reshape([len(Y), len(X),2])/jnp.linalg.norm(estimated_nf)

true_score = jax.vmap(jax.grad(distribution.log_prob))
true_s = true_score(points)
true_s = true_s.reshape([len(Y), len(X),2])/jnp.linalg.norm(true_s)

errors_sn = jnp.linalg.norm(estimated_sn - true_s, axis=2)
errors_dae = jnp.linalg.norm(estimated_dae - true_s, axis=2)
errors_nf = jnp.linalg.norm(estimated_nf - true_s, axis=2)

errors = [errors_sn, errors_dae, errors_nf]

v_min = jnp.min([jnp.min(e) for e in errors[:-1]])
v_max = jnp.max([jnp.max(e) for e in errors[:-1]])

plt.figure(dpi=100)

plt.subplot(311)
plt.imshow(errors_dae, origin='lower', vmin=v_min, vmax=v_max)
plt.colorbar()

plt.subplot(312)
plt.imshow(errors_sn, origin='lower', vmin=v_min, vmax=v_max)
plt.colorbar()

plt.subplot(313)
plt.imshow(errors_nf, origin='lower', vmin=v_min, vmax=v_max)
plt.colorbar()

#plt.imshow(errors[i], origin='lower', vmin=v_min, vmax=v_max)
#plt.subplot(122)
#quiver(X[::4], Y[::4], estimated_s[::4,::4,0], estimated_s[::4,::4,1]);

curve_error = []

estimated_sn, state = score_sn(points, 0.0*jnp.ones((len(points),1)))
estimated_sn /= jnp.linalg.norm(estimated_sn)

estimated_dae, state = dae_score(points, 0.0*jnp.ones((len(points),1)))
estimated_dae /= jnp.linalg.norm(estimated_dae)

estimated_nf = score_NF(points)
estimated_nf /= jnp.linalg.norm(estimated_nf)

true_s = true_score(points)
true_s /= onp.linalg.norm(true_s)

error_sn = onp.linalg.norm(estimated_sn - true_s, axis=1)
error_dae = onp.linalg.norm(estimated_dae - true_s, axis=1)
error_nf = onp.linalg.norm(estimated_nf - true_s, axis=1)

distance = distribution.log_prob(points)

argsort_distance = distance.argsort()
distance = distance[argsort_distance]
error_sn = error_sn[argsort_distance]
error_dae = error_dae[argsort_distance]
error_nf = error_nf[argsort_distance]

"""
table_sn = jnp.stack([distance, error_sn])
table_sn = jnp.sort(table_sn, 1)
table_dae = jnp.stack([distance, error_dae])
table_dae = jnp.sort(table_dae, 1)
table_nf = jnp.stack([distance, error_nf])
table_nf = jnp.sort(table_nf, 1)
"""

n_p = distance.shape[0]
r = 100
table_dae_bined = onp.zeros((2, n_p//r))
d_dae = onp.zeros(n_p//r)
table_sn_bined = onp.zeros((2, n_p//r))
d_sn = onp.zeros(n_p//r)
table_nf_bined = onp.zeros((2, n_p//r))
d_nf = onp.zeros(n_p//r)

for i in range(n_p//r):
    a = int(i*r)
    b = int((i+1)*r)
    table_dae_bined[0, i] = onp.mean(distance[a:b])
    table_dae_bined[1, i] = onp.mean(error_dae[a:b])
    d_dae[i] = onp.std(error_dae[a:b])/2
    table_sn_bined[0, i] = onp.mean(distance[a:b])
    table_sn_bined[1, i] = onp.mean(error_sn[a:b])
    d_sn[i] = jnp.std(error_sn[a:b])/2
    table_nf_bined[0, i] = onp.mean(distance[a:b])
    table_nf_bined[1, i] = onp.mean(error_nf[a:b])
    d_nf[i] = onp.std(error_nf[a:b])/2

plt.figure(dpi=100)

plt.plot(table_nf_bined[0,:], table_nf_bined[1,:], alpha=1, label='NSF', color='green')
plt.fill_between(table_nf_bined[0,:], table_nf_bined[1,:] - d_nf, table_nf_bined[1,:] + d_nf,
                 color='green', alpha=0.2)
plt.plot(table_dae_bined[0,:], table_dae_bined[1,:], alpha=1, label='DAE', color='red')
plt.fill_between(table_dae_bined[0,:], table_dae_bined[1,:] - d_dae, table_dae_bined[1,:] + d_dae,
                 color='red', alpha=0.2)
plt.plot(table_sn_bined[0,:], table_sn_bined[1,:], alpha=1, label='DAE w/ SN', color='blue')
plt.fill_between(table_sn_bined[0,:], table_sn_bined[1,:] - d_sn, table_sn_bined[1,:] + d_sn,
                 color='blue', alpha=0.2)

plt.ylabel('average error')
plt.xlabel('$\log p(x)$')
#plt.xscale('symlog')
plt.ylim((-.0005, .04))
#plt.xscale('log')
plt.legend()

plt.show()
