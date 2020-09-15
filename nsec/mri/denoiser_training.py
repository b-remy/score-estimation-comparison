import os

import haiku as hk
os.environ['XLA_FLAGS']='--xla_gpu_cuda_data_dir=/gpfslocalsys/cuda/10.1.2'
import jax
from jax.experimental import optix
import jax.numpy as jnp
import matplotlib.pyplot as plt
from nsec.normalization import SNParamsTree as CustomSNParamsTree
import pickle
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import tensorflow_probability as tfp; tfp = tfp.experimental.substrates.jax
import tensorflow_datasets as tfds
from tqdm.notebook import tqdm

os.environ['SINGLECOIL_TRAIN_DIR'] = 'singlecoil_train/singlecoil_train/'
from tf_fastmri_data.datasets.noisy import NoisyFastMRIDatasetBuilder
from nsec.models.dae.convdae import SmallUResNet
from nsec.samplers import ScoreHamiltonianMonteCarlo, ScoreMetropolisAdjustedLangevinAlgorithm


batch_size = 32

train_mri_ds = NoisyFastMRIDatasetBuilder(
    dataset='train',
    brain=False,
    scale_factor=1e6,
    noise_power_spec=30,
    noise_input=True,
    noise_mode='gaussian',
    residual_learning=True,
    batch_size=batch_size,
    slice_random=True,
)
n_steps = int(1e3)
mri_images_iterator = train_mri_ds.preprocessed_ds.take(n_steps).as_numpy_iterator()
##### BATCH DEFINITION
# (image_noisy, noise_power), noise_realisation
# here the noise_realisation is the full one, not the epsilon from the standard normal law

def forward(x, s, is_training=False):
    denoiser = SmallUResNet()
    return denoiser(x, s, is_training=is_training)

model = hk.transform_with_state(forward)

sn_fn = hk.transform_with_state(lambda x: CustomSNParamsTree(ignore_regex='[^?!.]*b$',val=2.)(x))

optimizer = optix.adam(1e-4)
rng_seq = hk.PRNGSequence(42)

params, state = model.init(next(rng_seq), jnp.zeros((1, 32, 32, 1)), jnp.zeros((1, 1, 1, 1)), is_training=True)
opt_state = optimizer.init(params)

_, sn_state = sn_fn.init(jax.random.PRNGKey(1), params)

@jax.jit
def loss_fn(params, state, rng_key, batch):
    (x, s), su = batch
    # this to stick to the original shape of the noise power
    s = s[..., None, None, None]
    res, state = model.apply(params, state, rng_key, x, s, is_training=True)
    loss = jnp.mean((su / s + s * res)**2)
    return loss, state

@jax.jit
def update(params, state, sn_state, rng_key, opt_state, batch):
    (loss, state), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, state, rng_key, batch)
    updates, new_opt_state = optimizer.update(grads, opt_state)

    new_params = optix.apply_updates(params, updates)

    new_params, new_sn_state = sn_fn.apply(None, sn_state, None, new_params)

    return loss, new_params, state, new_sn_state, new_opt_state

losses = []
for step, batch in tqdm(enumerate(mri_images_iterator), total=n_steps):
    loss, params, state, sn_state, opt_state = update(params, state, sn_state, next(rng_seq), opt_state, batch)
    losses.append(loss)
    if step%100==0:
        print(step, loss)

if False:
    plt.figure()
    plt.loglog(losses[10:])
    plt.show()
else:
    print(losses)

with open('conv-dae-L2-mri.pckl', 'wb') as file:
    pickle.dump([params, state, sn_state], file)
