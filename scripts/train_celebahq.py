# Script for training a denoiser on Celeb_a_hq
import os
os.environ['XLA_FLAGS']='--xla_gpu_cuda_data_dir=/gpfslocalsys/cuda/10.0'

from absl import app
from absl import flags
import haiku as hk
import jax
from jax.experimental import optix
import jax.numpy as jnp
import numpy as onp
import pickle

# Import tensorflow for dataset creation and manipulation
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import tensorflow_datasets as tfds

from nsec.models.dae.convdae_nostride import SmallUResNet
from nsec.normalization import SNParamsTree as CustomSNParamsTree

flags.DEFINE_string("output_dir", ".", "Folder where to store model.")
flags.DEFINE_integer("batch_size", 64, "Size of the batch to train on.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate for the optimizer.")
flags.DEFINE_integer("training_steps", 5000, "Number of training steps to run.")
flags.DEFINE_float("noise_dist_std", 1., "Standard deviation of the noise distribution.")
flags.DEFINE_float("spectral_norm", 2., "Standard deviation of the noise distribution.")
flags.DEFINE_integer("celeba_resolution", 128, "Resolution of celeb dataset, 128 to 1024.")

FLAGS = flags.FLAGS

dset_names={128: 'celeb_a_hq/128',
            128: 'celeb_a_hq/256',
            512: 'celeb_a_hq/512',
            1024: 'celeb_a_hq/1024'}

def load_dataset(resolution, batch_size, noise_dist_std):
  def pre_process(example):
    """ Pre-processing function preparing data for denoising task
    """
    # Retrieve image and normalize it
    x = tf.cast(example['image'], tf.float32) / 255.
    # Sample random Gaussian noise
    u = tf.random.normal(tf.shape(x))
    # Sample standard deviation of noise corruption
    s = noise_dist_std * tf.random.normal((batch_size, 1, 1, 1))
    # Create noisy image
    y = x + s * u
    return {'x':x, 'y':y, 'u':u,'s':s}
  ds = tfds.load(dset_names[resolution], split='train', shuffle_files=True)
  ds = ds.shuffle(buffer_size=10*batch_size)
  ds = ds.batch(batch_size)
  ds = ds.repeat()
  ds = ds.map(pre_process)
  ds = ds.prefetch(buffer_size=5)
  return iter(tfds.as_numpy(ds))

def forward_fn(x, s, is_training=False):
    denoiser = SmallUResNet(n_output_channels=3)
    return denoiser(x, s, is_training=is_training)

def lr_schedule(step):
  """Linear scaling rule optimized for 90 epochs."""
  steps_per_epoch = 30000 // FLAGS.batch_size

  current_epoch = step / steps_per_epoch  # type: float
  lr = (1.0 * FLAGS.batch_size) / 64
  boundaries = jnp.array((20, 40, 60)) * steps_per_epoch
  values = jnp.array([1., 0.1, 0.01, 0.001]) * lr

  index = jnp.sum(boundaries < step)
  return jnp.take(values, index)

def main(_):
  # Make the network
  model = hk.transform_with_state(forward_fn)
  sn_fn = hk.transform_with_state(lambda x: CustomSNParamsTree(ignore_regex='[^?!.]*b$',
                                                              val=FLAGS.spectral_norm)(x))

  # Initialisation
  optimizer = optix.chain(
      optix.adam(learning_rate=FLAGS.learning_rate),
      optix.scale_by_schedule(lr_schedule)
  )
  rng_seq = hk.PRNGSequence(42)
  params, state = model.init(next(rng_seq),
                             jnp.zeros((1, FLAGS.celeba_resolution, FLAGS.celeba_resolution, 3)),
                             jnp.zeros((1, 1, 1, 1)), is_training=True)
  opt_state = optimizer.init(params)
  _, sn_state = sn_fn.init(next(rng_seq), params)

  # Training loss
  def loss_fn(params, state, rng_key, batch):
    res, state = model.apply(params, state, rng_key, batch['y'], batch['s'], is_training=True)
    loss = jnp.mean((batch['u'] + batch['s'] * res)**2)
    return loss, state

  @jax.jit
  def update(params, state, sn_state, rng_key, opt_state, batch):
    (loss, state), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, state, rng_key, batch)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optix.apply_updates(params, updates)
    new_params, new_sn_state = sn_fn.apply(None, sn_state, None, new_params)
    return loss, new_params, state, new_sn_state, new_opt_state

  # Load dataset
  train = load_dataset(FLAGS.celeba_resolution, FLAGS.batch_size,
                       FLAGS.noise_dist_std)

  losses = []
  for step in range(FLAGS.training_steps):
    loss, params, state, sn_state, opt_state = update(params, state, sn_state,
                                                      next(rng_seq), opt_state,
                                                      next(train))
    losses.append(loss)
    if step%100==0:
        print(step, loss)

    if step%5000 ==0:
      with open(FLAGS.output_dir+'/model-%d.pckl'%step, 'wb') as file:
        pickle.dump([params, state, sn_state], file)


  with open(FLAGS.output_dir+'/model-final.pckl', 'wb') as file:
    pickle.dump([params, state, sn_state], file)

if __name__ == "__main__":
  app.run(main)
