# Script for training a denoiser on Celeb_a_hq
from absl import app
from absl import flags
from absl import loggin
import haiku as hk
import jax
from jax.experimental import optix
import jax.numpy as jnp
import numpy as onp
# Import tensorflow for dataset creation and manipulation
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import tensorflow_datasets as tfds

from nsec.models.dae.convdae_nostride import SmallUResNet
from nsec.normalization import SNParamsTree as CustomSNParamsTree

flags.DEFINE_integer("batch_size", 32, "Size of the batch to train on.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate for the optimizer.")
flags.DEFINE_integer("training_steps", 5000, "Number of training steps to run.")
flags.DEFINE_integer("eval_frequency", 100, "How often to evaluate the model.")
flags.DEFINE_float("noise_dist_std", 1.5, "Standard deviation of the noise distribution.")
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
  train_split = dataset.Split.from_string(FLAGS.train_split)

  # See Section 5.1 of https://arxiv.org/pdf/1706.02677.pdf.
  total_batch_size = FLAGS.train_device_batch_size * jax.device_count()
  steps_per_epoch = train_split.num_examples / total_batch_size

  current_epoch = step / steps_per_epoch  # type: float
  lr = (0.1 * total_batch_size) / 256
  lr_linear_till = 5
  boundaries = jnp.array((30, 60, 80)) * steps_per_epoch
  values = jnp.array([1., 0.1, 0.01, 0.001]) * lr

  index = jnp.sum(boundaries < step)
  lr = jnp.take(values, index)
  return lr * jnp.minimum(1., current_epoch / lr_linear_till)

def main(_):
  # Make the network
  model = hk.transform_with_state(forward_fn)
  sn_fn = hk.transform_with_state(lambda x: CustomSNParamsTree(ignore_regex='[^?!.]*b$',
                                                              val=FLAGS.spectral_norm)(x))

  # Initialisation
  optimizer = optix.adam(1e-3)
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

  for step in range(1000000):

    loss, params, state, sn_state, opt_state = update(params, state, sn_state,
                                                      next(rng_seq), opt_state,
                                                      next())
    losses.append(loss)
    if step%100==0:
        print(step, loss)





if __name__ == "__main__":
  app.run(main)
