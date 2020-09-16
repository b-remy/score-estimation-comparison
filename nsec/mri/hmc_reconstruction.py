from functools import partial
import os
from pathlib import Path

import click
os.environ['XLA_FLAGS']='--xla_gpu_cuda_data_dir=/gpfslocalsys/cuda/10.1.2'
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except IndexError:
    pass
import tensorflow_probability as tfp; tfp = tfp.experimental.substrates.jax

from nsec.mri.fourier import FFT2
from nsec.mri.model import get_model
from nsec.samplers import ScoreHamiltonianMonteCarlo
from tf_fastmri_data.datasets.cartesian import CartesianFastMRIDatasetBuilder

def hmc_mri_reconstruction(noise_power_spec=30, num_results=int(1e4), num_burnin_steps=10):
    model, _, _, _, _, _, _, rng_seq = get_model(opt=False)
    with open(str(Path(os.environ['CHECKPOINTS_DIR']) / f'conv-dae-L2-mri-{noise_power_spec}.pckl'), 'rb') as file:
        params, state, _ = pickle.load(file)
    score = partial(model.apply, params, state, next(rng_seq))
    def score_fn(x, y=None, mask=None, mode='prior'):
        """x is a float tensor, with imag and real concatenated in the channel axis
        """
        if y is None:
            w, h = 320, 320
        else:
            w, h = y.shape
        x_reshaped = x.reshape((1, w, h, 2))
        x_reshaped = x_reshaped[..., 0] + 1j * x_reshaped[..., 1]
        x_reshaped = x_reshaped[..., None]

        prior = score(x_reshaped, jnp.zeros((1,1,1,1))+1e-2, is_training=False)[0]
        if mode == 'prior':
            out = prior
        elif mode == 'data_consistency':
            assert y is not None
            assert mask is not None
            fourier_obj = FFT2(mask)
            data_consistency = fourier_obj.adj_op(fourier_obj.op(x_reshaped[..., 0]) - y)
            data_consistency = data_consistency[..., None]
            ####
            # NOTE
            ####
            # how to deal with scaling in the subsequent addition?
            out = prior + data_consistency
        out_float = jnp.concatenate([out.real, out.imag], axis=-1)
        out_reshaped = out_float.reshape((w*h*2,))
        return out_reshaped

    @partial(jax.jit, static_argnums=(1, 2, 3))
    def get_samples(x, y=None, mask=None, mode='prior'):
        """x is a complex tensor
        """
        # First running SHMC
        kernel_shmc = ScoreHamiltonianMonteCarlo(
                target_score_fn=partial(score_fn, y=y, mask=mask, mode=mode),
                num_leapfrog_steps=4,
                num_delta_logp_steps=4,
                step_size=0.01,
        )
        x_float = jnp.concatenate([x.real, x.imag], axis=-1)
        samples_shmc, is_accepted_shmc = tfp.mcmc.sample_chain(
              num_results=num_results,
              num_burnin_steps=num_burnin_steps,
              current_state= x_float[0].reshape((jnp.prod(x_float.shape),)),
              kernel=kernel_shmc,
              trace_fn=lambda _, pkr: pkr.is_accepted,
              seed=jax.random.PRNGKey(1),
        )
        return samples_shmc, is_accepted_shmc


    val_mri_recon_ds = CartesianFastMRIDatasetBuilder(
        dataset='val',
        af=4,
        brain=False,
        scale_factor=1e6,
        slice_random=True,
        kspace_size=(320, 320),
    )
    val_mri_recon_iterator = val_mri_recon_ds.preprocessed_ds.as_numpy_iterator()
    (kspace, mask), image = next(val_mri_recon_iterator)
    fourier_obj = FFT2(jnp.tile(mask, [1, kspace.shape[1], 1]))
    x = fourier_obj.adj_op(kspace[..., 0])[..., None]

    samples_shmc, is_accepted_shmc = get_samples(x, kspace[0, ..., 0], mask[0], 'data_consistency')

    samples_shmc = samples_shmc[jnp.where(is_accepted_shmc)[0]]
    plt.figure(figsize=(9, 5))
    for i in range(10):
      for j in range(10):
        plt.subplot(10,10,10*i+j+1)
        plt.imshow(jnp.linalg.norm(samples_shmc[(10*i+j)].reshape((320, -1, 2)), axis=-1))
        plt.axis('off')


@click.command()
@click.option('noise_power_spec', '-nps', type=float, default=30)
@click.option('num_results', '-n', type=int, default=int(1e4))
@click.option('num_burnin_steps', '-nb', type=int, default=10)
def hmc_mri_reconstruction_click(noise_power_spec, num_results, num_burnin_steps):
    hmc_mri_reconstruction(
        noise_power_spec=noise_power_spec,
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
    )


if __name__ == '__main__':
    hmc_mri_reconstruction_click()
