import os
from pathlib import Path

import click
os.environ['XLA_FLAGS']='--xla_gpu_cuda_data_dir=/gpfslocalsys/cuda/10.1.2'
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pickle
import tensorflow_probability as tfp; tfp = tfp.experimental.substrates.jax

from nsec.datasets.fastmri import mri_noisy_generator
from nsec.mri.model import get_model, get_additional_info, get_model_name
from nsec.samplers import ScoreHamiltonianMonteCarlo


checkpoints_dir = Path(os.environ['CHECKPOINTS_DIR'])
figures_dir = Path(os.environ['FIGURES_DIR'])

def sample_from_image_hmc(
        batch_size=4,
        contrast=None,
        magnitude_images=False,
        noise_power_spec=3,
        noise_power_spec_training=30.,
        image_size=320,
        temp=1.,
        num_results=10_000,
        num_burnin_steps=10,
        step_size=1e-1,
    ):
    val_mri_gen = mri_noisy_generator(
        split='val',
        scale_factor=1e6,
        noise_power_spec=noise_power_spec,
        batch_size=batch_size,
        contrast=contrast,
        magnitude=magnitude_images,
        image_size=image_size,
    )
    model, _, _, _, _, _, _, rng_seq = get_model(opt=False, magnitude_images=True, pad_crop=False, stride=False)

    # Importing saved model
    additional_info = get_additional_info(
        contrast=contrast,
        pad_crop=False,
        magnitude_images=magnitude_images,
        sn_val=2.,
        lr=1e-4,
        stride=False,
        image_size=image_size,
    )
    model_name = get_model_name(
        noise_power_spec=noise_power_spec_training,
        additional_info=additional_info,
    )
    with open(checkpoints_dir / model_name, 'rb') as file:
        params, state, _ = pickle.load(file)

    from functools import partial
    score = partial(model.apply, params, state, next(rng_seq))

    (x, s), _ = next(val_mri_gen)
    s = s[..., None, None, None]

    def score_fn(x, magnitude=False):
        """x is a float tensor, with imag and real concatenated in the channel axis
        """
        w, h = image_size, image_size
        if magnitude:
            n_channels = 1
        else:
            n_channels = 2
        x_reshaped = x.reshape((1, w, h, n_channels))
        if not magnitude:
            x_reshaped = x_reshaped[..., 0] + 1j * x_reshaped[..., 1]
            x_reshaped = x_reshaped[..., None]

        prior = score(x_reshaped, jnp.zeros((1,1,1,1))+temp, is_training=False)[0]
        out = prior
        if not magnitude:
            out_float = jnp.concatenate([out.real, out.imag], axis=-1)
        else:
            out_float = out
        out_reshaped = out_float.reshape((w*h*n_channels,))
        return out_reshaped

    @partial(jax.jit, static_argnums=(1,))
    def get_samples(x, magnitude=False):
        """x is a complex tensor
        """
        # First running SHMC
        kernel_shmc = ScoreHamiltonianMonteCarlo(
                target_score_fn=partial(score_fn, magnitude=magnitude),
                num_leapfrog_steps=4,
                num_delta_logp_steps=4,
                step_size=step_size,
        )
        if not magnitude:
            x_float = jnp.concatenate([x.real, x.imag], axis=-1)
        else:
            x_float = x
        samples_shmc, is_accepted_shmc = tfp.mcmc.sample_chain(
              num_results=num_results,
              num_burnin_steps=num_burnin_steps,
              current_state=x_float.reshape((jnp.prod(x_float.shape),)),
              kernel=kernel_shmc,
              trace_fn=lambda _, pkr: pkr.is_accepted,
              seed=jax.random.PRNGKey(1),
        )
        return samples_shmc, is_accepted_shmc

    for i_batch in range(batch_size):
        samples_shmc, is_accepted_shmc = get_samples(x[i_batch], magnitude_images)
        samples_shmc = samples_shmc.block_until_ready()
        is_accepted_shmc = is_accepted_shmc.block_until_ready()

        samples_shmc = samples_shmc[jnp.where(is_accepted_shmc)[0]]

        n_accepted = len(jnp.where(is_accepted_shmc)[0])
        print(f'For batch {i_batch}, {n_accepted} samples were accepted')

        _, axs = plt.subplots(
            5, 5,
            sharex=True, sharey=True,
            figsize=(9, 9),
            gridspec_kw={'wspace': 0, 'hspace': 0},
        )
        for i in range(5):
            for j in range(5):
                n_ax = i + 5 * j
                image = jnp.linalg.norm(
                    samples_shmc[(n_accepted//25) * n_ax].reshape((image_size, -1, 1)),
                    axis=-1,
                )
                axs[i, j].imshow(image)
                axs[i, j].axis('off')
        plt.tight_layout()
        plt.savefig(figures_dir / f'mri_sampling_{i_batch}.png')

@click.command()
@click.option('batch_size', '-b', type=int, default=4)
@click.option('contrast', '-c', type=str, default=None)
@click.option('magnitude_images', '-m', is_flag=True)
@click.option('noise_power_spec', '-nps', type=float, default=3.)
@click.option('noise_power_spec_training', '--nps-train', type=float, default=30.)
@click.option('image_size', '-is', type=int, default=320)
@click.option('temp', '-t', type=float, default=1.)
@click.option('num_results', '-ns', type=int, default=10_000)
@click.option('num_burnin_steps', '-nb', type=int, default=10)
@click.option('step_size', '-s', type=float, default=1e-1)
def sample_from_image_hmc_click(
        batch_size,
        contrast,
        magnitude_images,
        noise_power_spec,
        noise_power_spec_training,
        image_size,
        temp,
        num_results,
        num_burnin_steps,
        step_size,
    ):
    sample_from_image_hmc(
        batch_size=batch_size,
        contrast=contrast,
        magnitude_images=magnitude_images,
        noise_power_spec=noise_power_spec,
        noise_power_spec_training=noise_power_spec_training,
        image_size=image_size,
        temp=temp,
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        step_size=step_size,
    )


if __name__ == '__main__':
    sample_from_image_hmc_click()
