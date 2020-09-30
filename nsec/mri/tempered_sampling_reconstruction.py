import os
from pathlib import Path

import click
os.environ['XLA_FLAGS']='--xla_gpu_cuda_data_dir=/gpfslocalsys/cuda/10.1.2'
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pickle
# from skimage.metrics import peak_signal_noise_ratio as psnr
import tensorflow_probability as tfp; tfp = tfp.experimental.substrates.jax
import tensorflow as tf

from fastmri_recon.models.subclassed_models.updnet import UPDNet
from nsec.datasets.fastmri import mri_recon_generator
from nsec.mri.fourier import FFT2
from nsec.mri.model import get_model, get_additional_info, get_model_name
from nsec.samplers import ScoreHamiltonianMonteCarlo
from nsec.tempered_sampling import TemperedMC


checkpoints_dir = Path(os.environ['CHECKPOINTS_DIR'])
figures_dir = Path(os.environ['FIGURES_DIR'])
figures_dir.mkdir(exist_ok=True)
chains_dir = figures_dir / 'chains'
chains_dir.mkdir(exist_ok=True)

def reconstruct_image_tempered_sampling(
        initial_sigma=50.,
        batch_size=4,
        contrast=None,
        acceleration_factor=4,
        noise_power_spec_training=30.,
        image_size=320,
        num_results=2500,
        eps=10,
        soft_dc_sigma=0.01,
        n_repetitions=7,
        sn_val=2.,
        no_final_conv=False,
        scales=4,
        gamma=0.995,
        min_steps_per_temp=2,
    ):
    val_mri_gen = mri_recon_generator(
        split='val',
        batch_size=batch_size,
        contrast=contrast,
        acceleration_factor=acceleration_factor,
        scale_factor=1e6,
        image_size=image_size,
    )

    model, _, _, _, _, _, _, rng_seq = get_model(
        opt=False,
        magnitude_images=False,
        pad_crop=False,
        stride=False,
        no_final_conv=no_final_conv,
        scales=scales,
    )

    # Importing saved model
    additional_info = get_additional_info(
        contrast=contrast,
        pad_crop=False,
        magnitude_images=False,
        sn_val=sn_val,
        lr=1e-4,
        stride=False,
        image_size=image_size,
        no_final_conv=no_final_conv,
        scales=scales,

    )
    model_name = get_model_name(
        noise_power_spec=noise_power_spec_training,
        additional_info=additional_info,
    )
    with open(checkpoints_dir / model_name, 'rb') as file:
        params, state, _ = pickle.load(file)

    from functools import partial
    score = partial(model.apply, params, state, next(rng_seq))

    (kspace, mask), image_gt = next(val_mri_gen)

    #### NN recon zone
    ### with, for now, hardcoded arguments
    run_id = 'updnet_singlecoil__af4_compound_mssim_1599653590'
    n_epochs = 200
    model = UPDNet(
        n_primal=5,
        n_dual=1,
        primal_only=True,
        multicoil=False,
        n_layers=3,
        layers_n_channels=[16 * 2**i for i in range(3)],
        non_linearity='relu',
        n_iter=10,
        channel_attention_kwargs=None,
        refine_smaps=False,
        output_shape_spec=False,
    )
    kspace_size = [1, 640, 372]
    inputs = [
        tf.zeros(kspace_size + [1], dtype=tf.complex64),
        tf.zeros(kspace_size, dtype=tf.complex64),
    ]
    model(inputs)
    model.load_weights(checkpoints_dir / '..' / 'checkpoints' / f'{run_id}-{n_epochs:02d}.hdf5')

    recon_nn = model.predict((kspace.astype(np.complex64), mask))
    #### End of NN recon zone

    for ind in range(batch_size):
        fourier_obj = FFT2(mask[ind])
        x_zfilled = fourier_obj.adj_op(kspace[ind, ..., 0])[None, ..., None]
        final_samples = []
        for j in range(n_repetitions):
            ##### ACTUAL SAMPLING ZONE
            def likelihood_fn(x_, sigma):
                """ This is a likelihood function for masked and noisy data
                """
                current_measurements = fourier_obj.op(x_[..., 0] + 1j * x_[..., 1])
                likelihood = (jnp.linalg.norm(kspace[ind, ..., 0] - current_measurements)**2 / (soft_dc_sigma**2+sigma**2)) /2.
                likelihood = jnp.squeeze(likelihood)
                return likelihood
            score_likelihood = jax.vmap(jax.grad(likelihood_fn))
            def score_fn(x, sigma):
                x_float = x.reshape((1, image_size, image_size, 2))
                x_complex = x_float[..., 0] + 1j * x_float[..., 1]
                prior = score(x_complex[..., None], sigma.reshape((-1,1,1,1)), is_training=False)[0]
                prior = jnp.concatenate([prior.real, prior.imag], axis=-1)
                data_consistency = score_likelihood(x_float, sigma.reshape((-1,1,1,1)))
                return ( prior - data_consistency).reshape((-1, image_size*image_size*2))

            z = jax.random.normal(jax.random.PRNGKey(j), shape=x_zfilled.shape)
            z = z + 1j * jax.random.normal(jax.random.PRNGKey(j), shape=x_zfilled.shape)
            z = z*initial_sigma
            init_image = x_zfilled + z
            init_image = jnp.concatenate([init_image.real, init_image.imag], axis=-1)
            init_image = init_image.reshape((-1, 2*image_size**2))

            def make_kernel_fn(target_log_prob_fn, target_score_fn, sigma):
              return ScoreHamiltonianMonteCarlo(
                  target_log_prob_fn=target_log_prob_fn,
                  target_score_fn=target_score_fn,
                  step_size=eps*(sigma/initial_sigma)**1.5,
                  num_leapfrog_steps=3,
                  num_delta_logp_steps=4)

            tmc = TemperedMC(
                        target_score_fn=score_fn,
                        inverse_temperatures=initial_sigma*np.ones([1]),
                        make_kernel_fn=make_kernel_fn,
                        gamma=gamma,
                        min_steps_per_temp=min_steps_per_temp,
                        num_delta_logp_steps=4)

            num_burnin_steps = int(1e1)

            samples, trace = tfp.mcmc.sample_chain(
                    num_results=10,
                    num_steps_between_results=num_results//10,
                    current_state=init_image,
                    kernel=tmc,
                    num_burnin_steps=num_burnin_steps,
                    trace_fn=lambda _, pkr: (pkr.pre_tempering_results.is_accepted,
                                             pkr.post_tempering_inverse_temperatures,
                                             pkr.tempering_log_accept_ratio),
                    seed=jax.random.PRNGKey(j))
            ##### END OF SAMPLING ZONE
            # We need to generate the sampling chain image
            fig, axs = plt.subplots(1, 10, sharex=True, sharey=True, figsize=(10, 2), gridspec_kw={'wspace': 0, 'hspace': 0})
            for i in range(10):
                im = samples[i].reshape((image_size, image_size, 2))
                im_not_normed = im
                im = jnp.linalg.norm(im, axis=-1)
                im = jnp.squeeze(im)
                ax = axs[i]
                ax.imshow(im, vmin=0, vmax=200)
                ax.axis('off')
            plt.tight_layout()
            plt.savefig(chains_dir / f'reconstruction_chain_{ind}_{j}.png')
            final_samples.append(im_not_normed)
        target_image = jnp.squeeze(jnp.abs(image_gt[ind]))
        fig, axs = plt.subplots(2, 5, sharex=True, sharey=True, figsize=(10, 5), gridspec_kw={'wspace': 0, 'hspace': 0})
        axs[0, 0].imshow(target_image, vmin=0, vmax=200)
        axs[0, 0].axis('off')
        axs[0, 1].imshow(jnp.squeeze(jnp.abs(x_zfilled[0])), vmin=0, vmax=200)
        axs[0, 1].axis('off')
        axs[0, 2].imshow(np.squeeze(recon_nn[ind]), vmin=0, vmax=200)
        axs[0, 2].axis('off')
        for i in range(n_repetitions):
            im = final_samples[i]
            im = jnp.linalg.norm(im, axis=-1)
            im = jnp.squeeze(im)
            if i < 2:
                ax = axs[0, i+3]
            else:
                ax = axs[1, i - 2]
            ax.imshow(im, vmin=0, vmax=200)
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(figures_dir / f'reconstruction_samples_{ind}.png')



@click.command()
@click.option('batch_size', '-b', type=int, default=4)
@click.option('contrast', '-c', type=str, default=None)
@click.option('acceleration_factor', '-a', type=int, default=4)
@click.option('noise_power_spec_training', '-nps', type=float, default=30.)
@click.option('image_size', '-is', type=int, default=320)
@click.option('num_results', '-n', type=int, default=50_000)
@click.option('eps', '-e', type=float, default=1e-5)
@click.option('soft_dc_sigma', '-dcs', type=float, default=0.01)
@click.option('initial_sigma', '-si', type=float, default=50.)
@click.option('n_repetitions', '-nr', type=int, default=10)
@click.option('sn_val', '-sn', type=float, default=2.)
@click.option('no_final_conv', '--no-fcon', is_flag=True)
@click.option('scales', '-sc', type=int, default=4)
def reconstruct_image_tempered_sampling_click(
        initial_sigma,
        batch_size,
        contrast,
        acceleration_factor,
        noise_power_spec_training,
        image_size,
        num_results,
        eps,
        soft_dc_sigma,
        n_repetitions,
        sn_val,
        no_final_conv,
        scales,
    ):
    reconstruct_image_tempered_sampling(
        initial_sigma=initial_sigma,
        batch_size=batch_size,
        contrast=contrast,
        acceleration_factor=acceleration_factor,
        noise_power_spec_training=noise_power_spec_training,
        image_size=image_size,
        num_results=num_results,
        eps=eps,
        soft_dc_sigma=soft_dc_sigma,
        n_repetitions=n_repetitions,
        sn_val=sn_val,
        no_final_conv=no_final_conv,
        scales=scales,
    )


if __name__ == '__main__':
    reconstruct_image_tempered_sampling_click()
