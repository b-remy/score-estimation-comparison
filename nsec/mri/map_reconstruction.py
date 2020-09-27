import os
from pathlib import Path

import click
os.environ['XLA_FLAGS']='--xla_gpu_cuda_data_dir=/gpfslocalsys/cuda/10.1.2'
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pickle
from skimage.metrics import peak_signal_noise_ratio as psnr
import tensorflow_probability as tfp; tfp = tfp.experimental.substrates.jax

from nsec.datasets.fastmri import mri_recon_generator
from nsec.mri.fourier import FFT2
from nsec.mri.model import get_model, get_additional_info, get_model_name


checkpoints_dir = Path(os.environ['CHECKPOINTS_DIR'])
figures_dir = Path(os.environ['FIGURES_DIR'])
figures_dir.mkdir(exist_ok=True)

def reconstruct_image_map(
        batch_size=4,
        contrast=None,
        acceleration_factor=4,
        noise_power_spec_training=30.,
        image_size=320,
        temp=1e-4,
        n_steps=300_000,
        eps=1e-5,
        hard_data_consistency=True,
        soft_dc_lambda=1.,
        sn_val=2.0,
        no_final_conv=False,
        scales=4,
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

    (kspace, mask), image = next(val_mri_gen)

    for ind in range(batch_size):
        fourier_obj = FFT2(mask[ind])
        fourier_pure = FFT2(jnp.ones_like(mask[ind]))
        x_zfilled = fourier_obj.adj_op(kspace[ind, ..., 0])[None, ..., None]
        # gradient descent for MAP, with step size eps
        intermediate_images = []
        im_save_freq = n_steps//10

        @jax.jit
        def update(x_old):
            x_new = x_old + eps * score(x_old, jnp.zeros((1,1,1,1))+temp, is_training=False)[0]
            if hard_data_consistency:
                kspace_new = fourier_pure.op(x_new[0, ..., 0])
                kspace_new = mask[ind] * kspace[ind, ..., 0] + (1-mask[ind]) * kspace_new
                x_new = fourier_pure.adj_op(kspace_new)[None, ..., None]
            else:
                x_new = x_new - soft_dc_lambda * fourier_obj.adj_op(fourier_obj.op(x_new) - kspace[ind, ..., 0])
            return x_new

        x_old = np.copy(x_zfilled)
        for i_step in range(n_steps):
            x_new = update(x_old)
            if (i_step + 1) % im_save_freq == 0:
                intermediate_images.append(x_new.block_until_ready())
            x_old = x_new


        fig, axs = plt.subplots(2, 6, sharex=True, sharey=True, figsize=(9, 3), gridspec_kw={'wspace': 0, 'hspace': 0})
        target_image = jnp.squeeze(jnp.abs(image[ind]))
        axs[0, 0].imshow(target_image, vmin=0, vmax=150)
        axs[0, 0].axis('off')
        axs[0, 1].imshow(jnp.squeeze(jnp.abs(x_zfilled[0])), vmin=0, vmax=150)
        axs[0, 1].axis('off')
        for i in range(len(intermediate_images)):
            if i < 4:
                ax = axs[0, i+2]
            else:
                ax = axs[1, i - 4]
            ax.imshow(jnp.squeeze(jnp.abs(intermediate_images[i])), vmin=0, vmax=150)
            ax.axis('off')
        plt.tight_layout()
        beginning_psnr = psnr(
            target_image,
            jnp.squeeze(jnp.abs(x_zfilled[0])),
            data_range=jnp.max(target_image) - jnp.min(target_image),
        )
        end_psnr = psnr(
            target_image,
            jnp.squeeze(jnp.abs(intermediate_images[-1])),
            data_range=jnp.max(target_image) - jnp.min(target_image),
        )
        fig.suptitle(f'Beginning PSNR: {beginning_psnr}, End PSNR: {end_psnr}')
        plt.savefig(figures_dir / f'mri_recon_{ind}.png')

@click.command()
@click.option('batch_size', '-b', type=int, default=4)
@click.option('contrast', '-c', type=str, default=None)
@click.option('acceleration_factor', '-a', type=int, default=4)
@click.option('noise_power_spec_training', '-nps', type=float, default=30.)
@click.option('image_size', '-is', type=int, default=320)
@click.option('temp', '-t', type=float, default=1e-4)
@click.option('n_steps', '-n', type=int, default=300_000)
@click.option('eps', '-e', type=float, default=1e-5)
@click.option('hard_data_consistency', '-h', is_flag=True)
@click.option('soft_dc_lambda', '-l', type=float, default=1.)
@click.option('sn_val', '-sn', type=float, default=2.)
@click.option('no_final_conv', '--no-fcon', is_flag=True)
@click.option('scales', '-sc', type=int, default=4)
def reconstruct_image_map_click(
        batch_size,
        contrast,
        acceleration_factor,
        noise_power_spec_training,
        image_size,
        temp,
        n_steps,
        eps,
        hard_data_consistency,
        soft_dc_lambda,
        sn_val,
        no_final_conv,
        scales,
    ):
    reconstruct_image_map(
        batch_size=batch_size,
        contrast=contrast,
        acceleration_factor=acceleration_factor,
        noise_power_spec_training=noise_power_spec_training,
        image_size=image_size,
        temp=temp,
        n_steps=n_steps,
        eps=eps,
        hard_data_consistency=hard_data_consistency,
        soft_dc_lambda=soft_dc_lambda,
        sn_val=sn_val,
        no_final_conv=no_final_conv,
        scales=scales,
    )


if __name__ == '__main__':
    reconstruct_image_map_click()
