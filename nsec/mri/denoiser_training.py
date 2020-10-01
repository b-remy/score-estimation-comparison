import os
from pathlib import Path

import click
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import tensorflow_probability as tfp; tfp = tfp.experimental.substrates.jax
from tqdm import tqdm

from nsec.datasets.fastmri import mri_noisy_generator
from nsec.mri.model import get_model, get_additional_info, get_model_name


def train_denoiser_score_matching(
        batch_size=32,
        scale_factor=1e6,
        noise_power_spec=30,
        n_steps=int(1e3),
        lr=1e-3,
        contrast=None,
        magnitude_images=False,
        pad_crop=True,
        sn_val=2.,
        stride=True,
        image_size=320,
        no_final_conv=False,
        scales=4,
    ):
    train_mri_gen = mri_noisy_generator(
        split='train',
        scale_factor=scale_factor,
        noise_power_spec=noise_power_spec,
        batch_size=batch_size,
        contrast=contrast,
        magnitude=magnitude_images,
        image_size=image_size,
    )
    ##### BATCH DEFINITION
    # (image_noisy, noise_power), noise_realisation
    # here the noise_realisation is the full one, not the epsilon from the standard normal law
    print('Finished building dataset, now initializing jax')
    _, _, update, params, state, sn_state, opt_state, rng_seq = get_model(
        lr=lr,
        pad_crop=pad_crop,
        magnitude_images=magnitude_images,
        sn_val=sn_val,
        stride=stride,
        no_final_conv=no_final_conv,
        scales=scales,
    )

    losses = []
    print('Finished initializing jax, now onto the optim')
    additional_info = get_additional_info(
        contrast=contrast,
        pad_crop=pad_crop,
        magnitude_images=magnitude_images,
        sn_val=sn_val,
        lr=lr,
        stride=stride,
        image_size=image_size,
        no_final_conv=no_final_conv,
        scales=scales,
    )
    model_name = get_model_name(
        noise_power_spec=noise_power_spec,
        additional_info=additional_info,
    )
    for step, batch in tqdm(enumerate(train_mri_gen), total=n_steps, desc='Steps'):
        loss, params, state, sn_state, opt_state = update(params, state, sn_state, next(rng_seq), opt_state, batch)
        losses.append(loss)
        if step%100==0:
            print(step, loss)
        if (step+1)%1000==0:
            with open(str(Path(os.environ['CHECKPOINTS_DIR']) / model_name), 'wb') as file:
                pickle.dump([params, state, sn_state], file)
        if step > n_steps:
            break
    if False:
        plt.figure()
        plt.loglog(losses[10:])
        plt.show()




@click.command()
@click.option('batch_size', '-bs', type=int, default=32)
@click.option('n_steps', '-n', type=int, default=int(1e3))
@click.option('noise_power_spec', '-nps', type=float, default=30)
@click.option('lr', '-lr', type=float, default=1e-3)
@click.option('scale_factor', '-sf', type=float, default=1e6)
@click.option('sn_val', '-sn', type=float, default=2.)
@click.option('contrast', '-c', type=str, default=None)
@click.option('magnitude_images', '-m', is_flag=True)
@click.option('pad_crop', '-pc', is_flag=True)
@click.option('stride', '-st', is_flag=True)
@click.option('image_size', '-is', type=int, default=320)
@click.option('no_final_conv', '--no-fcon', is_flag=True)
@click.option('scales', '-s', type=int, default=4)
def train_denoiser_score_matching_click(
        batch_size,
        n_steps,
        noise_power_spec,
        lr,
        contrast,
        magnitude_images,
        pad_crop,
        sn_val,
        stride,
        image_size,
        no_final_conv,
        scales,
        scale_factor,
    ):
    train_denoiser_score_matching(
        batch_size=batch_size,
        noise_power_spec=noise_power_spec,
        n_steps=n_steps,
        lr=lr,
        contrast=contrast,
        magnitude_images=magnitude_images,
        pad_crop=pad_crop,
        sn_val=sn_val,
        stride=stride,
        image_size=image_size,
        no_final_conv=no_final_conv,
        scales=scales,
        scale_factor=scale_factor,
    )


if __name__ == '__main__':
    train_denoiser_score_matching_click()
