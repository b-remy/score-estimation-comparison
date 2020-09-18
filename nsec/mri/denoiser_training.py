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

from nsec.mri.model import get_model
os.environ['SINGLECOIL_TRAIN_DIR'] = 'singlecoil_train/singlecoil_train/'
from tf_fastmri_data.datasets.noisy import ComplexNoisyFastMRIDatasetBuilder, NoisyFastMRIDatasetBuilder


def train_denoiser_score_matching(
        batch_size=32,
        noise_power_spec=30,
        n_steps=int(1e3),
        lr=1e-3,
        contrast=None,
        magnitude_images=False,
        pad_crop=True,
        sn_val=2.,
    ):
    ds_kwargs = dict(
        dataset='train',
        brain=False,
        scale_factor=1e6,
        noise_power_spec=noise_power_spec,
        noise_input=True,
        noise_mode='gaussian',
        residual_learning=True,
        batch_size=batch_size,
        slice_random=True,
        contrast=contrast,
    )
    if magnitude_images:
        ds_class = NoisyFastMRIDatasetBuilder
    else:
        ds_class = ComplexNoisyFastMRIDatasetBuilder
        ds_kwargs.update(
            kspace_size=(320, 320),
        )
    train_mri_ds = ds_class(
        **ds_kwargs
    )
    mri_images_iterator = train_mri_ds.preprocessed_ds.take(n_steps).as_numpy_iterator()
    ##### BATCH DEFINITION
    # (image_noisy, noise_power), noise_realisation
    # here the noise_realisation is the full one, not the epsilon from the standard normal law
    print('Finished building dataset, now initializing jax')
    _, _, update, params, state, sn_state, opt_state, rng_seq = get_model(
        lr=lr,
        pad_crop=pad_crop,
        magnitude_images=magnitude_images,
        sn_val=sn_val,
    )

    losses = []
    print('Finished initializing jax, now onto the optim')
    additional_info = ""
    if contrast is not None:
        additional_info += f"_{contrast}"
    if pad_crop:
        additional_info += "_padcrop"
    if magnitude_images:
        additional_info += "_mag"
    if sn_val != 2.:
        additional_info += f'_sn{sn_val}'
    for step, batch in tqdm(enumerate(mri_images_iterator), total=n_steps, desc='Steps'):
        loss, params, state, sn_state, opt_state = update(params, state, sn_state, next(rng_seq), opt_state, batch)
        losses.append(loss)
        if step%100==0:
            print(step, loss)
        if (step+1)%1000==0:
            with open(str(Path(os.environ['CHECKPOINTS_DIR']) / f'conv-dae-L2-mri-{noise_power_spec}{additional_info}.pckl'), 'wb') as file:
                pickle.dump([params, state, sn_state], file)
    if False:
        plt.figure()
        plt.loglog(losses[10:])
        plt.show()




@click.command()
@click.option('batch_size', '-bs', type=int, default=32)
@click.option('n_steps', '-n', type=int, default=int(1e3))
@click.option('noise_power_spec', '-nps', type=float, default=30)
@click.option('lr', '-lr', type=float, default=1e-3)
@click.option('sn_val', '-sn', type=float, default=2.)
@click.option('contrast', '-c', type=str, default=None)
@click.option('magnitude_images', '-m', is_flag=True)
@click.option('pad_crop', '-pc', is_flag=True)
def train_denoiser_score_matching_click(
        batch_size,
        n_steps,
        noise_power_spec,
        lr,
        contrast,
        magnitude_images,
        pad_crop,
        sn_val,
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
    )


if __name__ == '__main__':
    train_denoiser_score_matching_click()
