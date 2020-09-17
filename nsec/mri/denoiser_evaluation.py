from functools import partial
import os
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except IndexError:
    pass

from nsec.mri.model import get_model
from tf_fastmri_data.datasets.noisy import ComplexNoisyFastMRIDatasetBuilder, NoisyFastMRIDatasetBuilder

def evaluate_denoiser_score_matching(
        batch_size=32,
        noise_power_spec=30,
        n_plots=2,
        contrast=None,
        magnitude_images=False,
        pad_crop=True,
    ):
    print('Building dataset')
    ds_kwargs = dict(
        dataset='val',
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
    val_mri_ds = ds_class(
        **ds_kwargs
    )
    mri_images_iterator = val_mri_ds.preprocessed_ds.take(1).as_numpy_iterator()
    print('Finished building dataset, now onto jax init')
    model, loss_fn, _, _, _, _, _, rng_seq = get_model(
        opt=False,
        pad_crop=pad_crop,
        magnitude_images=magnitude_images,
    )
    with open(str(Path(os.environ['CHECKPOINTS_DIR']) / f'conv-dae-L2-mri-{noise_power_spec}.pckl'), 'rb') as file:
        params, state, _ = pickle.load(file)

    print('Finished jax init, now getting the data')
    batch = next(mri_images_iterator)
    print('Got the data, now evaluating the loss')
    print(loss_fn(params, state, next(rng_seq), batch)[0])

    score = partial(model.apply, params, state, next(rng_seq))
    (x, s), su = batch
    s = s[..., None, None, None]
    res, state = score(x, s, is_training=False)
    for i in range(n_plots):
        ind = i
        fig, axs = plt.subplots(1, 4, sharex=True, sharey=True)
        axs[0].set_title("%0.3f"%s[ind,0,0,0])
        axs[0].imshow(np.abs(x)[ind,...,0],cmap='gray')
        axs[0].axis('off')
        axs[1].imshow(np.abs(x - su)[ind,...,0],cmap='gray')
        axs[1].axis('off')
        axs[2].imshow(np.abs(res)[ind,...,0],cmap='gray')
        axs[2].axis('off')
        axs[2].set_title("%0.3f"%np.std(s[ind,:,:,0]**2 *res[ind,...,0]))
        axs[3].imshow(np.abs(x[ind,...,0] + s[ind,:,:,0]**2 * res[ind,...,0]),cmap='gray')
        axs[3].axis('off')
        plt.show()


@click.command()
@click.option('batch_size', '-bs', type=int, default=32)
@click.option('n_plots', '-n', type=int, default=2)
@click.option('noise_power_spec', '-nps', type=float, default=30)
@click.option('contrast', '-c', type=str, default=None)
@click.option('magnitude_images', '-m', is_flag=True)
@click.option('pad_crop', '-pc', is_flag=True)
def evaluate_denoiser_score_matching_click(
        batch_size,
        n_plots,
        noise_power_spec,
        contrast,
        magnitude_images,
        pad_crop,
    ):
    evaluate_denoiser_score_matching(
        batch_size=batch_size,
        n_plots=n_plots,
        noise_power_spec=noise_power_spec,
        contrast=contrast,
        magnitude_images=magnitude_images,
        pad_crop=pad_crop,
    )


if __name__ == '__main__':
    evaluate_denoiser_score_matching_click()
