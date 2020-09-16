import os
from pathlib import Path

import click
import pickle
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from nsec.mri.model import get_model
os.environ['SINGLECOIL_TRAIN_DIR'] = 'singlecoil_train/singlecoil_train/'
from tf_fastmri_data.datasets.noisy import ComplexNoisyFastMRIDatasetBuilder

def evaluate_denoiser_score_matching(batch_size=32, noise_power_spec=30):
    val_mri_ds = ComplexNoisyFastMRIDatasetBuilder(
        dataset='val',
        brain=False,
        scale_factor=1e6,
        noise_power_spec=noise_power_spec,
        noise_input=True,
        noise_mode='gaussian',
        residual_learning=True,
        batch_size=batch_size,
        kspace_size=(320, 320),
        slice_random=True,
    )
    mri_images_iterator = val_mri_ds.preprocessed_ds.take(1).as_numpy_iterator()

    _, loss_fn, _, _, _, _, _, rng_seq = get_model(opt=False)
    with open(str(Path(os.environ['CHECKPOINTS_DIR']) / f'conv-dae-L2-mri-{noise_power_spec}.pckl'), 'rb') as file:
        params, state, _ = pickle.load(file)


    batch = next(mri_images_iterator)
    print(loss_fn(params, state, next(rng_seq), batch)[0])
