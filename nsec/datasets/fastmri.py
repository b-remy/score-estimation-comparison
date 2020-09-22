from functools import partial
import os
from pathlib import Path
import random

import h5py
from joblib import Parallel, delayed
import numpy as np
from skimage.transform import resize


fastmri_path = Path(os.environ['FASTMRI_DATA_DIR'])
train_path = fastmri_path / 'singlecoil_train' / 'singlecoil_train'
val_path = fastmri_path / 'singlecoil_val'

def load_image(file, image_size=320):
    with h5py.File(file, 'r') as h5_obj:
        im_h5 = h5_obj['reconstruction_esc']
        im_shape = im_h5.shape
        n_slices = im_shape[0]
        i_slice = random.randint(0, n_slices - 1)
        image = im_h5[i_slice]
    if image_size != 320:
        image = resize(image, (image_size, image_size))
    return image

def draw_gaussian_noise_power(batch_size, noise_power_spec):
    noise_power = np.random.normal(
        size=(batch_size,),
        loc=0.0,
        scale=noise_power_spec,
    )
    return noise_power

def mri_noisy_mag_generator(
        split='train',
        batch_size=32,
        noise_power_spec=30,
        scale_factor=1e6,
        image_size=320,
    ):
    i = 0
    if split == 'train':
        data_path = train_path
    elif split == 'val':
        data_path = val_path
    data_files = list(data_path.glob('*.h5'))
    n_files = len(data_files)
    n_batches = n_files // batch_size
    while True:
        relative_i = i % n_batches
        next_batch_files = data_files[relative_i*batch_size: (relative_i+1)*batch_size]
        batch = np.array(Parallel(n_jobs=batch_size)(
            delayed(partial(load_image, image_size=image_size))(file)
            for file in next_batch_files
        ))
        batch = batch[..., None]
        batch = scale_factor * batch
        noise_power = draw_gaussian_noise_power(batch_size=batch_size, noise_power_spec=noise_power_spec)
        normal_noise = np.random.normal(
            size=batch.shape,
            loc=0.0,
            scale=1.0,
        )
        noise_power_bdcast = noise_power[:, None, None, None]
        noise = normal_noise * noise_power_bdcast
        image_noisy = batch + noise
        model_inputs = (image_noisy, noise_power)
        model_outputs = noise
        i += 1
        yield model_inputs, model_outputs
