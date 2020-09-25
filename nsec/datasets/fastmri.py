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

def ifft(kspace):
    scaling_norm = np.sqrt(np.prod(kspace.shape[-2:]))
    image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace, axes=(-2, -1))), axes=(-2, -1))
    image = image * scaling_norm
    return image

def crop_center(img, cropx=320, cropy=None):
    y, x = img.shape
    if cropy is None:
        cropy = cropx
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]

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

def load_image_complex(file, image_size=320):
    with h5py.File(file, 'r') as h5_obj:
        kspace_h5 = h5_obj['kspace']
        kspace_shape = kspace_h5.shape
        n_slices = kspace_shape[0]
        i_slice = random.randint(0, n_slices - 1)
        kspace = kspace_h5[i_slice]
    image = ifft(kspace)
    image = crop_center(image)
    if image_size != 320:
        image = resize(image, (image_size, image_size))
    return image

def load_contrast(file):
    with h5py.File(file, 'r') as h5_obj:
        contrast = h5_obj.attrs['acquisition']
    return contrast

def draw_gaussian_noise_power(batch_size, noise_power_spec):
    noise_power = np.random.normal(
        size=(batch_size,),
        loc=0.0,
        scale=noise_power_spec,
    )
    return noise_power

def mri_noisy_generator(
        split='train',
        batch_size=32,
        noise_power_spec=30,
        scale_factor=1e6,
        image_size=320,
        contrast=None,
        magnitude=True,
    ):
    i = 0
    if split == 'train':
        data_path = train_path
    elif split == 'val':
        data_path = val_path
    data_files = list(data_path.glob('*.h5'))
    if contrast is not None:
        data_files = [f for f in data_files if load_contrast(f) == contrast]
    n_files = len(data_files)
    n_batches = n_files // batch_size
    while True:
        relative_i = i % n_batches
        next_batch_files = data_files[relative_i*batch_size: (relative_i+1)*batch_size]
        if magnitude:
            load_fun = load_image
        else:
            load_fun = load_image_complex
        batch = np.array(Parallel(n_jobs=batch_size)(
            delayed(partial(load_fun, image_size=image_size))(file)
            for file in next_batch_files
        ))
        batch = batch[..., None]
        batch = scale_factor * batch
        noise_power = draw_gaussian_noise_power(batch_size=batch_size, noise_power_spec=noise_power_spec)
        noise_shape = list(batch.shape)
        if not magnitude:
            noise_shape[-1] = 2
        normal_noise = np.random.normal(
            size=noise_shape,
            loc=0.0,
            scale=1.0,
        )
        noise_power_bdcast = noise_power[:, None, None, None]
        noise = normal_noise * noise_power_bdcast
        if not magnitude:
            noise = noise[..., 0] + 1j * noise[..., 1]
            noise = noise[..., None]
        image_noisy = batch + noise
        model_inputs = (image_noisy, noise_power)
        model_outputs = noise
        i += 1
        yield model_inputs, model_outputs
