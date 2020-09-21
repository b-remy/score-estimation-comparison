import os

import haiku as hk
os.environ['XLA_FLAGS']='--xla_gpu_cuda_data_dir=/gpfslocalsys/cuda/10.1.2'
import jax
from jax.experimental import optix
import jax.numpy as jnp

from nsec.models.dae.convdae import SmallUResNet
from nsec.models.dae.convdae_notstride import SmallUResNet as SmallUResNetNoStride
from nsec.normalization import SNParamsTree as CustomSNParamsTree


def get_model(opt=True, lr=1e-3, magnitude_images=False, pad_crop=True, sn_val=2., stride=True):
    def forward(x, s, is_training=False):
        if magnitude_images:
            n_out = 1
        else:
            n_out = 2
        if stride:
            denoiser_cls = SmallUResNet
        else:
            denoiser_cls = SmallUResNetNoStride
        denoiser = denoiser_cls(use_bn=True, n_output_channels=n_out, pad_crop=pad_crop)
        if not magnitude_images:
            x = jnp.concatenate([x.real, x.imag], axis=-1)
        denoised_float = denoiser(x, s, is_training=is_training)
        if not magnitude_images:
            denoised_complex = denoised_float[..., 0] + 1j * denoised_float[..., 1]
            denoised_complex = denoised_complex[..., None]
            out = denoised_complex
        else:
            out = denoised_float
        return out

    model = hk.transform_with_state(forward)

    sn_fn = hk.transform_with_state(lambda x: CustomSNParamsTree(ignore_regex='[^?!.]*b$',val=sn_val)(x))

    rng_seq = hk.PRNGSequence(42)
    if magnitude_images:
        init_dtype = jnp.float32
    else:
        init_dtype = jnp.complex64
    params, state = model.init(next(rng_seq), jnp.zeros((1, 32, 32, 1), dtype=init_dtype), jnp.zeros((1, 1, 1, 1)), is_training=True)
    if opt:
        optimizer = optix.adam(lr)
        opt_state = optimizer.init(params)
    else:
        opt_state = None
    _, sn_state = sn_fn.init(jax.random.PRNGKey(1), params)


    @jax.jit
    def loss_fn(params, state, rng_key, batch):
        (x, s), su = batch
        # this to stick to the original shape of the noise power
        s = s[..., None, None, None]
        res, state = model.apply(params, state, rng_key, x, s, is_training=True)
        if not magnitude_images:
            real_loss = jnp.mean((su.real / s + s * res.real)**2)
            imag_loss = jnp.mean((su.imag / s + s * res.imag)**2)
            loss = real_loss + imag_loss
        else:
            loss = jnp.mean((su / s + s * res)**2)
        return loss, state

    @jax.jit
    def update(params, state, sn_state, rng_key, opt_state, batch):
        (loss, state), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, state, rng_key, batch)
        updates, new_opt_state = optimizer.update(grads, opt_state)

        new_params = optix.apply_updates(params, updates)

        new_params, new_sn_state = sn_fn.apply(None, sn_state, None, new_params)

        return loss, new_params, state, new_sn_state, new_opt_state

    return model, loss_fn, update, params, state, sn_state, opt_state, rng_seq