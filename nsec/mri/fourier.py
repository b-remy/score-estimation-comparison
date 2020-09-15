"""Fourier utilities"""
import jax.numpy as jnp


class FFT2:
    """This class defines the masked fourier transform operator in 2D, where
    the mask is defined on shifted fourier coefficients.
    """
    def __init__(self, mask):
        self.mask = mask
        self.shape = mask.shape
        self.scaling_norm = np.sqrt(np.prod(mask.shape[-2:]))

    def op(self, img):
        """ This method calculates the masked Fourier transform of a 2-D image.

        Parameters
        ----------
        img: jnp.ndarray
            input 2D array with the same shape as the mask.

        Returns
        -------
        x: jnp.ndarray
            masked Fourier transform of the input image.
        """

        fft_coeffs = jnp.fft.ifftshift(jnp.fft.fft2(jnp.fft.fftshift(img, axes=(-2, -1))), axes=(-2, -1))
        fft_coeffs = fft_coeffs / self.scaling_norm
        return self.mask * fft_coeffs

    def adj_op(self, x):
        """ This method calculates inverse masked Fourier transform of a 2-D
        image.

        Parameters
        ----------
        x: jnp.ndarray
            masked Fourier transform data.

        Returns
        -------
        img: jnp.ndarray
            inverse 2D discrete Fourier transform of the input coefficients.
        """
        masked_fft_coeffs = self.mask * x
        masked_fft_coeffs = masked_fft_coeffs * self.scaling_norm
        return jnp.fft.fftshift(jnp.fft.ifft2(jnp.fft.ifftshift(masked_fft_coeffs, axes=(-2, -1))), axes=(-2, -1))


def fft(image):
    """Perform the fft of an image"""
    fourier_op = FFT2(jnp.ones_like(image))
    kspace = fourier_op.op(image)
    return kspace

def ifft(kspace):
    """Perform the ifft of an image"""
    fourier_op = FFT2(jnp.ones_like(kspace))
    image = fourier_op.adj_op(kspace)
    return image
