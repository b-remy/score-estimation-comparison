import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import tensorflow_probability as tfp; tfp = tfp.experimental.substrates.jax
tfd = tfp.distributions

def display_score_two_moons(score, distribution, dpi=100, n=28, is_amortized=True, is_reg=True):
    xc,yc = jnp.meshgrid(jnp.linspace(-1.5,2.5,204),jnp.linspace(-1.,1.5,128))
    Z = jnp.stack([xc.flatten(), yc.flatten()],axis=1).astype('float32')
    S = distribution.log_prob(Z)
    plt.figure(dpi=dpi)
    plt.imshow(jnp.exp(S.reshape((128,204))), cmap='Oranges', origin='lower')

    X = jnp.linspace(-1.5, 2.5, n)
    Y = jnp.linspace(-1., 1.5, n)

    points = jnp.stack(jnp.meshgrid(X, Y), axis=-1).reshape((-1, 2))

    if is_amortized:
        if is_reg:
            res, state = score(points, 0.0*jnp.ones((len(points),1)))
        else:
            res = score(points, 0.0*jnp.ones((len(points),1)))
    else:
        if is_reg:
            res, state = score(points)
        else:
            res = score(points)

    g = res.reshape([n, n, 2])

    #print(g.shape)
    #g = score(points).reshape([len(Y), len(X), 2])
    _x = jnp.linspace(0, 204, n)
    _y = jnp.linspace(0, 128, n)
    plt.quiver(_x, _y, g[:,:,0], g[:,:,1])
    plt.axis('off')

def display_score_gm(score, distribution, dpi=100, n=28):
    xc,yc = jnp.meshgrid(jnp.linspace(-7.,7.,128), jnp.linspace(-7.,7.,128))
    Z = jnp.stack([xc.flatten(), yc.flatten()],axis=1).astype('float32')
    S = distribution.log_prob(Z)

    plt.figure(dpi=dpi)
    plt.imshow(jnp.exp(S.reshape((128,128))), cmap='Oranges', origin='lower')

    X = jnp.linspace(-7., 7., n)
    Y = jnp.linspace(-7., 7., n)
    points = jnp.stack(jnp.meshgrid(X, Y), axis=-1).reshape((-1, 1, 2))

    g = score(points).reshape((n, n, 2))

    #print(g.shape)
    #g = score(points).reshape([len(Y), len(X), 2])
    _x = jnp.linspace(0, 128, n)
    _y = jnp.linspace(0, 128, n)
    plt.quiver(_x, _y, g[:,:,0], g[:,:,1])
    plt.axis('off')
