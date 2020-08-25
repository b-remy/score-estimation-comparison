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


def display_score_error_gm(true_score, estimated_scores, labels, distribution=None, n=28, is_amortized=True, is_reg=True, scale=1, offset=[0, 0], is_NF=False):
    scale = scale
    offset = jnp.array(offset)
    c1 = scale * jnp.array([-7., -7]) + offset
    c2 = scale * jnp.array([7., 7]) + offset
    n = 100
    X = jnp.linspace(c1[0], c2[0], n)
    Y = jnp.linspace(c1[1], c2[1], n)

    if distribution:
        _x, _y = jnp.meshgrid(jnp.arange(0, len(X), 1), jnp.arange(0, len(Y), 1))
        Z = jnp.stack(jnp.meshgrid(X, Y), axis=-1).reshape((-1, 2))
        S = distribution.log_prob(Z)
        dist = jnp.exp(S.reshape((len(Y), len(X))))

    if is_NF:
        points = jnp.stack(jnp.meshgrid(X, Y), axis=-1).reshape((-1, 1, 2))
    else:
        points = jnp.stack(jnp.meshgrid(X, Y), axis=-1).reshape((-1, 2))

    true_s = true_score(points.reshape((-1, 2)))

    true_s = true_s.reshape([len(Y), len(X),2])/jnp.linalg.norm(true_s)

    n_s = len(labels)
    estimated_vector_fileds = []
    errors = []

    for score in estimated_scores:
        if is_amortized:
            if is_reg:
                estimated_s, state = score(points, 0.0*jnp.ones((len(points),1)))
            else:
                estimated_s = score(points, 0.0*jnp.ones((len(points),1)))
        else:
            if is_reg:
                estimated_s, state = score(points)
            else:
                estimated_s = score(points)

        estimated_s = estimated_s.reshape([len(Y), len(X),2])/jnp.linalg.norm(estimated_s)

        estimated_vector_fileds.append(estimated_s)
        errors.append(jnp.linalg.norm(estimated_s - true_s, axis=2))

    v_min = jnp.min([jnp.min(e) for e in errors])
    v_max = jnp.max([jnp.max(e) for e in errors])

    for i in range(n_s):
        plt.subplot(n_s, 2, 2*i+1)
        plt.imshow(errors[i], origin='lower', vmin=v_min, vmax=v_max)
        plt.axis('off')
        plt.colorbar()
        if distribution:
            plt.contour(_x, _y, dist, levels=[1], colors='white')
        plt.title('Score error ({}), avg={:.2e}'.format(labels[i], jnp.mean(errors[i])), fontsize=9)
        plt.subplot(n_s, 2, 2*i+2)
        g = estimated_vector_fileds[i]
        plt.quiver(X[::4], Y[::4], g[::4,::4,0], g[::4,::4,1])
        plt.axis('off')

    plt.tight_layout()

def display_score_error_two_moons(true_score, estimated_scores, labels, distribution=None, dpi=100, n=28, is_amortized=True, is_reg=True, scale=1, offset=[0, 0], is_NF=False):
    scale = scale
    offset = jnp.array(offset)
    d_offset = jnp.array([.5, .25])
    c1 = scale * (jnp.array([-.7, -0.5])) + d_offset + offset
    c2 = scale * (jnp.array([.7, 0.5])) + d_offset + offset
    #X = jnp.arange(c1[0], c2[0], 0.1)
    #Y = jnp.arange(c1[1], c2[1], 0.1)
    n = 100
    X = jnp.linspace(c1[0], c2[0], int(n*7/5))
    Y = jnp.linspace(c1[1], c2[1], n)

    if distribution:
        _x, _y = jnp.meshgrid(jnp.arange(0, len(X), 1), jnp.arange(0, len(Y), 1))
        Z = jnp.stack(jnp.meshgrid(X, Y), axis=-1).reshape((-1, 2))
        S = distribution.log_prob(Z)
        dist = jnp.exp(S.reshape((len(Y), len(X))))

    if is_NF:
        points = jnp.stack(jnp.meshgrid(X, Y), axis=-1).reshape((-1, 1, 2))
    else:
        points = jnp.stack(jnp.meshgrid(X, Y), axis=-1).reshape((-1, 2))

    true_s = true_score(points.reshape((-1, 2)))
    true_s = true_s.reshape([len(Y), len(X),2])/jnp.linalg.norm(true_s)
    n_s = len(labels)
    estimated_vector_fileds = []
    errors = []

    for k, score in enumerate(estimated_scores):
        if k==0 and is_NF:
            estimated_s = score(points)

        else:
            points = points.reshape((-1, 2))
            if is_amortized:
                if is_reg:
                    estimated_s, state = score(points, 0.0*jnp.ones((len(points),1)))
                else:
                    estimated_s = score(points, 0.0*jnp.ones((len(points),1)))
            else:
                if is_reg:
                    estimated_s, state = score(points)
                else:
                    estimated_s = score(points)
        estimated_s = estimated_s.reshape([len(Y), len(X),2])/jnp.linalg.norm(estimated_s)
        estimated_vector_fileds.append(estimated_s)
        errors.append(jnp.linalg.norm(estimated_s - true_s, axis=2))

    v_min = jnp.min([jnp.min(e) for e in errors[1:]])
    v_max = jnp.max([jnp.max(e) for e in errors[1:]])

    for i in range(n_s):
        plt.subplot(n_s, 2, 2*i+1)
        plt.imshow(errors[i], origin='lower', vmin=v_min, vmax=v_max)
        plt.axis('off')
        plt.colorbar()
        if distribution:
            plt.contour(_x, _y, dist, levels=[1], colors='white')
        plt.title('Score error ({}), avg={:.2e}'.format(labels[i], jnp.mean(errors[i])), fontsize=9)
        plt.subplot(n_s, 2, 2*i+2)

        g = estimated_vector_fileds[i]

        plt.quiver(X[::4], Y[::4], g[::4,::4,0], g[::4,::4,1])
        plt.axis('off')

    plt.tight_layout()
