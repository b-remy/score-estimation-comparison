import jax
import jax.numpy as np
import tensorflow_probability as tfp; tfp = tfp.experimental.substrates.jax
tfd = tfp.distributions

def get_two_moons(sigma, resolution=1024):
  """
  Returns two moons distribution as a TFP distribution

  Parameters
  ----------
  sigma: float
    Spread of the 2 moons distribution.

  resolution: int
    Number of components in the gaussian mixture approximation of the
    distribution (default: 1024)

  Returns
  -------
  distribution: TFP distribution
    Two moon distribution
  """

  outer_circ_x = np.cos(np.linspace(0, np.pi, resolution))
  outer_circ_y = np.sin(np.linspace(0, np.pi, resolution))
  inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, resolution))
  inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, resolution)) - .5

  X = np.append(outer_circ_x, inner_circ_x)
  Y = np.append(outer_circ_y, inner_circ_y)
  coords = np.vstack([X,Y])

  distribution = tfd.MixtureSameFamily(
    mixture_distribution=tfd.Categorical(probs=np.ones(2*resolution)/resolution/2),
    components_distribution=tfd.MultivariateNormalDiag(loc=coords.T, scale_identity_multiplier=sigma)
    )
  return distribution


 def display_score_two_moons(score, distribution, n=28):
    xc,yc = meshgrid(linspace(-1.5,2.5,204),linspace(-1.,1.5,128))
    Z = stack([xc.flatten(), yc.flatten()],axis=1).astype('float32')
    S = distribution.log_prob(Z)
    figure(dpi=100)
    imshow(exp(S.reshape((128,204))), cmap='Oranges', origin='lower')

    X = linspace(-1.5,2.5,n)
    Y = linspace(-1.,1.5,n)
    points = stack(meshgrid(X, Y), axis=-1).reshape((-1, 2))
    res, state = score(points, 0.0*jnp.ones((len(points),1)))
    g = res.reshape([n, n,2])

    #print(g.shape)
    #g = score(points).reshape([len(Y), len(X), 2])
    _x = linspace(0, 204, n)
    _y = linspace(0, 128, n)
    quiver(_x, _y, g[:,:,0], g[:,:,1])
    axis('off')
