import jax
import haiku as hk

class DAE(hk.Module):
    def __init__(self):
        super(DAE, self).__init__()

    def __call__(self, x, is_training=False):
        # Encoder
        net = hk.Linear(128)(x)
        net = hk.BatchNorm(True, True, 0.9)(net, is_training)
        net = jax.nn.leaky_relu(net)
        net = hk.Linear(128)(net)
        net = hk.BatchNorm(True, True, 0.9)(net, is_training)
        net = jax.nn.leaky_relu(net)
        net = hk.Linear(2)(net)

        # Decoder
        net = hk.Linear(128)(net)
        net = hk.BatchNorm(True, True, 0.9)(net, is_training)
        net = jax.nn.leaky_relu(net)
        net = hk.Linear(128)(net)
        net = hk.BatchNorm(True, True, 0.9)(net, is_training)
        net = jax.nn.leaky_relu(net)
        net = hk.Linear(2)(net)

        return net
