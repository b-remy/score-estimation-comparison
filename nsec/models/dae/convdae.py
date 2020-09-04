import jax
import jax.numpy as jnp
import haiku as hk

class UResNet(hk.Module):
  """ Implementation of a denoising auto-encoder based on a resnet architecture
  """

  BlockGroup = hk.nets.resnets.BlockGroup  # pylint: disable=invalid-name
  BlockV1 = hk.nets.resnets.BlockV1  # pylint: disable=invalid-name
  BlockV2 = hk.nets.resnets.BlockV2  # pylint: disable=invalid-name

  def __init__(self,
               blocks_per_group,
               bn_config,
               resnet_v2,
               bottleneck,
               channels_per_groups,
               use_projection,
               name=None):
    """Constructs a Residual UNet model based on a traditional ResNet.
    Args:
      blocks_per_group: A sequence of length 4 that indicates the number of
        blocks created in each group.
      num_classes: The number of classes to classify the inputs into.
      bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
        passed on to the :class:`~haiku.BatchNorm` layers. By default the
        ``decay_rate`` is ``0.9`` and ``eps`` is ``1e-5``.
      resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults to
        ``False``.
      bottleneck: Whether the block should bottleneck or not. Defaults to
        ``True``.
      channels_per_group: A sequence of length 4 that indicates the number
        of channels used for each block in each group.
      use_projection: A sequence of length 4 that indicates whether each
        residual block should use projection.
      name: Name of the module.
    """
    super().__init__(name=name)
    self.resnet_v2 = resnet_v2

    bn_config = dict(bn_config or {})
    bn_config.setdefault("decay_rate", 0.9)
    bn_config.setdefault("eps", 1e-5)
    bn_config.setdefault("create_scale", True)
    bn_config.setdefault("create_offset", True)

    # Number of blocks in each group for ResNet.
    check_length(4, blocks_per_group, "blocks_per_group")
    check_length(4, channels_per_group, "channels_per_group")

    self.initial_conv = hk.Conv2D(
        output_channels=64,
        kernel_shape=7,
        stride=2,
        with_bias=False,
        padding="SAME",
        name="initial_conv")

    if not self.resnet_v2:
      self.initial_batchnorm = hk.BatchNorm(name="initial_batchnorm",
                                            **bn_config)

    self.block_groups = []
    strides = (1, 2, 2, 2)
    for i in range(4):
      self.block_groups.append(
          BlockGroup(channels=channels_per_group[i],
                     num_blocks=blocks_per_group[i],
                     stride=strides[i],
                     bn_config=bn_config,
                     resnet_v2=resnet_v2,
                     bottleneck=bottleneck,
                     use_projection=use_projection[i],
                     name="block_group_%d" % (i)))

    if self.resnet_v2:
      self.final_batchnorm = hk.BatchNorm(name="final_batchnorm", **bn_config)

    self.final_conv = hk.Conv2D(output_channels=1,
                                kernel_shape=5,
                                stride=2,
                                padding="SAME",
                                name="final_conv")

  def __call__(self, inputs, is_training, test_local_stats=False):
    out = inputs
    out = self.initial_conv(out)

    if not self.resnet_v2:
      out = self.initial_batchnorm(out, is_training, test_local_stats)
      out = jax.nn.relu(out)

    levels = [out]

    out = hk.avg_pool(out,
                      window_shape=(1, 3, 3, 1),
                      strides=(1, 2, 2, 1),
                      padding="SAME")

    # Decreasing resolution
    for block_group in self.block_groups:
      levels.append(out)
      out = block_group(out, is_training, test_local_stats)

    # Increasing resolution
    for i, block_group in enumerate(self.inv_block_groups):
      out = block_group(out, is_training, test_local_stats)
      # Concatenating and upsampling
      out = jnp.concatenate([out, levels[-(i+1)]])
      out = hk.Conv2DTranspose(out)

    if self.resnet_v2:
      out = self.final_batchnorm(out, is_training, test_local_stats)
      out = jax.nn.relu(out)

    return self.final_conv(out)
