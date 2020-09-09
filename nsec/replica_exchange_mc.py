# Copyright 2018 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Replica Exchange Monte Carlo Transition Kernel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import warnings
import numpy as np

from functools import partial

import jax
import jax.numpy as jnp

from tensorflow_probability.python.internal.backend.jax.compat import v2 as tf

from tensorflow_probability.python.internal._jax import assert_util
from tensorflow_probability.python.internal._jax import distribution_util
from tensorflow_probability.python.internal._jax import prefer_static
from tensorflow_probability.python.internal._jax import samplers
from tensorflow_probability.python.internal._jax import tensorshape_util
from tensorflow_probability.python.mcmc._jax import hmc
from tensorflow_probability.python.mcmc._jax import kernel as kernel_base
from tensorflow_probability.python.mcmc._jax import metropolis_hastings
from tensorflow_probability.python.mcmc._jax import random_walk_metropolis
from tensorflow_probability.python.mcmc.internal._jax import util as mcmc_util
from tensorflow_probability.python.util._jax.seed_stream import SeedStream
from tensorflow_probability.python.internal.backend.jax import deprecation  # pylint: disable=g-direct-tensorflow-import


__all__ = [
    'ReplicaExchangeMC',
    'default_swap_proposal_fn',
]


# Cause all warnings to always be triggered.
# Not having this means subsequent calls wont trigger the warning.
warnings.filterwarnings('always',
                        module='tensorflow_probability.*replica_exchange_mc',
                        append=True)  # Don't override user-set filters.


class ReplicaExchangeMCKernelResults(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple(
        # All tensors `x` with shape [num_replica, ...] are "ordered", meaning
        # x[k,...] holds values for replica `k`.
        'ReplicaExchangeMCKernelResults',
        [
            # List-like of [num_replica] + batch_shape Tensor (or list thereof)
            # holding state parts for all replicas, after swaps.
            # This will be state parts, *even* if the chain is working with
            # states.
            'post_swap_replica_states',

            # Kernel results for replicas, before any swaps.
            'pre_swap_replica_results',

            # Kernel results for replicas, after swaps.
            # Some fields are updated, and some removed!
            # The theme is to update whatever is necessary to to obtain correct
            # state swaps, and remove fields that are ambiguous (e.g.
            # proposed_results inside Metropolis KR).
            'post_swap_replica_results',

            # Shape [num_replica, num_replica] + batch_shape boolean Tensor
            # where is_swap_proposed[i, j, ...] == True indicates a swap between
            # replicas `i` and `j` has been proposed.
            # Note is_swap_proposed[i, i, ...] == True indicates no move is
            # proposed for replica `i`.
            # TODO(b/144166689) Consider whether it may be better to make the
            # user compute this post-sampling rather than in kernel results.
            'is_swap_proposed',

            # Similar to is_swap_proposed.
            # is_swap_accepted[i, j, ...] indicates a swap between replicas
            # `i` and `j` was accepted.
            'is_swap_accepted',

            # Shape [num_replica - 1] + batch_shape boolean vectors equal to the
            # first lower sub-diagonal of is_swap_proposed.
            # is_swap_proposed_adjacent[k, ...] == True just when an swap is
            # proposed between replicas `k` and `k+1`.
            # This is sufficient to track swaps in the common (and default)
            # case where swaps are between adjacent replicas only.
            'is_swap_proposed_adjacent',
            'is_swap_accepted_adjacent',

            # The inverse_temperatures used to calculate these results. (Other
            # TransitionKernels which want to intercept the inverse_temperatures
            # should rewrite this field.)  Shape [num_replica].
            'inverse_temperatures',

            # Shape [num_replica] + batch_shape permutation used to propose
            # swaps.
            'swaps',

            # Random seed for this step.
            'seed',
        ])):
  """Internal state and diagnostics for Replica Exchange MC."""
  __slots__ = ()


def default_swap_proposal_fn(prob_swap, name=None):
  """Default swap proposal function, for replica swap MC.

  With probability `prob_swap`, propose combinations of replicas to swap
  When exchanging, create combinations of adjacent replicas in
  [Replica Exchange Monte Carlo](
  https://en.wikipedia.org/wiki/Parallel_tempering).  See also review paper [1].

  ```
  swap_fn = default_swap_proposal_fn(prob_swap=0.5)

  swap_fn(num_replica=3)
  ==> [1, 0, 2]  # 1 swap, 0 <--> 1

  swap_fn(num_replica=3)
  ==> [0, 1, 2]  # 0 swaps

  swap_fn(num_replica=3, batch_shape=[2])
  ==> [[0, 1],
       [2, 0],
       [1, 2]]
  ```

  Args:
    prob_swap: Scalar `Tensor` giving probability that any swaps will
      be generated.
    name: Python `str` name given to ops created by this function.
      Default value: `'adjacent_swaps'`.

  Returns:
    default_swap_proposal_fn_: Python callable which take a number of
      replicas (a Python integer), and integer `Tensor` `batch_shape`, and
      returns `swaps`, a shape `[num_replica] + batch_shape` `Tensor`, where
      axis 0 indexes "one-time swaps", i.e., such that (if `rank(swaps) == 1`,
      `range(num_replicas) == tf.gather(swaps, swaps)`.

  #### References

  [1]: David J. Earl, Michael W. Deem
       Parallel Tempering: Theory, Applications, and New Perspectives
       https://arxiv.org/abs/physics/0508111
  """
  def adjacent_swaps(num_replica, batch_shape=(), seed=None):
    """Make random shuffle using only one time swaps."""
    with tf.name_scope(name or 'adjacent_swaps'):
      parity_seed, proposal_seed = samplers.split_seed(
          seed, salt='random_adjacent_shuffle')
      # u selects parity.  E.g.,
      #  u==True ==>  [0, 2, 1, 4, 3] like swaps
      #  u==False ==> [1, 0, 3, 2, 4] like swaps
      # If there are only 2 replicas, then the "True" swaps are null
      # swaps...which would contradict the user provided `prob_swap`.
      # So special case num_replica==2, forcing u==False in this case.
      u_shape = prefer_static.concat((
          tf.ones(1, dtype=tf.int32), tf.cast(batch_shape, tf.int32)), axis=0)
      u = samplers.uniform(u_shape, seed=parity_seed) < 0.5
      u = tf.where(num_replica > 2, u, False)

      x = mcmc_util.left_justified_expand_dims_to(
          tf.range(num_replica, dtype=tf.int64),
          rank=prefer_static.size(u_shape))
      y = tf.where(tf.equal(x % 2, tf.cast(u, dtype=tf.int64)), x + 1, x - 1)
      y = tf.clip_by_value(y, 0, num_replica - 1)
      # TODO(b/142689785): Consider using tf.cond and returning an empty list
      # then in REMC consider using a tf.cond for short-circuiting.
      return tf.where(
          samplers.uniform(batch_shape, seed=proposal_seed) < prob_swap, y, x)

  return adjacent_swaps


class ReplicaExchangeMC(kernel_base.TransitionKernel):
  """Runs one step of the Replica Exchange Monte Carlo.

  [Replica Exchange Monte Carlo](
  https://en.wikipedia.org/wiki/Parallel_tempering) is a Markov chain
  Monte Carlo (MCMC) algorithm that is also known as Parallel Tempering. This
  algorithm takes multiple samples (from tempered distributions) in parallel,
  then swaps these samples according to the Metropolis-Hastings criterion.
  See also the review paper [1].

  The `K` replicas are parameterized in terms of `inverse_temperature`'s,
  `(beta[0], beta[1], ..., beta[K-1])`.  If the target distribution has
  probability density `p(x)`, the `kth` replica has density `p(x)**beta_k`.

  Typically `beta[0] = 1.0`, and `1.0 > beta[1] > beta[2] > ... > 0.0`.
  Trying geometrically decaying `beta` is good starting point.

  * `beta[0] == 1` ==> First replicas samples from the target density, `p`.
  * `beta[k] < 1`, for `k = 1, ..., K-1` ==> Other replicas sample from
    "flattened" versions of `p` (peak is less high, valley less low).  These
    distributions are somewhat closer to a uniform on the support of `p`.

  By default, samples from adjacent replicas `i`, `i + 1` are used as proposals
  for each other in a Metropolis step.  This allows the lower `beta` samples,
  which explore less dense areas of `p`, to eventually swap state with the
  `beta == 1` chain, allowing it to explore these new regions.

  Samples from replica 0 are returned, and the others are discarded.

  #### Examples

  ##### Sampling from the Standard Normal Distribution.

  ```python
  import numpy as np
  from tensorflow_probability.python.internal.backend import jax as tf
  import tensorflow_probability as tfp; tfp = tfp.experimental.substrates.jax
  tfd = tfp.distributions

  dtype = tf.float32

  target = tfd.Normal(loc=dtype(0), scale=dtype(1))

  # Geometric decay is a good rule of thumb.
  inverse_temperatures = 0.5**tf.range(4, dtype=dtype)

  # If everything was Normal, step_size should be ~ sqrt(temperature).
  step_size = 0.5 / tf.sqrt(inverse_temperatures)

  def make_kernel_fn(target_log_prob_fn, seed):
    return tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        seed=seed, step_size=step_size, num_leapfrog_steps=3)

  remc = tfp.mcmc.ReplicaExchangeMC(
      target_log_prob_fn=target.log_prob,
      inverse_temperatures=inverse_temperatures,
      make_kernel_fn=make_kernel_fn)

  def trace_swaps(unused_state, results):
    return (results.is_swap_proposed_adjacent,
            results.is_swap_accepted_adjacent)

  samples, is_swap_proposed_adjacent, is_swap_accepted_adjacent = (
      tfp.mcmc.sample_chain(
          num_results=1000,
          current_state=1.0,
          kernel=remc,
          num_burnin_steps=500,
          trace_fn=trace_swaps)
  )

  # conditional_swap_prob[k] = P[ExchangeAccepted | ExchangeProposed],
  # for the swap between replicas k and k+1.
  conditional_swap_prob = (
      tf.reduce_sum(tf.cast(is_swap_accepted_adjacent, tf.float32), axis=0)
      /
      tf.reduce_sum(tf.cast(is_swap_proposed_adjacent, tf.float32), axis=0))
  ```

  ##### Sampling from a 2-D Mixture Normal Distribution.

  ```python
  import numpy as np
  from tensorflow_probability.python.internal.backend import jax as tf
  import tensorflow_probability as tfp; tfp = tfp.experimental.substrates.jax
  import matplotlib.pyplot as plt
  tfd = tfp.distributions

  dtype = tf.float32

  target = tfd.MixtureSameFamily(
      mixture_distribution=tfd.Categorical(probs=[0.5, 0.5]),
      components_distribution=tfd.MultivariateNormalDiag(
          loc=[[-1., -1], [1., 1.]],
          scale_identity_multiplier=[0.1, 0.1]))

  inverse_temperatures = 0.5**tf.range(4, dtype=dtype)

  # step_size must broadcast with all batch and event dimensions of target.
  # Here, this means it must broadcast with:
  #  [len(inverse_temperatures)] + target.event_shape
  step_size = 0.5 / tf.reshape(tf.sqrt(inverse_temperatures), shape=(4, 1))

  def make_kernel_fn(target_log_prob_fn):
    return tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        step_size=step_size, num_leapfrog_steps=3)

  remc = tfp.mcmc.ReplicaExchangeMC(
      target_log_prob_fn=target.log_prob,
      inverse_temperatures=inverse_temperatures,
      make_kernel_fn=make_kernel_fn)

  samples = tfp.mcmc.sample_chain(
      num_results=1000,
      # Start near the [1, 1] mode.  Standard HMC would get stuck there.
      current_state=tf.ones(2, dtype=dtype),
      kernel=remc,
      trace_fn=None,
      num_burnin_steps=500)

  plt.figure(figsize=(8, 8))
  plt.xlim(-2, 2)
  plt.ylim(-2, 2)
  plt.plot(samples_[:, 0], samples_[:, 1], '.')
  plt.show()
  ```

  #### References

  [1]: David J. Earl, Michael W. Deem
       Parallel Tempering: Theory, Applications, and New Perspectives
       https://arxiv.org/abs/physics/0508111
  """

  @deprecation.deprecated_args(
      '2020-09-20', 'The `seed` argument is deprecated (but will work until '
      'removed). Pass seed to `tfp.mcmc.sample_chain` instead.', 'seed')
  def __init__(self,
               target_score_fn,
               inverse_temperatures,
               make_kernel_fn,
               num_delta_logp_steps=4,
               swap_proposal_fn=default_swap_proposal_fn(1.),
               seed=None,
               validate_args=False,
               name=None):
    """Instantiates this object.

    Args:
      target_log_prob_fn: Python callable which takes an argument like
        `current_state` (or `*current_state` if it's a list) and returns its
        (possibly unnormalized) log-density under the target distribution.
      inverse_temperatures: `Tensor` of inverse temperatures to temper each
        replica. The leftmost dimension is the `num_replica` and the
        second dimension through the rightmost can provide different temperature
        to different batch members, doing a left-justified broadcast.
      make_kernel_fn: Python callable which takes a `target_log_prob_fn`
        arg and returns a `tfp.mcmc.TransitionKernel` instance. Passing a
        function taking `(target_log_prob_fn, seed)` deprecated but supported
        until 2020-09-20.
      swap_proposal_fn: Python callable which take a number of replicas, and
        returns `swaps`, a shape `[num_replica] + batch_shape` `Tensor`, where
        axis 0 indexes a permutation of `{0,..., num_replica-1}`, designating
        replicas to swap.
      seed: Python integer to seed the random number generator. Deprecated, pass
        seed to `tfp.mcmc.sample_chain`. Default value: `None` (i.e., no seed).
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., "remc_kernel").

    Raises:
      ValueError: `inverse_temperatures` doesn't have statically known 1D shape.
    """
    # We begin by creating a fake logp, with the correct scores
    @jax.custom_jvp
    def fake_logp(x, sigma):
      return jnp.zeros(x.shape[:-1])
    @fake_logp.defjvp
    def fake_logp_jvp(primals, tangents):
      x, sigma = primals
      x_dot,_ = tangents
      primal_out = jnp.zeros(x.shape[:-1])
      s = target_score_fn(x, sigma)
      tangent_out = jnp.sum(x_dot*s, axis=-1)
      return primal_out, tangent_out

    self._parameters = {k: v for k, v in locals().items() if v is not self}
    self._seed_stream = SeedStream(seed, salt='replica_mc')
    self._parameters['target_log_prob_fn'] = fake_logp

  @property
  def target_score_fn(self):
    return self._parameters['target_score_fn']

  @property
  def target_log_prob_fn(self):
    return self._parameters['target_log_prob_fn']

  @property
  def num_delta_logp_steps(self):
    return self._parameters['num_delta_logp_steps']

  @property
  def inverse_temperatures(self):
    return self._parameters['inverse_temperatures']

  def num_replica(self):
    """Integer (`Tensor`) number of replicas being tracked."""
    return tf.constant(prefer_static.size0(self.inverse_temperatures))

  @property
  def make_kernel_fn(self):
    return self._parameters['make_kernel_fn']

  @property
  def swap_proposal_fn(self):
    return self._parameters['swap_proposal_fn']

  @property
  def seed(self):
    return self._parameters['seed']

  @property
  def validate_args(self):
    return self._parameters['validate_args']

  @property
  def name(self):
    return self._parameters['name']

  @property
  def parameters(self):
    """Return `dict` of ``__init__`` arguments and their values."""
    return self._parameters

  @property
  def is_calibrated(self):
    return True

  def one_step(self, current_state, previous_kernel_results, seed=None):
    """Takes one step of the TransitionKernel.

    Args:
      current_state: `Tensor` or Python `list` of `Tensor`s representing the
        current state(s) of the Markov chain(s).
      previous_kernel_results: A (possibly nested) `tuple`, `namedtuple` or
        `list` of `Tensor`s representing internal calculations made within the
        previous call to this function (or as returned by `bootstrap_results`).
      seed: Optional, a seed for reproducible sampling.

    Returns:
      next_state: `Tensor` or Python `list` of `Tensor`s representing the
        next state(s) of the Markov chain(s).
      kernel_results: A (possibly nested) `tuple`, `namedtuple` or `list` of
        `Tensor`s representing internal calculations made within this function.
        This inculdes replica states.
    """

    # The code below propagates one step states of shape
    #  [n_replica] + batch_shape + event_shape.
    #
    # The step is done in three parts:
    #  1) Call one_step to transition states via a tempered version of
    #     self.target_log_prob_fn (see _replica_target_log_prob).
    #  2) Permute values in states
    #  3) Update state-dependent values, such as log_probs.
    #
    # We chose to swap states, rather than temperatures, because...
    # (i)  If swapping temperatures, you *still* have to swap log_probs to
    #      determine acceptance, as well as states (for kernel results).
    #      So it's just as difficult to swap temperatures.
    # (ii) If swapping temperatures, you have to take care to swap any user-
    #      supplied temperature related things (like step size).
    #      A-priori, we don't know what else will need to be swapped!
    # (iii)In both cases, the kernel results need to be updated in a non-trivial
    #      manner....so we either special-case, or use bootstrap.

    with tf.name_scope(mcmc_util.make_name(self.name, 'remc', 'one_step')):
      # Force a read in case the `inverse_temperatures` is a `tf.Variable`.
      inverse_temperatures = tf.convert_to_tensor(
          previous_kernel_results.inverse_temperatures,
          name='inverse_temperatures')

      target_score_for_inner_kernel = partial(self.target_score_fn,
                                              sigma=inverse_temperatures)
      target_log_prob_for_inner_kernel = partial(self.target_log_prob_fn,
                                                sigma=inverse_temperatures)

      # Seed handling complexity is due to users possibly expecting an old-style
      # stateful seed to be passed to `self.make_kernel_fn`, and no seed
      # expected by `kernel.one_step`.
      # In other words:
      # - We try `make_kernel_fn` without a seed first; this is the future. The
      #   kernel will receive a seed later, as part of `one_step`.
      # - If the user code doesn't like that (Python complains about a missing
      #   required argument), we warn and fall back to the previous behavior.
      try:
        inner_kernel = self.make_kernel_fn(  # pylint: disable=not-callable
            target_log_prob_for_inner_kernel,
            target_score_for_inner_kernel)
      except TypeError as e:
        if 'argument' not in str(e):
          raise
        warnings.warn(
            'The `seed` argument to `ReplicaExchangeMC`s `make_kernel_fn` is '
            'deprecated. `TransitionKernel` instances now receive seeds via '
            '`one_step`.')
        inner_kernel = self.make_kernel_fn(  # pylint: disable=not-callable
            target_log_prob_for_inner_kernel,
            target_score_for_inner_kernel, self._seed_stream())

      # Now that we've constructed the TransitionKernel instance:
      # - If we were given a seed, we sanitize it to stateless and pass along
      #   to `kernel.one_step`. If it doesn't like that, we crash and propagate
      #   the error.  Rationale: The contract is stateless sampling given
      #   seed, and doing otherwise would not meet it.
      # - If not given a seed, we don't pass one along. This avoids breaking
      #   underlying kernels lacking a `seed` arg on `one_step`.
      # TODO(b/159636942): Clean up after 2020-09-20.
      if seed is not None:
        seed = samplers.sanitize_seed(seed)
        inner_seed, swap_seed, logu_seed = samplers.split_seed(
            seed, n=3, salt='remc_one_step')
        inner_kwargs = dict(seed=inner_seed)
      else:
        if self._seed_stream.original_seed is not None:
          warnings.warn(mcmc_util.SEED_CTOR_ARG_DEPRECATION_MSG)
        inner_kwargs = {}
        swap_seed, logu_seed = samplers.split_seed(self._seed_stream())
      [
          pre_swap_replica_states,
          pre_swap_replica_results,
      ] = inner_kernel.one_step(
          previous_kernel_results.post_swap_replica_states,
          previous_kernel_results.post_swap_replica_results,
          **inner_kwargs)

      pre_swap_replica_target_log_prob = _get_field(
          # These are tempered log probs (have been divided by temperature).
          pre_swap_replica_results, 'target_log_prob')

      dtype = pre_swap_replica_target_log_prob.dtype
      replica_and_batch_shape = prefer_static.shape(
          pre_swap_replica_target_log_prob)
      batch_shape = replica_and_batch_shape[1:]
      replica_and_batch_rank = prefer_static.rank(
          pre_swap_replica_target_log_prob)
      num_replica = prefer_static.size0(inverse_temperatures)

      inverse_temperatures = mcmc_util.left_justified_broadcast_to(
          inverse_temperatures, replica_and_batch_shape)

      # Now that each replica has done one_step, it is time to consider swaps.

      # swap.shape = [n_replica], and is a "once only" permutation, meaning it
      # is achievable by a sequence of pairwise permutations, where each element
      # is moved at most once.
      # E.g. if swaps = [1, 0, 2], we will consider swapping temperatures 0 and
      # 1, keeping 2 fixed.  This exact same swap is considered for *every*
      # batch member.  Of course some batch members may accept and some reject.
      swaps = tf.cast(
          self.swap_proposal_fn(  # pylint: disable=not-callable
              num_replica, batch_shape=batch_shape, seed=swap_seed),
          dtype=tf.int32)
      null_swaps = mcmc_util.left_justified_expand_dims_like(
          tf.range(num_replica, dtype=swaps.dtype), swaps)
      swaps = _maybe_embed_swaps_validation(swaps, null_swaps,
                                            self.validate_args)

      # Un-temper the log probs.  E.g., for replica k, at point x_k, this is
      # Log[p(x_k)], and *not* Log[p_x(x_k)] = Log[p(x_k)] * beta_k.
      # untempered_pre_swap_replica_target_log_prob = (
      #     pre_swap_replica_target_log_prob / inverse_temperatures)

      # Since `swaps` is its own inverse permutation we automatically know the
      # swap counterpart: range(num_replica). We use this idea to compute the
      # acceptance in a vectorized manner at the cost of wasting roughly half
      # our computation. Although we could use `unique` to solve this problem,
      # we expect the cost of `unique` to be higher than the dozens of wasted
      # arithmetic calculations. Worse, it'd mean we need dynamic sized Tensors
      # (eg, using `tf.where(bool)`) and so we wouldn't be able to XLA compile.

      # Note: diffs would normally be "proposed - current" however energy is
      # flipped since `energy == -log_prob`.
      # energy_diff = (
      #     untempered_pre_swap_replica_target_log_prob -
      #     mcmc_util.index_remapping_gather(
      #         untempered_pre_swap_replica_target_log_prob,
      #         swaps, name='gather_swap_tlp'))
      swapped_inverse_temperatures = mcmc_util.index_remapping_gather(
          inverse_temperatures, tf.cast(swaps, tf.int64), name='gather_swap_temps')
      # inverse_temp_diff = swapped_inverse_temperatures - inverse_temperatures

      # Here we compute the acceptance ratio for transition
      # First, at a given temperature
      v = mcmc_util.index_remapping_gather(pre_swap_replica_states,
                              swaps, name='gather_swap_steps')[0] - pre_swap_replica_states[0]
      cs = pre_swap_replica_states[0]
      @jax.vmap
      def integrand(t):
        return jnp.sum(self._parameters['target_score_fn']( t * v + cs, inverse_temperatures)*v, axis=-1)
      delta_logp1 = simps(integrand, 0.,1., self._parameters['num_delta_logp_steps'])

      # Now we compute the reverse
      v = -v
      cs = mcmc_util.index_remapping_gather(pre_swap_replica_states,
                              swaps, name='gather_swap_steps')[0]
      @jax.vmap
      def integrand(t):
        return jnp.sum(self._parameters['target_score_fn']( t * v + cs, swapped_inverse_temperatures)*v, axis=-1)
      delta_logp2 = simps(integrand, 0.,1., self._parameters['num_delta_logp_steps'])

      # If i and j are swapping, log_accept_ratio[] i and j are equal.
      log_accept_ratio = delta_logp1 + delta_logp2

      log_accept_ratio = tf.where(
          tf.math.is_finite(log_accept_ratio),
          log_accept_ratio, tf.constant(-np.inf, dtype=dtype))

      # Produce Log[Uniform] draws that are identical at swapped indices.
      log_uniform = tf.math.log(
          samplers.uniform(shape=replica_and_batch_shape,
                           dtype=dtype,
                           seed=logu_seed))
      anchor_swaps = tf.cast(tf.minimum(swaps, null_swaps),tf.int64)
      log_uniform = mcmc_util.index_remapping_gather(log_uniform, anchor_swaps)

      is_swap_accepted_mask = tf.less(
          log_uniform,
          log_accept_ratio,
          name='is_swap_accepted_mask')

      def _swap_tensor(x):
        return mcmc_util.choose(
            is_swap_accepted_mask,
            mcmc_util.index_remapping_gather(x, swaps), x)

      post_swap_replica_states = [
          _swap_tensor(s) for s in pre_swap_replica_states]

      expanded_null_swaps = mcmc_util.left_justified_broadcast_to(
          null_swaps, replica_and_batch_shape)
      is_swap_proposed = _compute_swap_notmatrix(
          # Broadcast both so they have shape [num_replica] + batch_shape.
          # This (i) makes them have same shape as is_swap_accepted, and
          # (ii) keeps shape consistent if someday swaps has a batch shape.
          expanded_null_swaps,
          mcmc_util.left_justified_broadcast_to(swaps, replica_and_batch_shape))

      # To get is_swap_accepted in ordered position, we use
      # _compute_swap_notmatrix on current and next replica positions.
      post_swap_replica_position = _swap_tensor(expanded_null_swaps)

      is_swap_accepted = _compute_swap_notmatrix(
          post_swap_replica_position,
          expanded_null_swaps)

      post_swap_states = [s[0] for s in post_swap_replica_states]

      post_swap_replica_results = _make_post_swap_replica_results(
          pre_swap_replica_results,
          inverse_temperatures,
          swapped_inverse_temperatures,
          is_swap_accepted_mask,
          _swap_tensor)

      if mcmc_util.is_list_like(current_state):
        # We *always* canonicalize the states in the kernel results.
        states = post_swap_states
      else:
        states = post_swap_states[0]

      post_swap_kernel_results = ReplicaExchangeMCKernelResults(
          post_swap_replica_states=post_swap_replica_states,
          pre_swap_replica_results=pre_swap_replica_results,
          post_swap_replica_results=post_swap_replica_results,
          is_swap_proposed=is_swap_proposed,
          is_swap_accepted=is_swap_accepted,
          is_swap_proposed_adjacent=_sub_diag(is_swap_proposed),
          is_swap_accepted_adjacent=_sub_diag(is_swap_accepted),
          # Store the original pkr.inverse_temperatures in case its a
          # `tf.Variable`.
          inverse_temperatures=previous_kernel_results.inverse_temperatures,
          swaps=swaps,
          seed=samplers.zeros_seed() if seed is None else seed,
      )

      return states, post_swap_kernel_results

  def bootstrap_results(self, init_state):
    """Returns an object with the same type as returned by `one_step`.

    Args:
      init_state: `Tensor` or Python `list` of `Tensor`s representing the
        initial state(s) of the Markov chain(s).

    Returns:
      kernel_results: A (possibly nested) `tuple`, `namedtuple` or `list` of
        `Tensor`s representing internal calculations made within this function.
        This inculdes replica states.
    """
    with tf.name_scope(mcmc_util.make_name(
        self.name, 'remc', 'bootstrap_results')):
      init_state, unused_is_multipart_state = mcmc_util.prepare_state_parts(
          init_state)

      inverse_temperatures = tf.convert_to_tensor(
          self.inverse_temperatures,
          name='inverse_temperatures')

      # We will now replicate each of a possible batch of initial stats, one for
      # each inverse_temperature. So if init_state=[x, y] of shapes [Sx, Sy]
      # then the new shape is [(T, Sx), (T, Sy)] where (a, b) means
      # concatenation and T=shape(inverse_temperature).
      num_replica = prefer_static.size0(inverse_temperatures)
      replica_shape = tf.convert_to_tensor([num_replica])

      replica_states = [
          tf.broadcast_to(  # pylint: disable=g-complex-comprehension
              x,
              prefer_static.concat([replica_shape, prefer_static.shape(x)],
                                   axis=0),
              name='replica_states')
          for x in init_state
      ]


      target_score_for_inner_kernel = partial(self.target_score_fn,
                                              sigma=inverse_temperatures)
      target_log_prob_for_inner_kernel = partial(self.target_log_prob_fn,
                                                sigma=inverse_temperatures)

      # Seed handling complexity is due to users possibly expecting an old-style
      # stateful seed to be passed to `self.make_kernel_fn`.
      # In other words:
      # - We try `make_kernel_fn` without a seed first; this is the future. The
      #   kernel will receive a seed later, as part of `one_step`.
      # - If the user code doesn't like that (Python complains about a missing
      #   required argument), we fall back to the previous behavior and warn.
      try:
        inner_kernel = self.make_kernel_fn(  # pylint: disable=not-callable
            target_log_prob_for_inner_kernel,
            target_score_for_inner_kernel)
      except TypeError as e:
        if 'argument' not in str(e):
          raise
        warnings.warn(
            'The second (`seed`) argument to `ReplicaExchangeMC`s '
            '`make_kernel_fn` is deprecated. `TransitionKernel` instances now '
            'receive seeds via `bootstrap_results` and `one_step`. This '
            'fallback may become an error 2020-09-20.')
        inner_kernel = self.make_kernel_fn(  # pylint: disable=not-callable
            target_log_prob_for_inner_kernel,
            target_score_for_inner_kernel, self._seed_stream())

      replica_results = inner_kernel.bootstrap_results(replica_states)

      pre_swap_replica_target_log_prob = _get_field(
          replica_results, 'target_log_prob')

      replica_and_batch_shape = prefer_static.shape(
          pre_swap_replica_target_log_prob)
      batch_shape = replica_and_batch_shape[1:]

      inverse_temperatures = mcmc_util.left_justified_broadcast_to(
          inverse_temperatures, replica_and_batch_shape)

      # Pretend we did a "null swap", which will always be accepted.
      swaps = mcmc_util.left_justified_broadcast_to(
          tf.range(num_replica), replica_and_batch_shape)
      # is_swap_accepted.shape = [n_replica, n_replica] + batch_shape.
      is_swap_accepted = distribution_util.rotate_transpose(
          tf.eye(num_replica, batch_shape=batch_shape, dtype=tf.bool),
          shift=2)

      post_swap_replica_results = _make_post_swap_replica_results(
          replica_results,
          inverse_temperatures,
          inverse_temperatures,
          is_swap_accepted[0],
          lambda x: x,
      )

      return ReplicaExchangeMCKernelResults(
          post_swap_replica_states=replica_states,
          pre_swap_replica_results=replica_results,
          post_swap_replica_results=post_swap_replica_results,
          is_swap_proposed=is_swap_accepted,
          is_swap_accepted=is_swap_accepted,
          is_swap_proposed_adjacent=_sub_diag(is_swap_accepted),
          is_swap_accepted_adjacent=_sub_diag(is_swap_accepted),
          inverse_temperatures=self.inverse_temperatures,
          swaps=swaps,
          seed=samplers.zeros_seed(),
      )

def _maybe_embed_swaps_validation(swaps, null_swaps, validate_args):
  """Return `swaps`, possibly with embedded "once only" assertion."""
  if not validate_args:
    return swaps

  assertions = [
      assert_util.assert_equal(
          null_swaps,
          mcmc_util.index_remapping_gather(swaps, swaps),
          message=(
              'Proposed replica swaps must be consist of "once only '
              'swaps," i.e., be a self-inverse permutation, '
              '`range(swaps.shape[0]) == gather(swaps, swaps).')),
  ]
  with tf.control_dependencies(assertions):
    return tf.identity(swaps)


def _make_post_swap_replica_results(pre_swap_replica_results,
                                    inverse_temperatures,
                                    swapped_inverse_temperatures,
                                    is_swap_accepted_mask, swap_tensor_fn):
  """Return Kernel results, valid for post-swap states.

  Fields will be removed if they cannot be updated in an unambiguous manner.

  Args:
    pre_swap_replica_results: Kernel results obtained by running
      inner_kernel.one_step before swapping.
    inverse_temperatures: Tensor of inverse temperatures.
    swapped_inverse_temperatures: Tensor of inverse temperatures, permuted by
      swaps.
    is_swap_accepted_mask: Shape [num_replica] + batch_shape boolean Tensor
      telling which swaps were accepted.  Returns Kernel results of same type as
      pre_swap_replica_results.
    swap_tensor_fn: Callable. For `x.shape = [num_replica] + batch_shape`,
      swap_tensor_fn(x) performs swaps where they are accepted, and does not
      swap otherwise.

  Returns:
    new_replica_results:  Same type as pre_swap_replica_results.

  Raises:
    NotImplementedError: If type of [nested] results is not handled.
  """
  if not isinstance(pre_swap_replica_results,
                    metropolis_hastings.MetropolisHastingsKernelResults):
    # TODO(b/143702650) Handle other kernels.
    raise NotImplementedError(
        '`pre_swap_replica_results` currently works only for '
        'MetropolisHastingsKernelResults.  Found: {}. '
        'Please file a request with the TensorFlow Probability team.'.format(
            type(pre_swap_replica_results)))

  kr = pre_swap_replica_results
  dtype = swapped_inverse_temperatures.dtype

  # Hard to modify proposed_results in an um-ambiguous manner.
  # ...we also don't need to.
  kr = kr._replace(
      proposed_results=tf.convert_to_tensor(np.nan, dtype=dtype),
      proposed_state=tf.convert_to_tensor(np.nan, dtype=dtype),
  )

  replica_and_batch_rank = prefer_static.rank(kr.log_accept_ratio)

  # After using swap_tensor_fn on "values", values will be multiplied by the
  # swapped_inverse_temperatures.  We need it to be multiplied instead by the
  # inverse temperature corresponding to its index.
  it_ratio_raw = inverse_temperatures / swapped_inverse_temperatures
  it_ratio = tf.where(
      is_swap_accepted_mask,
      mcmc_util.left_justified_expand_dims_to(it_ratio_raw,
                                              replica_and_batch_rank),
      tf.convert_to_tensor(1.0, dtype=dtype))

  def _swap_then_retemper(x):
    x, is_multipart = mcmc_util.prepare_state_parts(x)
    it_ratio_ = mcmc_util.left_justified_expand_dims_like(it_ratio, x[0])
    x = [swap_tensor_fn(x_part) * it_ratio_ for x_part in x]
    if not is_multipart:
      x = x[0]
    return x

  if isinstance(kr.accepted_results,
                hmc.UncalibratedHamiltonianMonteCarloKernelResults):
    kr = kr._replace(
        accepted_results=kr.accepted_results._replace(
            target_log_prob=_swap_then_retemper(
                kr.accepted_results.target_log_prob),
            grads_target_log_prob=_swap_then_retemper(
                kr.accepted_results.grads_target_log_prob)))
  elif isinstance(kr.accepted_results,
                  random_walk_metropolis.UncalibratedRandomWalkResults):
    kr = kr._replace(
        accepted_results=kr.accepted_results._replace(
            target_log_prob=_swap_then_retemper(
                kr.accepted_results.target_log_prob)))
  else:
    # TODO(b/143702650) Handle other kernels.
    raise NotImplementedError(
        'Only HMC and RWMH Kernels are handled at this time. Please file a '
        'request with the TensorFlow Probability team.')

  return kr


# TODO(b/111801087): Use a standardized API, when available.
def _get_field(kernel_results, field_name):
  """Get field value from kernel_results or kernel_results.accepted_results."""
  attr = getattr(kernel_results, field_name, None)
  if attr is not None:
    return attr
  accepted_results = getattr(kernel_results, 'accepted_results', None)
  if accepted_results is None:
    raise TypeError('Cannot extract {} from {}'.format(
        field_name, kernel_results))
  attr = getattr(accepted_results, field_name)
  if attr is None:
    raise TypeError('Cannot extract {} from {}'.format(
        field_name, kernel_results))
  return attr


def _compute_swap_notmatrix(before_positions, after_positions):
  """Determine which before_positions were equal to after_positions.

  Args:
    before_positions: Shape [n_replica,...] integer Tensor.  Replica positions
      before an swap.
    after_positions: Shape [n_replica,...] integer Tensor.  Replica positions
      after an swap.

  Returns:
    swaps: Shape [n_replica, n_replica, ...] boolean Tensor.
      swaps[i, j, ..] = True just when
      before_positions[i,...] == after_positions[j, ...].
  """
  with tf.name_scope('compute_swap_notmatrix'):
    # Which one is expanded doesn't matter for once-only swaps, since i --> j if
    # and only if j --> i.
    # Nonetheless, these are the correct expansions if general permutations are
    # allowed.
    return tf.equal(
        before_positions,
        after_positions[:, tf.newaxis],
    )


def _sub_diag(nonmatrix):
  """Get the first sub-diagonal of a shape [N, N, ...] 'non matrix'."""
  with tf.name_scope('sub_matrix'):
    # TODO(b/143702351) Once array_ops.matrix_diag_part_v3 is ready and exposed,
    # replace the call to matrix_diag_part_v2 below with tf.linalg.matrix_diag.
    # We can also stop special casing for matrix_dim < 2 at that point.
    # Until then, OpError raised for 1x1 matricies without static shape.
    # In fact, non-static shape breaks matrix_diag_part_v2, so we must raise
    # this message now.
    # See http://b/138403336 for the TF issue tracker.
    if not tensorshape_util.is_fully_defined(nonmatrix.shape[:2]):
      raise ValueError(
          '`inverse_temperatures did not have statically defined shape, '
          'which breaks tracking of is_swap_{proposed,accepted}.  '
          'Please provide an inverse_temperatures with statically known shape.')

    # The sub-matrix of a 1x1 matrix is not defined (throws exception), so in
    # this special case return an empty matrix.
    # TODO(b/143702351) Remove this special case handling once
    # matrix_diag_part_v3 is ready.
    matrix_dim = prefer_static.size0(nonmatrix)
    if matrix_dim is not None and matrix_dim < 2:
      # Shape is [..., 0], so returned tensor is empty, thus contains no
      # values...and therefore the fact that we use 'ones' doesn't matter.
      shape = prefer_static.pad(
          prefer_static.shape(nonmatrix)[2:],
          paddings=[[0, 1]],
          constant_values=0)
      matrix_sub_diag = tf.cast(tf.ones(shape), nonmatrix.dtype)

    else:
      # Get first sub-diagonal.  `padding_value` is not used (since matrix is
      # square), but is required for the API since this is raw gen_array_ops.
      matrix_sub_diag = tf.raw_ops.MatrixDiagPartV2(
          input=distribution_util.rotate_transpose(nonmatrix, shift=-2),
          k=tf.convert_to_tensor(-1, dtype=tf.int32),
          padding_value=tf.cast(0.0, dtype=nonmatrix.dtype))

    return distribution_util.rotate_transpose(matrix_sub_diag, shift=1)


def simps(f, a, b, N=128):
    """Approximate the integral of f(x) from a to b by Simpson's rule.
    Simpson's rule approximates the integral \int_a^b f(x) dx by the sum:
    (dx/3) \sum_{k=1}^{N/2} (f(x_{2i-2} + 4f(x_{2i-1}) + f(x_{2i}))
    where x_i = a + i*dx and dx = (b - a)/N.
    Parameters
    ----------
    f : function
        Vectorized function of a single variable
    a , b : numbers
        Interval of integration [a,b]
    N : (even) integer
        Number of subintervals of [a,b]
    Returns
    -------
    float
        Approximation of the integral of f(x) from a to b using
        Simpson's rule with N subintervals of equal length.
    Examples
    --------
    >>> simps(lambda x : 3*x**2,0,1,10)
    1.0
    Notes:
    ------
    Stolen from: https://www.math.ubc.ca/~pwalls/math-python/integration/simpsons-rule/
    """
    if N % 2 == 1:
        raise ValueError("N must be an even integer.")
    dx = (b - a) / N
    x = jnp.linspace(a, b, N + 1)
    y = f(x)
    S = dx / 3 * jnp.sum(y[0:-1:2] + 4 * y[1::2] + y[2::2], axis=0)
    return S
