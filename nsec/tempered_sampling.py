
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
    'TemperedMC',
]

class TemperedMCKernelResults(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple(
        'TemperedMCKernelResults',
        [
            # Kernel results for replicas, before any tempering
            'pre_tempering_results',

            # Kernel results for replicas, after tempering.
            'post_tempering_results',

            # The inverse_temperatures used to calculate these results.
            'pre_tempering_inverse_temperatures',

            # The inverse_temperatures used to calculate these results.
            'post_tempering_inverse_temperatures',

            # Acceptance ratio for lowering temperature
            'tempering_log_accept_ratio'

            # Random seed for this step.
            'seed',
        ])):
  """Internal state and diagnostics for Annealed MC."""
  __slots__ = ()


class TemperedMC(kernel_base.TransitionKernel):
  """Runs one step of the Replica Exchange Monte Carlo.
  """

  @deprecation.deprecated_args(
      '2020-09-20', 'The `seed` argument is deprecated (but will work until '
      'removed). Pass seed to `tfp.mcmc.sample_chain` instead.', 'seed')
  def __init__(self,
               target_score_fn,
               inverse_temperatures,
               make_kernel_fn,
               gamma,
               num_delta_logp_steps=4,
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
    self._seed_stream = SeedStream(seed, salt='tempered_mc')
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

  @property
  def make_kernel_fn(self):
    return self._parameters['make_kernel_fn']

  @property
  def gamma(self):
    return self._parameters['gamma']

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

    with tf.name_scope(mcmc_util.make_name(self.name, 'tmc', 'one_step')):
      # Force a read in case the `inverse_temperatures` is a `tf.Variable`.
      inverse_temperatures = tf.convert_to_tensor(
          previous_kernel_results.post_tempering_inverse_temperatures,
          name='inverse_temperatures')

      target_score_for_inner_kernel = partial(self.target_score_fn,
                                              sigma=inverse_temperatures)
      target_log_prob_for_inner_kernel = partial(self.target_log_prob_fn,
                                              sigma=inverse_temperatures)

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

      if seed is not None:
        seed = samplers.sanitize_seed(seed)
        inner_seed, swap_seed, logu_seed = samplers.split_seed(
            seed, n=3, salt='tmc_one_step')
        inner_kwargs = dict(seed=inner_seed)
      else:
        if self._seed_stream.original_seed is not None:
          warnings.warn(mcmc_util.SEED_CTOR_ARG_DEPRECATION_MSG)
        inner_kwargs = {}
        swap_seed, logu_seed = samplers.split_seed(self._seed_stream())
      [
          new_state,
          pre_tempering_results,
      ] = inner_kernel.one_step(
          current_state,
          previous_kernel_results.post_tempering_results,
          **inner_kwargs)

      # Now that we have run one step, we consider maybe lowering the temperature
      # Proposed new temperature
      proposed_inverse_temperatures = self.gamma * inverse_temperatures

      # We will lower the temperature if this new proposed step is compatible with
      # a temperature swap
      v = new_state - current_state
      cs = current_state
      @jax.vmap
      def integrand(t):
        return jnp.sum(self._parameters['target_score_fn']( t * v + cs, inverse_temperatures)*v, axis=-1)
      delta_logp1 = simps(integrand, 0.,1., self._parameters['num_delta_logp_steps'])

      # Now we compute the reverse
      v = -v
      cs = new_state
      @jax.vmap
      def integrand(t):
        return jnp.sum(self._parameters['target_score_fn']( t * v + cs, swapped_inverse_temperatures)*v, axis=-1)
      delta_logp2 = simps(integrand, 0.,1., self._parameters['num_delta_logp_steps'])

      log_accept_ratio = (delta_logp1 + delta_logp2)

      log_accept_ratio = tf.where(
          tf.math.is_finite(log_accept_ratio),
          log_accept_ratio, tf.constant(-np.inf, dtype=dtype))

      # Produce Log[Uniform] draws that are identical at swapped indices.
      log_uniform = tf.math.log(
          samplers.uniform(shape=replica_and_batch_shape,
                           dtype=dtype,
                           seed=logu_seed))

      is_tempering_accepted_mask = tf.less(
          log_uniform,
          log_accept_ratio,
          name='is_tempering_accepted_mask')

      # Updating accepted inverse temperatures
      post_tempering_inverse_temperatures = mcmc_util.choose(
            is_tempering_accepted_mask,
            proposed_inverse_temperatures, inverse_temperatures)

      # Invalidating and recomputing results
      [
        new_target_log_prob,
        new_grads_target_log_prob,
      ] = mcmc_util.maybe_call_fn_and_grads(partial(self.target_log_prob_fn,
                                                    sigma=post_tempering_inverse_temperatures),
                                            new_state)

      # Updating inner kernel results
      post_tempering_results = pre_tempering_results._replace(
          proposed_results=tf.convert_to_tensor(np.nan, dtype=dtype),
          proposed_state=tf.convert_to_tensor(np.nan, dtype=dtype),
      )

      if isinstance(post_tempering_results.accepted_results,
                    hmc.UncalibratedHamiltonianMonteCarloKernelResults):
        post_tempering_results = post_tempering_results._replace(
            accepted_results=post_tempering_results.accepted_results._replace(
                target_log_prob=new_target_log_prob,
                grads_target_log_prob=new_grads_target_log_prob))
      elif isinstance(post_tempering_results.accepted_results,
                      random_walk_metropolis.UncalibratedRandomWalkResults):
        post_tempering_results = post_tempering_results._replace(
            accepted_results=post_tempering_results.accepted_results._replace(
                target_log_prob=new_target_log_prob))
      else:
        # TODO(b/143702650) Handle other kernels.
        raise NotImplementedError(
            'Only HMC and RWMH Kernels are handled at this time. Please file a '
            'request with the TensorFlow Probability team.')

      new_kernel_results = TemperedMCKernelResults(
          pre_tempering_results=pre_tempering_results,
          post_tempering_results=post_tempering_results,
          pre_tempering_inverse_temperatures=inverse_temperatures,
          post_tempering_inverse_temperatures=post_tempering_inverse_temperatures,
          tempering_log_accept_ratio=log_accept_ratio,
          seed=samplers.zeros_seed() if seed is None else seed,
      )

      return states, new_kernel_results


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
        self.name, 'tmc', 'bootstrap_results')):
      init_state, unused_is_multipart_state = mcmc_util.prepare_state_parts(
          init_state)

      inverse_temperatures = tf.convert_to_tensor(
          self.inverse_temperatures,
          name='inverse_temperatures')

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

      inner_results = inner_kernel.bootstrap_results(init_state)

      return TemperedMCKernelResults(
          pre_tempering_results=inner_results,
          post_tempering_results=inner_results,
          pre_tempering_inverse_temperatures=inverse_temperatures,
          post_tempering_inverse_temperatures=inverse_temperatures,
          tempering_log_accept_ratio=tf.zeros_like(inverse_temperatures),
          seed=samplers.zeros_seed() if seed is None else seed,
      )

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