"""Score-based MCMC samplers"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax
import jax.numpy as jnp
import tensorflow_probability as tfp; tfp = tfp.experimental.substrates.jax

__all__ = [
    'ScoreUncalibratedHamiltonianMonteCarlo',
    'ScoreHamitonianMonteCarlo'
]

class ScoreUncalibratedHamiltonianMonteCarlo(tfp.mcmc.UncalibratedHamiltonianMonteCarlo):
  def __init__(self,
               target_score_fn,
               step_size,
               num_leapfrog_steps,
               num_delta_logp_steps,
               state_gradients_are_stopped=False,
               seed=None,
               store_parameters_in_results=False,
               name=None):
    # We begin by creating a fake logp, with the correct scores
    @jax.custom_jvp
    def fake_logp(x):
      return 0.
    @fake_logp.defjvp
    def fake_logp_jvp(primals, tangents):
      x, = primals
      x_dot, = tangents
      primal_out = fake_logp(x)
      s = target_score_fn(x)
      tangent_out = x_dot.dot(s)
      return primal_out, tangent_out

    super().__init__(fake_logp,
                     step_size,
                     num_leapfrog_steps,
                     state_gradients_are_stopped,
                     seed,
                     store_parameters_in_results, name)
    self._parameters['target_score_fn'] = target_score_fn
    self._parameters['num_delta_logp_steps'] = num_delta_logp_steps

  def one_step(self, current_state, previous_kernel_results, seed=None):
    """
    Wrapper over the normal HMC steps
    """
    next_state_parts, new_kernel_results = super().one_step(current_state,
                                                            previous_kernel_results,
                                                            seed)
    # We need to integrate the score over a path between input and output points
    # Direction of integration
    v = next_state_parts - current_state
    @jax.vmap
    def integrand(t):
      return self._parameters['target_score_fn']( t * v + current_state).dot(v)
    delta_logp = simps(integrand,0.,1., self._parameters['num_delta_logp_steps'])
    new_kernel_results2 = new_kernel_results._replace(log_acceptance_correction=new_kernel_results.log_acceptance_correction + delta_logp)
    return next_state_parts, new_kernel_results2


class ScoreHamiltonianMonteCarlo(tfp.mcmc.HamiltonianMonteCarlo):

  def __init__(self,
               target_score_fn,
               step_size,
               num_leapfrog_steps,
               num_delta_logp_steps,
               state_gradients_are_stopped=False,
               step_size_update_fn=None,
               seed=None,
               store_parameters_in_results=False,
               name=None):
    """Initializes this transition kernel.
    Args:
      target_score_fn: Python callable which takes an argument like
        `current_state` (or `*current_state` if it's a list) and returns the score
        of the log-density under the target distribution.
      step_size: `Tensor` or Python `list` of `Tensor`s representing the step
        size for the leapfrog integrator. Must broadcast with the shape of
        `current_state`. Larger step sizes lead to faster progress, but
        too-large step sizes make rejection exponentially more likely. When
        possible, it's often helpful to match per-variable step sizes to the
        standard deviations of the target distribution in each variable.
      num_leapfrog_steps: Integer number of steps to run the leapfrog integrator
        for. Total progress per HMC step is roughly proportional to
        `step_size * num_leapfrog_steps`.
      num_delta_logp_steps: Integer number of steps to run the integrator
        for estimating the change in logp.
      state_gradients_are_stopped: Python `bool` indicating that the proposed
        new state be run through `tf.stop_gradient`. This is particularly useful
        when combining optimization over samples from the HMC chain.
        Default value: `False` (i.e., do not apply `stop_gradient`).
      step_size_update_fn: Python `callable` taking current `step_size`
        (typically a `tf.Variable`) and `kernel_results` (typically
        `collections.namedtuple`) and returns updated step_size (`Tensor`s).
        Default value: `None` (i.e., do not update `step_size` automatically).
      seed: Python integer to seed the random number generator. Deprecated, pass
        seed to `tfp.mcmc.sample_chain`.
      store_parameters_in_results: If `True`, then `step_size` and
        `num_leapfrog_steps` are written to and read from eponymous fields in
        the kernel results objects returned from `one_step` and
        `bootstrap_results`. This allows wrapper kernels to adjust those
        parameters on the fly. This is incompatible with `step_size_update_fn`,
        which must be set to `None`.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., 'hmc_kernel').
    """
    if step_size_update_fn and store_parameters_in_results:
      raise ValueError('It is invalid to simultaneously specify '
                       '`step_size_update_fn` and set '
                       '`store_parameters_in_results` to `True`.')
    self._seed_stream = tfp.util.SeedStream(seed, salt='hmc')
    uhmc_kwargs = {} if seed is None else dict(seed=self._seed_stream())
    mh_kwargs = {} if seed is None else dict(seed=self._seed_stream())
    self._impl = tfp.mcmc.MetropolisHastings(
        inner_kernel=ScoreUncalibratedHamiltonianMonteCarlo(
            target_score_fn=target_score_fn,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
            num_delta_logp_steps=num_delta_logp_steps,
            state_gradients_are_stopped=state_gradients_are_stopped,
            name=name or 'hmc_kernel',
            store_parameters_in_results=store_parameters_in_results,
            **uhmc_kwargs),
        **mh_kwargs)
    self._parameters = self._impl.inner_kernel.parameters.copy()
    self._parameters['step_size_update_fn'] = step_size_update_fn
    self._parameters['seed'] = seed

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
