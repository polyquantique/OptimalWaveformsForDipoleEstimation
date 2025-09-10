"""
This file contains functions that can generate Hermite Gaussian functions, H_n(t). This
file was originally taken from https://github.com/Rob217/hermite-functions/tree/master
and modified here so that it is compatible with JAX.

The main function of this file "scaled_Hermite" which scales
the Hermite Gaussian functions so that Psi_n(t) =(1/sqrt(T))*H_n(t/T) (see Appendix D2 in paper)

For more information on Hermite Gaussian
functions see https://en.wikipedia.org/wiki/Hermite_polynomials#Hermite_functions

"""

from functools import partial
import jax
import jax.numpy as jnp
from jax import jit

## Restricting the usage to the CPU
jax.config.update("jax_default_device", jax.devices("cpu")[0])

################################################################################
@jit
def herm_recursive(ind, xval, psi_m_minus_2, psi_m_minus_1):
    """
    Parameters
    ----------
    ind: non-negative integer, Hermite-Gaussian function of order = ind is the output of
         this function
    xval: real number or a 1d array containing real numbers,
          the value(s) (can be negative) at which the Hermite Gaussian
          function is evaluated
    psi_m_minus_2:real number or a 1d array containing real numbers with len = len(xval),
                    the value(s) of the (ind-2) order Hermite-Gaussian function
    psi_m_minus_1:real number or a 1d array containing real numbers with len = len(xval),
                    the value(s) of the (ind-1) order Hermite-Gaussian function

    Returns
    -------
    psi_n: 1d array with real numbers with len = len(xval).
    Contains the value of the function evaluated at different xvals evaluted
    using the recursion relation

    The Hermite Gaussian functions are shown here:
    https://en.wikipedia.org/wiki/Hermite_polynomials#Hermite_functions
    """

    return (
        jnp.sqrt(2 / ind) * xval * psi_m_minus_1
        - jnp.sqrt((ind - 1) / ind) * psi_m_minus_2
    )


@jit
def herm_analytic(ind, xval):
    """
    Parameters
    ----------
    ind: non-negative integer, Hermite-Gaussian function of order = ind is the output of
         this function
    xval: real number or a 1d array containing real numbers,
          the value(s) (can be negative) at which the Hermite Gaussian
          function is evaluated
    Returns
    -------
    psi_n: 1d array with real numbers with len = len(xval).
    Contains the value of the function evaluated at different xvals evaluted
    using the analytic expression

    The Hermite Gaussian functions are shown here:
    https://en.wikipedia.org/wiki/Hermite_polynomials#Hermite_functions
    """

    # Compute all possible cases
    case_0 = jnp.pi ** (-1 / 4) * jnp.exp(-(xval**2) / 2)

    case_1 = jnp.sqrt(2) * jnp.pi ** (-1 / 4) * xval * jnp.exp(-(xval**2) / 2)

    case_2 = (
        (jnp.sqrt(2) * jnp.pi ** (1 / 4)) ** (-1)
        * (2 * xval**2 - 1)
        * jnp.exp(-(xval**2) / 2)
    )

    case_3 = (
        (jnp.sqrt(3) * jnp.pi ** (1 / 4)) ** (-1)
        * (2 * xval**3 - 3 * xval)
        * jnp.exp(-(xval**2) / 2)
    )

    case_4 = (
        (2 * jnp.sqrt(6) * jnp.pi ** (1 / 4)) ** (-1)
        * (4 * xval**4 - 12 * xval**2 + 3)
        * jnp.exp(-(xval**2) / 2)
    )

    case_5 = (
        (2 * jnp.sqrt(15) * jnp.pi ** (1 / 4)) ** (-1)
        * (4 * xval**5 - 20 * xval**3 + 15 * xval)
        * jnp.exp(-(xval**2) / 2)
    )

    # Define conditions
    conditions = [ind == 0, ind == 1, ind == 2, ind == 3, ind == 4, ind == 5]

    # Define corresponding values
    cases = [case_0, case_1, case_2, case_3, case_4, case_5]

    # Use jnp.select to choose the correct value
    result = jnp.select(conditions, cases, default=jnp.nan)

    return result


@partial(jax.jit, static_argnums=(0,))  # Mark n as static
def herm_all_n(nmax, xval):
    """
    Parameters
    ----------
    nmax: non-negative integer, Hermite functions upto nmax is the output of
         this function
    xval: real number or a 1d array containing real numbers,
          the value(s) (can be negative) at which the Hermite Gaussian
          functions are evaluated
    Returns
    -------
    psi_n: 2d array with real numbers with shape = (nmax+1,len(xval)).
    Each row corresponds to different order Hermite Gaussian function
    ranging from 0 to nmax evaluated at different xvals along the columns

    """
    # Check if xval is a scalar
    if jnp.isscalar(xval):
        psi_n = jnp.zeros((nmax + 1, 1))  # Use (nmax+1, 1) when x is scalar
    else:
        # Use x.shape when x is an array
        psi_n = jnp.zeros((nmax + 1,) + xval.shape)

    psi_n = psi_n.at[0, :].set(herm_analytic(0, xval))
    if nmax == 0:
        return psi_n

    psi_n = psi_n.at[1, :].set(herm_analytic(1, xval))
    if nmax == 1:
        return psi_n

    for m in range(2, nmax + 1):
        psi_n = psi_n.at[m, :].set(
            herm_recursive(m, xval, psi_n[m - 2, :], psi_n[m - 1, :])
        )

    return psi_n


##########################################################################################
### Following function built for our optimization:

def scaled_hermite(nmax, tval, tcap):
    """
    Parameters
    ----------
    nmax: non-negative integer, Hermite functions upto nmax is the output of
         this function
    tval: real number or a 1d array containing real numbers,
          the time value(s) (can be negative) at which the Hermite Gaussian
          functions are evaluated
    tcap: non-negative real number, determines the width of the pulses
          (same as T in Appendix D2, see paper). The variance of the
          nth-order Hermite Gaussian is given by sqrt(n+0.5)*T

    Returns
    -------
    scaled_pulse: 2d array with real numbers with shape = (nmax+1,len(tval)).
    Each row corresponds to different order Hermite Gaussian function (scaled)
    ranging from 0 to nmax evaluated at different tvals along the columns

     The Hermite Gaussian functions are evaluated with extra scaling:
     Psi_{n}(t) = sqrt(1/T)*H_{n}(t/T) (this will ensure the variance
     of the pulse =(n+0.5)*T) where H_{n}(t) are the functions show here:
     https://en.wikipedia.org/wiki/Hermite_polynomials#Hermite_functions
    """
    ### We will scale the argument and the whole function so the norm is 1
    ### and we have additional width parameter:
    scaled_t = tval / tcap
    scaled_pulse = jnp.sqrt(1 / tcap) * herm_all_n(nmax, scaled_t)

    return scaled_pulse
