"""
This file generates all the real harmonics, which are the basis functions of the
optimization.
"""

import jax
import jax.numpy as jnp

## Restricting the usage to the CPU
jax.config.update("jax_default_device", jax.devices("cpu")[0])


##################################################################################
def single_basis_func(ind, tval, twidth):
    """
    ind: integer greater than zero. Index corresponding to the harmonic.
    tval: positive real number, time value at which the pulse is evaluated
    twidth: positive real number, width of the pulse.

    Returns
    -------
    pulse: returns a scalar corresponding to the value of
            sqrt(2/twidth)*sin((ind*pi*tval)/twidth) when 0<= tval <= twidth
            else it returns 0
    """

    pulse = jnp.piecewise(
        tval,
        [tval < 0, (tval >= 0) & (tval <= twidth), tval > twidth],
        [
            0,
            lambda tval: jnp.sqrt(2.0 / twidth)
            * jnp.sin(ind * jnp.pi * (tval / twidth)),
            0,
        ],
    )

    return pulse


### Vectorizing the single_basis_func over "ind" argument:
# pylint: disable=invalid-name
sine_basis_func = jax.vmap(single_basis_func, in_axes=(0, None, None))


def sine_basis(nmax, tval, twidth):
    """
    nmax: integer greater than zero. Index corresponding to the maximum harmonic.
         This function returns all harmonics up to nmax.
    tval: positive real number, time value at which the pulse is evaluated
          (or an array of tvals)
    twidth: positive real number, width of the pulse.

    Returns
    -------
    pulse: returns an array corresponding to the values of
            sqrt(2/twidth)*sin((ind*pi*tval)/twidth) where ind = (1,2,...,nmax)
            when 0<= tval <= twidth else it returns zero array
    """
    ns = jnp.arange(1, nmax + 1)
    full_basis = sine_basis_func(ns, tval, twidth)
    return full_basis
