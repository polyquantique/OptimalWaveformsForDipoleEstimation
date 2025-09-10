"""
This file contains functions that can generate Harmonics: exp(-i delta t)*sin((n*pi/T)*t)
They form a basis for a function that goes to zero at the boundaries
even if the function is complex
"""

import jax
import jax.numpy as jnp

### Restricting the usage to the CPU:
jax.config.update("jax_default_device", jax.devices("cpu")[0])

##############################################################################################

def single_basis_func(tval, ind, delta, twidth):
    """
    tval: real non-negative number, time at which the function is evaluated
    ind: positive integer, number of the harmonic function
    delta: real number, value of detuning
    twidth: real positive number, width of the pulse

    Returns
    -------
    (real_part, imag_part): tuple containing of two real numbers corresponding
                        to the real and the imaginary part of the
    function, exp(-i*delta*tval)*sin((ind*pi/twidth)*tval) when 0<=t<=twidth
    else it returns tuple (0.0, 0.0).

    """
    ### Harmonic:
    envelope = jnp.sqrt(2.0 / twidth) * jnp.sin((ind * jnp.pi * tval) / twidth)
    real_part1 = envelope * jnp.cos(delta * tval)
    imag_part1 = -envelope * jnp.sin(delta * tval)

    ### Masking so that the harmonic is returned only
    ### for 0<=t<=twidth and zero elsewhere
    mask = (tval >= 0.0) & (tval <= twidth)
    real_part = jnp.where(mask, real_part1, 0.0)
    imag_part = jnp.where(mask, imag_part1, 0.0)

    return (real_part, imag_part)


### With the following function, the "ind" argument can be
### an array for the function "single_basis_func":
# pylint: disable=invalid-name
sine_basis_func = jax.vmap(single_basis_func, in_axes=(None, 0, None, None))


def sine_basis_complex(tval, nmax, delta, twidth):
    """
    tval: real non-negative number, time at which the function is evaluated
    nmax: positive integer, harmonics up to nmax are produced
    delta: real number, value of detuning
    twidth: real positive number, width of the pulse

    Returns
    -------
    (full_basis_real, full_basis_imag): returns two, one dimensional arrays
    consisting of the real and the imaginary part of the
    harmonics for 0< n <= nmax, exp(-i*delta*tval)*sin((ind*pi/twidth)*tval)
    for 0<=t<=twidth
    """
    ### Range of n values:
    nindices = jnp.arange(1, nmax + 1)

    ### Calling the function for range of nvalues
    (full_basis_real, full_basis_imag) = sine_basis_func(tval, nindices, delta, twidth)

    return (full_basis_real, full_basis_imag)


### With the following function, both the "ind" argument and
### "tval" argument can be arrays for the function "single_basis_func"
# pylint: disable=invalid-name
sine_basis_func_tarr = jax.vmap(sine_basis_func, in_axes=(0, None, None, None))
