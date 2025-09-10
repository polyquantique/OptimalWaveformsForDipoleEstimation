"""
This script has all the functions for computing the
global quantum Fisher information (QFI) of a square pulse using the
new set of ODEs derived in the paper
and the double-sided Master equation, to optimize the QFI over the
width of the pulse
"""

import jax
from jax import numpy as jnp
from jax import config, jit
from diffrax import diffeqsolve, ODETerm, Dopri5, PIDController, DirectAdjoint
import jaxopt

######################################################################################
### Restricting the usage to CPU:
jax.config.update("jax_default_device", jax.devices("cpu")[0])

# # JAX supports single-precisions numbers by default.For double precision, use:
# # https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html
config.update("jax_enable_x64", True)

# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments


#######################################################################################
# Define the pulse function:
@jit
def step_func(tval, var_epsilon):
    """
    Parameters
    ----------
    tval: scalar, the argument at which the function is evaluated.
    var_epsilon: scalar, determines the sharpness of the step function.
            The smaller the var_epsilon, the sharper the jump from 0 to 1.

    Returns
    -------
    scalar, the values correspond to a step function, which is defined as
    being zero for tval<0 and 1 for tval>0
    """
    return 0.5 * (1 + jnp.tanh(tval / var_epsilon))


@jit
def square_pulse_unnormalized(tval, twidth, avg_num_phot):
    """
    Parameters
    ----------
    tval: scalar, the argument at which the function is evaluated.
    twidth: positive scalar, it is the width of the square pulse
    avg_num_phot: positive real number, average number of pump photons

    Returns
    -------
    scalar, the values correspond to a square pulse, which is defined as
    being zero for t<0.25*twidth and jnp.sqrt(avg_num_phot/twidth) for t>twidth+0.25*twidth.
    Note that this square pulse is not normalized exactly to avg_num_phot.
    The norm might be off by a small factor from avg_num_phot due to
    var_epsilon being nonzero (smoothness of the square pulse)

    !!! The square pulse is shifted by 0.25*twidth, so that it is zero for t=0
    and smoothly goes to one
    """
    ### var_epsilon, determines the sharpness of the step function.
    ### The smaller the var_epsilon, the sharper the jump from 0 to 1
    ### or from 1 to 0.
    var_epsilon = 0.01 * twidth
    shift = 0.25 * twidth
    pulse_unnormalized = step_func(tval - shift, var_epsilon) - step_func(
        tval - twidth - shift, var_epsilon
    )

    return jnp.sqrt(avg_num_phot / twidth) * pulse_unnormalized


def pulse_norm(twidth, avg_phot_num):
    """
    Parameters
    ----------
    twidth: positive scalar, it is the width of the square pulse
    avg_num_phot: positive real number, average number of pump photons

    Returns
    -------
    norm: scalar, the values correspond to the norm of the pulse

    The norm is given by int_{0}^{Tpulse+0.5*Tpulse} prefac
    Because of smoothness of the square pulse, the norm is not prefac*twidth

    !!! The square pulse is shifted by 0.25*Tpulse, so that it is zero for t=0
    and smoothly goes to one
    """
    arr_len = 10**5  ### num of data points to provide to the integrator

    ##Estimate of the maximum range over which the function is nonzero
    tvals = jnp.linspace(0, 1.5 * twidth, arr_len)

    ## Integrating the function
    norm = jnp.trapezoid(
        square_pulse_unnormalized(tvals, twidth, avg_phot_num) ** 2, tvals
    )

    return norm


@jit
def square_pulse(tval, twidth, avg_phot_num, norm_value):
    """
    Parameters
    ----------
    tval: non-negative number, time at which the pulse amplitude is needed
    twidth: real number corresponding to the width of the square pulse
    avg_num_phot: real number corresponding to the average number of
                    photons in the pulse
    norm_value: positive real number, it is the norm of the smooth square pulse

    Returns
    -------
    pulse: pulse is a scalar, the values correspond to prefac,
    where prefac is chosen such that
    the L2 norm of the function is avg_num_phot.

    !!! The square pulse is shifted by 0.25*Tpulse, so that it is zero for t=0
    and smoothly goes to one
    """
    pulse = jnp.sqrt(avg_phot_num / norm_value) * square_pulse_unnormalized(
        tval, twidth, avg_phot_num
    )

    return pulse


### Discontinuous square wave:
@jit
def square_pulse_discont(tval, twidth, avg_num_phot):
    """
    Parameters
    ----------
    tval: non-negative number, time at which the pulse amplitude is needed
    twidth: real number corresponding to the width of the square pulse
    avg_num_phot: real number corresponding to the average number of
                  photons in the pulse
    """
    return jnp.sqrt(avg_num_phot / twidth) * jnp.where(
        (tval >= 0) & (tval < twidth), 1.0, 0.0
    )


########################################################################################
########################################################################################
### Solving for QFI using the ODEs derived in the paper (Method 1):
@jit
def ode_system(tval, rval, args):
    """
    Parameters
    ----------
    tval: non-negative number, time at which the derivative is needed
    rval: tuple consisting of nine real numbers, (x,z,xi,F,xi1,xi2,Fp,Fz,Fx)
          Note that xi1 and xi2 are additional variables introduced to
          obtain the contributions QFI: Fp and Fx (see paper)
    args: tuple consisting of three real numbers, (twidth, avg_num_phot, norm_value)
          where twidth corresponds to the width of the square pulse,
          avg_num_phot is the average number of photons in the pulse,
          norm_value is the norm of the smooth square pulse

    Returns
    -------
    tuple consisting of nine real numbers corresponding to the
    derivatives of the variables in rval: (dx_dt, dz_dt, dxi_dt,
                                          df_dt, dxi1_dt, dxi2_dt,
                                          dfp_dt, dfz_dt, dfx_dt)
    """
    xval, zval, xival, _, xi1val, xi2val, *_ = rval

    twidth, avg_num_phot, norm_value = args
    ########################################################
    ### ODES for QFI:#######################################
    ftval = square_pulse(tval, twidth, avg_num_phot, norm_value)

    dxdt = -xval / 2 + 2 * zval * ftval
    dzdt = -zval - 2 * xval * ftval - 1
    dxidt = -xival / 2 + ftval / 2 + xval / 4
    dfdt = 0.5 * (1 + zval) + 4 * ftval * xival
    #########################################################
    #########################################################
    ### ODEs to identify the contributions from Fp, Fz and Fx
    ### Note that dxidt = dxi1dt + dxi2dt:
    dxi1dt = -xi1val / 2 + ftval / 2
    dxi2dt = -xi2val / 2 + xval / 4

    fpval = 4 * ftval * xi1val
    fzval = 0.5 * (1 + zval)
    fxval = 4 * ftval * xi2val
    #########################################################

    state_final = [dxdt, dzdt, dxidt, dfdt, dxi1dt, dxi2dt, fpval, fzval, fxval]

    return jnp.array(state_final)


@jit
def qfi_odes_contribs(twidth, avg_num_phot):
    """
    Parameters
    ----------
    twidth: real number corresponding to the width of the square pulse
    avg_num_phot: real number corresponding to the average number of
                  photons in the pulse

    Returns
    -------
    tuple consisting of three real numbers corresponding to the
    three different contributions of the QFI: (Fp, Fz, Fx)
    where F = Fp + Fz + Fx
    """
    gamma_cap = 1.0  # coupling parameter set to one

    ### Initial conditions:
    coef_ini = jnp.zeros(9)
    coef_ini = coef_ini.at[1].set(-1.0)  ### Setting zvalue to be -1 at t=0

    ### ODE solver:
    term = ODETerm(ode_system)
    solver = Dopri5()

    ### final time of the evolution: 12*variance + spontaneous decay of atom
    tfinal = twidth + (10 / gamma_cap) * jnp.log(10)
    norm_value = pulse_norm(twidth, avg_num_phot)

    ### Solution of the ODEs:
    sol = diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=tfinal,
        dt0=None,
        y0=coef_ini,
        args=(twidth, avg_num_phot, norm_value),
        stepsize_controller=PIDController(rtol=1e-14, atol=1e-14),
        max_steps=10**6,
    )

    qfi_contributions = [
        sol.ys[:, 6] / avg_num_phot,
        sol.ys[:, 7] / avg_num_phot,
        sol.ys[:, 8] / avg_num_phot,
    ]

    return qfi_contributions


@jit
def qfi_odes(twidth, avg_num_phot):
    """
    Parameters
    ----------
    twidth: real number corresponding to the width of the square pulse
    avg_num_phot: real number corresponding to the average number of
                  photons in the pulse

    Returns
    -------
    Real numbers corresponding to the QFI: F = Fp+Fx+Fz
    """
    fpval, fzval, fxval = qfi_odes_contribs(twidth, avg_num_phot)
    return jnp.array(fpval + fzval + fxval)


####################################################################################
####################################################################################
### Solving for QFI using the double-sided Master equation (Method 2):
@jit
def coef_mat(tval, gamma1_cap, gamma2_cap, twidth, avg_num_phot, norm_value):
    """
    Parameters
    ----------
    tval: non-negative number, time at which the function is evaluated
    gamma1_cap: positive real number, left side coupling strength in the
               double-sided master equation
    gamma2_cap: positive real number, right side coupling strength in the
                double-sided master equation
    twidth: real number corresponding to the width of the square pulse
    avg_num_phot: positive real number, average number of pump photons
    norm_value: positive real number, norm of the smooth square pulse

    Returns
    -------
    two-dimensional array, 8 by 8 dimensional matrix.
    Matrix corresponding to the derivative. That is, for d/dt (coef(t)) = A coef(t)
    matrix A is returned by this function
    """

    # Square pulse:
    ftval = square_pulse(tval, twidth, avg_num_phot, norm_value)

    detuning = 0.0  # value of the detuning

    sqg1 = jnp.sqrt(gamma1_cap)
    sqg2 = jnp.sqrt(gamma2_cap)

    mat = jnp.array(
        [
            [
                -0.25 * gamma1_cap + 0.5 * sqg1 * sqg2 - 0.25 * gamma2_cap,
                0,
                0,
                -0.25 * gamma1_cap + 0.5 * sqg1 * sqg2 - 0.25 * gamma2_cap,
                0,
                0,
                sqg1 * ftval - sqg2 * ftval,
                0,
            ],
            [
                0,
                -0.25 * gamma1_cap - 0.25 * gamma2_cap,
                -detuning,
                sqg1 * ftval + sqg2 * ftval,
                0,
                0,
                -0.25 * gamma1_cap + 0.25 * gamma2_cap,
                0,
            ],
            [
                0,
                detuning,
                -0.25 * gamma1_cap - 0.25 * gamma2_cap,
                0,
                sqg1 * ftval - sqg2 * ftval,
                0.25 * gamma1_cap - 0.25 * gamma2_cap,
                0,
                0,
            ],
            [
                -0.25 * gamma1_cap - 0.5 * sqg1 * sqg2 - 0.25 * gamma2_cap,
                -sqg1 * ftval - sqg2 * ftval,
                0,
                -0.25 * gamma1_cap - 0.5 * sqg1 * sqg2 - 0.25 * gamma2_cap,
                0,
                0,
                0,
                0,
            ],
            [
                0,
                0,
                -sqg1 * ftval + sqg2 * ftval,
                0,
                -0.25 * gamma1_cap + 0.5 * sqg1 * sqg2 - 0.25 * gamma2_cap,
                0,
                0,
                -0.25 * gamma1_cap + 0.5 * sqg1 * sqg2 - 0.25 * gamma2_cap,
            ],
            [
                0,
                0,
                0.25 * gamma1_cap - 0.25 * gamma2_cap,
                0,
                0,
                -0.25 * gamma1_cap - 0.25 * gamma2_cap,
                -detuning,
                sqg1 * ftval + sqg2 * ftval,
            ],
            [
                -sqg1 * ftval + sqg2 * ftval,
                -0.25 * gamma1_cap + 0.25 * gamma2_cap,
                0,
                0,
                0,
                detuning,
                -0.25 * gamma1_cap - 0.25 * gamma2_cap,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                -0.25 * gamma1_cap - 0.5 * sqg1 * sqg2 - 0.25 * gamma2_cap,
                -sqg1 * ftval - sqg2 * ftval,
                0,
                -0.25 * gamma1_cap - 0.5 * sqg1 * sqg2 - 0.25 * gamma2_cap,
            ],
        ]
    )
    return mat


### ODE solver:
@jit
def dc_dt(tval, coef, args):
    """
    Parameters
    ----------
    tval: non-negative number, time at which the function is evaluated
    coef: tuple of eight real numbers, (d0,d1,d2,d3,d4,d5,d6,d7)
          which describe the generalized density operator.
          The coefficients in the Pauli basis are given by
          (c0,c1,c2,c3)=(d0+1j*d4, d1+1j*d5, d2+1j*d6, d3+1j*d7)

    args: tuple consisting of six real numbers, (gamma1_cap, gamma2_cap,
                                      delta, tcap, avg_num_phot, norm_value)
        gamma1_cap: positive real number, left side coupling strength in the
                                                double-sided master equation
        gamma2_cap: positive real number, right side coupling strength in the
                                               double-sided master equation
        twidth: real number corresponding to the width of the square pulse
        avg_num_phot: positive real number, average number of pump photons
        norm_value: positive real number, norm of the smooth square pulse

    Returns
    -------
    two-dimensional array, 8 by 8 dimensional matrix.
    Matrix corresponding to the derivative. That is, for d/dt (coef(t)) = A coef(t)
    matrix A is returned by this function
    """
    gamma1_cap, gamma2_cap, tcap, avg_num_phot, norm_value = args
    mat = coef_mat(tval, gamma1_cap, gamma2_cap, tcap, avg_num_phot, norm_value)

    return jnp.dot(mat, coef)


@jit
def mu_norm(d0, d1, d2, d3, d4, d5, d6, d7):
    """
    Parameters
    ----------
    (d0val,d1val,d2val,d3val,d4val,d5val,d6val,d7val): 8 real numbers,
                which correspond to the solution
                of the differential equations of the double-sided
                master equation. The coefficients in the Pauli basis are given by
                (c0,c1,c2,c3)=(d0val+1j*d4val, d1val1j*d5val,
                              d2val+1j*d6val, d3val+1j*d7val)

    Returns
    -------
    returns real number, which is the trace norm of the generlaized density operator
    """

    ### coefficients of mu in the Pauli basis (complex numbers)
    (c0, c1, c2, c3) = (d0 + 1j * d4, d1 + 1j * d5, d2 + 1j * d6, d3 + 1j * d7)

    ### mu matrix:
    mat = 0.5 * jnp.array([[c0 + c3, c1 - 1j * c2], [c1 + 1j * c2, c0 - c3]])

    trace_norm = jnp.linalg.norm(mat, ord="nuc")  # Nuclear norm (trace norm)

    return trace_norm


### The following function allows evaluation of mu_norm for vector input:
# pylint: disable=invalid-name
mu_norm_vec = jit(jax.vmap(mu_norm))


@jit
def fidelity_trace(twidth, avg_num_phot, xival):
    """
    Parameters
    ----------
    twidth: real number corresponding to the width of the square pulse
    avg_num_phot: positive real number, average number of pump photons
    xival: real number, the parameter used in the obtaining the derivative
             of the Fisher information

    Returns
    -------
    fid_tr: real number, fidelity trace (normalized by avg photon number)
    !!! We multiply by -4.0 so that the derivative of this function gives Fisher information
    """
    gamma_cap = 1.0  # coupling parameter set to one
    ### Initial conditions (fixed):
    coef_ini = jnp.zeros(8)
    coef_ini = coef_ini.at[0].set(1.0)
    coef_ini = coef_ini.at[3].set(-1.0)

    ### ODE solver:
    term = ODETerm(dc_dt)
    solver = Dopri5()

    ### Final time for the square pulse
    ### contains two parts: width of the pulse + spontaneous decay time
    tfinal = twidth + (10 / gamma_cap) * jnp.log(10)
    norm_value = pulse_norm(twidth, avg_num_phot)

    ### Note that we add adjoint=DirectAdjoint() so that the
    ### derivative of the output can be taken:
    sol = diffeqsolve(
        term,
        solver,
        t0=0,
        t1=tfinal,
        dt0=None,
        y0=coef_ini,
        args=(gamma_cap, gamma_cap + xival, twidth, avg_num_phot, norm_value),
        stepsize_controller=PIDController(rtol=1e-14, atol=1e-14),
        max_steps=10**10,
        adjoint=DirectAdjoint(),
    )

    ### Fisher information of the emission field:
    fid_tr = mu_norm(
        sol.ys[0][0],
        sol.ys[0][1],
        sol.ys[0][2],
        sol.ys[0][3],
        sol.ys[0][4],
        sol.ys[0][5],
        sol.ys[0][6],
        sol.ys[0][7],
    )

    return -4 * (fid_tr / avg_num_phot)


### Double derivative of the fidelity trace gives the QFI:
# pylint: disable=invalid-name
qfi_twosided = jit(jax.jacfwd(jax.jacfwd(fidelity_trace, argnums=2), argnums=2))


####################################################################################
#### Optimization functions:
####################################################################################
@jit
def objective_func_jax(twidth, avg_num_phot):
    """
    Parameters
    ----------
    twidth: real number corresponding to the width of the square pulse
    avg_num_phot: positive real number, average number of pump photons

    Returns
    -------
    -1*finfo_field: real number, minus of the fisher information(normalized by the avg_num_phot)
    This is done so that the optimizer can minimize the function (we want to maximize the QFI)
    """
    ### [0] part is taken because the optimizer wants a scalar rather an array with one value
    return -1.0 * (qfi_odes(twidth, avg_num_phot)[0])


### The following function returns the function value and the gradient with respect
### to twidth (they are not utilized in the optimizer but can be called to compute
### the gradient of the function)
# pylint: disable=invalid-name
fisher_opt_value_grad = jit(jax.value_and_grad(objective_func_jax, argnums=0))
fisher_opt_grad = jit(jax.grad(objective_func_jax, argnums=0))


def sing_inst_qfi_optimizer(ini_seed, avg_num_phot):
    """
    Parameters
    ----------
    ini_seed: initial seed for the optimizer
    avg_num_phot: positive real number, average number of pump photons

    Returns
    -------------
    (res.params, -1*res.state.fun_val, res.state.success*1.0) where
    res.params: real number, optimum value of the twidth given by the optimizer
    -1*res.state.fun_val: positive real number, value of the QFI at res.params
    res.state.success*1.0: binary value (0.0 or 1.0), zero if the optimizer fails and one
             if the optimizer succeeds.

    """

    ### Optimizing using the Scipy optimizer with the initial seed passed to it

    solver = jaxopt.ScipyMinimize(
        fun=objective_func_jax, method="L-BFGS-B", maxiter=50000, tol=1e-6
    )
    res = solver.run(ini_seed, avg_num_phot=avg_num_phot)

    return (res.params, -1 * res.state.fun_val, res.state.success * 1.0)


###############################################################################
###############################################################################
### Analytic expressions for zero detuning case:


def qfi_analytic(tval, twidth, alpha):
    """
    Parameters
    ----------
    tval: non-negative number, time value at which the function is evaluated
    twidth: real number corresponding to the width of the square pulse
    alpha: real number, alpha**2 = average number of photons in the pulse

    Returns
    -------
    scalar, QFI of the square pulse per unit photon
           for a given tval > twidth assuming gamma_cap =1.0

    """
    ### Since the square pulse is shifted by 0.25*twidth:
    tval = tval - 0.25 * twidth
    ###############################################################################################
    ### Trignometric expression assuming 64*(alpha**2/twidth)-1 > 0:

    expr_trig = (
        4
        * (jnp.exp(-tval - 0.75 * twidth))
        * (jnp.exp(tval) - (jnp.exp(twidth)))
        * (alpha**2)
        * ((-1 + (64 * (alpha**2)) / twidth) ** -0.5)
        * (
            (jnp.exp(0.75 * twidth)) * (jnp.sqrt(-1 + (64 * (alpha**2)) / twidth))
            - (
                (jnp.sqrt(-1 + (64 * (alpha**2)) / twidth))
                * jnp.cos(0.25 * twidth * (jnp.sqrt(-1 + (64 * (alpha**2)) / twidth)))
            )
            - 3 * jnp.sin(0.25 * twidth * (jnp.sqrt(-1 + (64 * (alpha**2)) / twidth)))
        )
    ) / (twidth + 8 * (alpha**2)) + (
        twidth
        * (
            -64 * (twidth**-2) * (alpha**4)
            + (32 * (alpha**4)) / twidth
            + (12 * (alpha**2)) / (twidth + 8 * (alpha**2))
            + (jnp.exp(-0.5 * twidth)) * (-1 + 64 * (twidth**-2) * (alpha**4))
            + (
                (jnp.exp(-0.75 * twidth))
                * twidth
                * (
                    (1 + (-4 * (alpha**2)) / twidth)
                    * jnp.cos(
                        0.25 * twidth * (jnp.sqrt(-1 + (64 * (alpha**2)) / twidth))
                    )
                    + (1 + (-28 * (alpha**2)) / twidth)
                    * ((-1 + (64 * (alpha**2)) / twidth) ** -0.5)
                    * jnp.sin(
                        0.25 * twidth * (jnp.sqrt(-1 + (64 * (alpha**2)) / twidth))
                    )
                )
            )
            / (twidth + 8 * (alpha**2))
        )
    ) / (
        twidth + 8 * (alpha**2)
    )

    ###############################################################################################
    ### Hyperbolic expression assuming 64*(alpha**2/twidth)-1 < 0:

    expr_hyper = (
        4
        * jnp.exp(-tval - (3 * twidth) / 4)
        * (jnp.exp(tval) - jnp.exp(twidth))
        * (alpha**2)
        * (
            jnp.exp(3 * twidth / 4) * jnp.sqrt(1 - (64 * (alpha**2)) / twidth)
            - jnp.sqrt(1 - (64 * (alpha**2)) / twidth)
            * jnp.cosh(0.25 * jnp.sqrt(1 - (64 * (alpha**2)) / twidth) * twidth)
            - 3 * jnp.sinh(0.25 * jnp.sqrt(1 - (64 * (alpha**2)) / twidth) * twidth)
        )
    ) / (jnp.sqrt(1 - (64 * (alpha**2)) / twidth) * (8 * (alpha**2) + twidth)) + (
        twidth / (8 * (alpha**2) + twidth)
    ) * (
        jnp.exp(-tval / 2) * ((64 * (alpha**4)) / twidth**2 - 1)
        - (64 * (alpha**4)) / twidth**2
        + (32 * (alpha**4) * tval) / twidth**2
        + (12 * (alpha**2)) / (8 * (alpha**2) + twidth)
        + (jnp.exp(-3 * tval / 4) * twidth)
        / (8 * (alpha**2) + twidth)
        * (
            (1 - (4 * (alpha**2)) / twidth)
            * jnp.cosh(0.25 * tval * jnp.sqrt(1 - (64 * (alpha**2)) / twidth))
            + (1 - (28 * (alpha**2)) / twidth)
            * jnp.sinh(0.25 * tval * jnp.sqrt(1 - (64 * (alpha**2)) / twidth))
            / jnp.sqrt(1 - (64 * (alpha**2)) / twidth)
        )
    )

    ###############################################################################################
    ### When the sqrroot is negative, we want hyperbolic expression
    ### else trignometric expression:
    sqrroot = 64.0 * (alpha**2 / twidth) - 1.0
    use_trig = sqrroot > 0.0

    # uses trig expression when sqroot is positive else hyperbolic expression:
    expr = jnp.where(use_trig, expr_trig, expr_hyper)

    return expr / alpha**2

@jit
def fpval_analytic(twidth):
    """
    Parameters
    ----------
    twidth: real number corresponding to T (=width) in the square pulse shape(see paper)

    Returns
    -------------
    analytic expression for Fp per unti photon with the square pulse shape

    """

    return 4.0 * (1.0 - ((2.0 /twidth) * (1.0 - jnp.exp(-0.5 * twidth))))
