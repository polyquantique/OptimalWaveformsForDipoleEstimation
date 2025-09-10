"""
This script has all the functions for computing the
global quantum Fisher information (QFI) of a Gaussian pulse using the new set
of ODEs derived in the paper
and compare with the double-sided Master equation, to optimize the QFI over the
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

### JAX supports single-precisions numbers by default.For double precision, use:
### https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html
config.update("jax_enable_x64", True)

# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments


#######################################################################################
@jit
def gauss_pulse(tval, tcap, avg_num_phot):
    """
    Parameters
    ----------
    tval: non-negative number, time at which pulse amplitude is required
    tcap: non-negative number, T in the Gaussian pulse shape(see paper)
    avg_num_phot: non-negative number corresponding to the average number of
                   photons in the pulse
         (Note: int_{-inf}^{inf} dt gauss_pulse(t, tcap, num_phot)**2 = avg_num_phot)

    Returns
    -------
    real number, which corresponds to the amplitude of the Gaussian
    function
    """
    tcen = 6 * (
        tcap / jnp.sqrt(2)
    )  # Gaussian pulse centered at 6 times the width from t=0

    return (
        jnp.sqrt(avg_num_phot)
        / (jnp.pi * tcap**2) ** (1 / 4)
        * jnp.exp(-((tval - tcen) ** 2) / (2 * tcap**2))
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
    args: tuple consisting of two real numbers, (tcap, avg_num_phot)
          corresponding to T in the Gaussian pulse shape(see paper) and
          the average number of photons in the pulse

    Returns
    -------
    tuple consisting of nine real numbers corresponding to the
    derivatives of the variables in rval: (dx_dt, dz_dt, dxi_dt,
                                          dF_dt, dxi1_dt, dxi2_dt,
                                          dFp_dt, dFz_dt, dFx_dt)
    """
    xval, zval, xival, _, xi1val, xi2val, *_ = rval

    tcap, avg_num_phot = args
    ########################################################
    ### ODES for QFI:#######################################
    ftval = gauss_pulse(tval, tcap, avg_num_phot)

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
def qfi_odes_contribs(tcap, avg_num_phot):
    """
    Parameters
    ----------
    tcap: real number corresponding to T in the Gaussian pulse shape(see paper)
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
    tvar = tcap / jnp.sqrt(2)
    tfinal = 12 * tvar + (10 / gamma_cap) * jnp.log(10)

    ### Solution of the ODEs:
    sol = diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=tfinal,
        dt0=None,
        y0=coef_ini,
        args=(tcap, avg_num_phot),
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
def qfi_odes(tcap, avg_num_phot):
    """
    Parameters
    ----------
    tcap: real number corresponding to T in the Gaussian pulse shape(see paper)
    avg_num_phot: real number corresponding to the average number of
                  photons in the pulse

    Returns
    -------
    Real numbers corresponding to the QFI: F = Fp+Fx+Fz
    """
    fpval, fzval, fxval = qfi_odes_contribs(tcap, avg_num_phot)
    return jnp.array(fpval + fzval + fxval)


####################################################################################
####################################################################################
### Solving for QFI using the double-sided Master equation (Method 2):
@jit
def coef_mat(tval, gamma1_cap, gamma2_cap, tcap, avg_num_phot):
    """
    Parameters
    ----------
    tval: non-negative number, time at which the function is evaluated
    gamma1_cap: positive real number, left side coupling strength in the
                double-sided master equation
    gamma2_cap: positive real number, right side coupling strength in the
                double-sided master equation
    tcap: real number corresponding to T in the Gaussian pulse shape(see paper)
    avg_num_phot: positive real number, average number of pump photons

    Returns
    -------
    two-dimensional array, 8 by 8 dimensional matrix.
    Matrix corresponding to the derivative. That is, for d/dt (coef(t)) = A coef(t)
    matrix A is returned by this function
    """

    # Gaussian pulse:
    ftval = gauss_pulse(tval, tcap, avg_num_phot)

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

    args: tuple consisting of five real numbers, (gamma1_cap, gamma2_cap, delta, tcap, avg_num_phot)
        gamma1_cap: positive real number, left side coupling strength in the
                double-sided master equation
        gamma2_cap: positive real number, right side coupling strength in the
                double-sided master equation
        tcap: real number corresponding to T in the Gaussian pulse shape(see paper)
        avg_num_phot: positive real number, average number of pump photons

    Returns
    -------
    two-dimensional array, 8 by 8 dimensional matrix.
    Matrix corresponding to the derivative. That is, for d/dt (coef(t)) = A coef(t)
    matrix A is returned by this function
    """
    gamma1_cap, gamma2_cap, tcap, avg_num_phot = args
    mat = coef_mat(tval, gamma1_cap, gamma2_cap, tcap, avg_num_phot)

    return jnp.dot(mat, coef)


@jit
def mu_norm(d0, d1, d2, d3, d4, d5, d6, d7):
    """
    Parameters
    ----------
    (d0,d1,d2,d3,d4,d5,d6,d7): 8 real numbers, which correspond to the solution
                              of the differential equations of the double-sided
                            Master equation
                            The coefficients in the Pauli basis are given by
                          (c0,c1,c2,c3)=(d0+1j*d4, d1+1j*d5, d2+1j*d6, d3+1j*d7)

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
def fidelity_trace(tcap, avg_num_phot, epsilon):
    """
    Parameters
    ----------
    tcap: real number corresponding to T in the Gaussian pulse shape(see paper)
    avg_num_phot: positive real number, average number of pump photons
    epsilon: real number, the parameter used in the obtaining the derivative
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

    ### Final time for the Gaussian pulse
    ### contains two parts: width of the pulse + spontaneous decay time
    tfinal = 12 * (tcap / jnp.sqrt(2)) + (10 / gamma_cap) * jnp.log(10)

    ### Note that we add adjoint=DirectAdjoint() so that the
    ### derivative of the output can be taken:
    sol = diffeqsolve(
        term,
        solver,
        t0=0,
        t1=tfinal,
        dt0=None,
        y0=coef_ini,
        args=(gamma_cap, gamma_cap + epsilon, tcap, avg_num_phot),
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
def objective_func_jax(tcap, avg_num_phot):
    """
    Parameters
    ----------
    tcap: real number corresponding to T in the Gaussian pulse shape(see paper)
    avg_num_phot: positive real number, average number of pump photons

    Returns
    -------
    -1*finfo_field: real number, minus of the fisher information(normalized by the avg_num_phot)
    This is done so that the optimizer can minimize the function (we want to maximize the QFI)
    """
    ### [0] part is taken because the optimizer wants a scalar rather an array with one value
    return -1.0 * (qfi_odes(tcap, avg_num_phot)[0])


### The following function returns the function value and the gradient with respect
### to tcap (they are not utilized in the optimizer but can be called to compute
### the gradient of the function):
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
    res.params: real number, optimum value of the tcap given by the optimizer
    -1*res.state.fun_val: positive real number, value of the QFI at res.params
    res.state.success*1.0: binary value (0.0 or 1.0), zero if the optimizer fails and one
             if the optimizer succeeds.

    """

    ### Optimizing using the Scipy optimizer with the initial seed passed to it

    solver = jaxopt.ScipyMinimize(
        fun=objective_func_jax, method="L-BFGS-B", maxiter=1000, tol=1e-6
    )
    res = solver.run(ini_seed, avg_num_phot=avg_num_phot)

    return (res.params, -1 * res.state.fun_val, res.state.success * 1.0)

#####################################################################################
from scipy.special import erfcx  # scaled erfc

def fpval_analytic(tcap):
    """
    Parameters
    ----------
    tcap: real number corresponding to T in the Gaussian pulse shape(see paper)

    Returns
    -------------
    analytic expression for Fp per unti photon with the Gaussian pulse shape

    """
    return 2.0 * jnp.sqrt(jnp.pi) * tcap * erfcx(0.5 * tcap)
