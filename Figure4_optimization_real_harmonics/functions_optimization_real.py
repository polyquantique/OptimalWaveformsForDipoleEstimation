"""
This script has all the functions for computing quantum Fisher information
objective function for optimizing, optimizer that uses JAX
and gradient from JAX using the real harmonics basis
"""

import jax
import numpy as np
from jax import numpy as jnp
from jax import config, jit, random
from diffrax import diffeqsolve, ODETerm, Dopri5, PIDController, DirectAdjoint
import jaxopt
import optax
from functions_sines import sine_basis

######################################################################################
### Restricting the usage to CPU:
jax.config.update("jax_default_device", jax.devices("cpu")[0])

# # JAX supports single-precisions numbers by default.For double precision, use:
# # https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html
config.update("jax_enable_x64", True)


########################################################################################
@jit
def arb_pulse(tval, pulse_coef, twidth):
    """
    tval: positive real number (or an array). The time value(s) at which the pulse is evaluated
    pulse_coef: 1d array containing real numbers, and it is
            normalized to avg_num_phot. That is, jnp.sum((pulse_coef)**2)=avg_num_phot.
            These are the coefficients of basis functions, {c0,...,cn-1}
    twidth: non-negative number, maximum width of the pulse

    Returns
    -------
    pulse: real number(s) which corresponds to the pulse value at time=tval(s)
            (f(tval) = c1 f1(tval) + c2 f2(tval) + ... + cn fn(tval)
            where the basis functions are given by (f1, f2, ..., fn))
    """
    ### Number of basis coefficients (same as number of sine functions/harmonics):
    nmax = len(pulse_coef)

    ### Evaluating the basis functions at tval:
    ### The following contains basis functions up to nmax along different rows
    basis_funcs = sine_basis(nmax, tval, twidth)

    ### pulse: sum_{n} pulse_coef[n]*basis_funcs[n,:]
    pulse = jnp.dot(pulse_coef, basis_funcs)

    return pulse


### ODE solver:
@jit
def ode_system(tval, rvec, args):
    """
    Parameters
    ----------
    tval: non-negative number, time at which the function is evaluated
    rvec: one dimensional array with len =4, describes the state of the system
          (x(t), z(t), xi(t), (1/alpha**2)*qfi)
    args:
        (pulse_coef, twidth): where pulse_coef is a 1d array of size = nmax+1, it
                    contains real numbers, and it is normalized to avg_num_phot.
                    That is, jnp.sum(pulse_coef**2)=avg_num_phot.
                    These are the coefficients of basis functions, {c1, c2, c3,....,cn}.
                    twidth is non-negative number, maximum width of the pulse

    Returns
    -------
    drval_dt: one-dimensional array with length=4.
    Derivative of the state of the system, (dx/dt, dz/dt, dxi/dt, (1/alpha**2)*dqfi/dt )
    This assumes that the detuning of the system is zero.

    """
    # Define the system of ODEs
    x, z, xi, _ = rvec

    ### Pulse:
    pulse_coefs, twidth = args
    ftval = arb_pulse(tval, pulse_coefs, twidth)
    alpha2 = jnp.linalg.norm(pulse_coefs) ** 2

    ### Gamma value is set to 1.0:
    dxdt = -x / 2 + 2 * z * ftval
    dzdt = -z - 2 * x * ftval - 1
    dxidt = -(xi / 2) + (ftval / 2) + (x / 4)
    dqfidt = (0.5 / alpha2) * (1 + z) + (4 * (ftval / alpha2) * xi)

    drvec_dt = jnp.array([dxdt, dzdt, dxidt, dqfidt])

    return drvec_dt


@jit
def glb_finfo(pulse_coefs, twidth):
    """
    Parameters
    ----------
    pulse_coefs: 1d array of size = nmax+1, it contains real numbers, and it is
                normalized to avg_num_phot. That is, jnp.sum(pulse_coef**2)=avg_num_phot.
                These are the coefficients of basis functions, {c1, c2, c3,....,cn}.
    twidth: non-negative number, maximum width of the pulse

    Returns
    -------
    real number, QFI per unit photon of the pulse
    """
    gamma_cap = 1.0

    ### Initial conditions (fixed):
    coef_ini = jnp.zeros(4)
    coef_ini = coef_ini.at[1].set(-1.0)

    ### ODE solver:
    term = ODETerm(ode_system)
    solver = Dopri5()

    ### Final time for the pulse
    ### contains two parts: width of the pulse + spontaneous decay time
    tfinal = twidth + (10.0 / gamma_cap) * jnp.log(10.0)

    ### Note that we add adjoint=DirectAdjoint() so that the
    ### derivative of the output can be taken:
    sol = diffeqsolve(
        term,
        solver,
        t0=0,
        t1=tfinal,
        dt0=None,
        y0=coef_ini,
        args=(pulse_coefs, twidth),
        stepsize_controller=PIDController(rtol=1e-9, atol=1e-9),
        max_steps=10**6,
        adjoint=DirectAdjoint(),
    )

    return sol.ys[0, 3]


####################################################################################
#### Optimization functions:
####################################################################################
@jit
def objective_func_jax(pulse_coefs, avg_num_phot, twidth):
    """
    Parameters
    ----------
    pulse_coef: 1d array of size = nmax+1, it contains real numbers, and it may
                 not be normalized to avg_num_phot
    avg_num_phot: real number, average number of photons
    twidth: non-negative number, maximum width of the pulse

    Returns
    -------
    qfi: negative real number, -1*quantum Fisher information per unit photon
                of the pulse

    """

    ##### Normalizing the pulse to sqrt(avg_num_phot):
    pulse_coefs_normalized = (
        jnp.sqrt(avg_num_phot) / jnp.linalg.norm(pulse_coefs)
    ) * pulse_coefs

    ### Passing the normalized pulse for computing fisher information
    qfi = -1 * glb_finfo(pulse_coefs_normalized, twidth)

    return qfi


def sing_inst_optimum_lbfgsb(ini_seed, avg_num_phot, twidth):
    """
    Parameters
    ----------
    ini_seed: 1d array containing real numbers. Size of the array equals the
              number of basis functions. This array serves as a random seed
              for the optimizer. Note that this seed may or may not be normalized to
              avg_num_phot.
    avg_num_phot: real number, average number of photons
    twidth: non-negative number, maximum width of the pulse

    Returns
    -------
    Results of the jaxopt optimizer for a given ini_seed:
        (opt_pulse_coef, opt_qfi, opt_success)
        where opt_pulse_coef is the set of pulse coefficients that
        optimize the objective function,
        opt_qfi is the QFI value at optimal parameters,
        opt_success is 1.0 for successful optimization otherwise 0.0
    """

    ### BFGS
    solver = jaxopt.ScipyMinimize(
        fun=objective_func_jax, method="L-BFGS-B", maxiter=10**4, tol=1e-9
    )

    ### Running the optimizer:
    res = solver.run(ini_seed, avg_num_phot=avg_num_phot, twidth=twidth)

    opt_qfi = -1*res.state.fun_val

    ### Normalizing the pulse:
    pulse_norm = jnp.linalg.norm(res.params)
    opt_pulse_coef = jnp.sqrt(avg_num_phot) * (res.params / pulse_norm)

    opt_success = res.state.success * 1.0

    return (opt_pulse_coef, opt_qfi, opt_success)


### The following function gives the gradient of the objective function
# pylint: disable=invalid-name
fisher_opt_grad = jit(jax.grad(objective_func_jax, argnums=0))

### The following function returns the function value along with the gradient:
# pylint: disable=invalid-name
fisher_opt_value_grad = jit(jax.value_and_grad(objective_func_jax, argnums=0))


def multi_inst_optimum_lbfgsb(num_rseeds, avg_num_phot, num_basis_funcs, twidth):
    """
    Parameters
    ----------
    num_rseeds: positive integer, number of the random seeds passed to the optimizer
    avg_num_phot: positive real number, average number of photons
    num_basis_funcs: positive integer, number of basis functions for the input pulse
    twidth: non-negative number, maximum width of the pulse

    Returns
    -------
    Results of the optimizer for a given number of random seeds:
    (opt_pulse_coef, opt_fisher, opt_success) ----> optimal pulses (stored along rows),
    optimal Fisher information and their successes correspondingly for a number of random seeds.
    """
    #### Obtaining random seeds:########################################################
    #### Key for random Seeds:
    key = random.PRNGKey(4391)
    ### Picking random pulse coefficients between 1.0 and 10.0
    pulse_coef = random.uniform(
        key, shape=(num_rseeds, num_basis_funcs), minval=0.0, maxval=1.0
    )
    row_sums = jnp.linalg.norm(pulse_coef, axis=1, keepdims=True)
    ### Following step not necessaay, but we normalize the random seeds
    pulse_coef_normalized = jnp.sqrt(avg_num_phot) * (pulse_coef / row_sums)

    # First random seed as the optimal pulse in the long width limit: sin((pi*t)/(2*twidth))
    ind_opt = int(twidth / (2 * jnp.pi))

    pulse_coef_normalized = pulse_coef_normalized.at[0, :].set(
        jnp.zeros(num_basis_funcs).at[ind_opt].set(jnp.sqrt(avg_num_phot))
    )

    ##### Collecting the results of the optimizer obtained for various random seeds:#######
    opt_pulse_coef = np.zeros(num_rseeds * num_basis_funcs)
    opt_qfi = np.zeros(num_rseeds)
    opt_success = np.zeros(num_rseeds)

    for ind in range(num_rseeds):
        (
            opt_pulse_coef[ind * num_basis_funcs : (ind + 1) * num_basis_funcs],
            opt_qfi[ind],
            opt_success[ind],
        ) = sing_inst_optimum_lbfgsb(pulse_coef_normalized[ind], avg_num_phot, twidth)

    return (opt_pulse_coef, opt_qfi, opt_success)


def sing_inst_optimum_adam(ini_seed, avg_num_phot, twidth):
    """
    Parameters
    ----------
    ini_seed: 1d array containing real numbers. Size of the array equals the
              number of basis functions. This array serves as a random seed
              for the optimizer. Note that this seed may or may not be normalized to
              avg_num_phot.
    avg_num_phot: real number, average number of photons
    twidth: non-negative number, maximum width of the pulse

    Returns
    -------
    Results of the jaxopt Adam optimizer for a given ini_seed:
        (opt_pulse_coef, opt_qfi, opt_success)
        where opt_pulse_coef is the set of pulse coefficients that
        optimize the objective function,
        opt_qfi is the QFI value at optimal parameters,
        opt_success is 1.0 for successful optimization otherwise 0.0

    !!! Takes more time than the other optimizer
    """

    ### jitting the objective function:
    @jit
    def fun(theta):
        return objective_func_jax(theta, avg_num_phot=avg_num_phot, twidth=twidth)

    ### auto step size
    opt = optax.adafactor(learning_rate=None)

    solver = jaxopt.OptaxSolver(
        fun=fun,
        opt=opt,
        maxiter=10**4,
        tol=1e-5,  # this can be decreased, but very time consuming
        implicit_diff=False,  # <- big compile-time saver
    )

    # compile once per (avg_num_phot, twidth):
    run_once = jax.jit(solver.run)
    res = run_once(ini_seed) # pylint: disable=not-callable

    pulse_norm = jnp.linalg.norm(res.params)
    opt_pulse_coef = jnp.sqrt(avg_num_phot) * (res.params / pulse_norm)

    opt_qfi = -1*res.state.value
    opt_success = (res.state.error <= solver.tol) * 1.0

    return (opt_pulse_coef, opt_qfi, opt_success)
