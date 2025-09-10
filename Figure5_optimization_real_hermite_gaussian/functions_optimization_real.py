"""
This script has all the functions for computing quantum Fisher information
objective function for optimizing, optimizer that uses JAX
and gradient from JAX using the real Hermite-Gaussian basis
"""


import jax
import jaxopt
import optax
import numpy as np
from jax import numpy as jnp
from jax import config, jit, random
from diffrax import diffeqsolve, ODETerm, Dopri5, PIDController, DirectAdjoint
from functions_hermite import scaled_hermite

######################################################################################
### Restricting the usage to CPU:
jax.config.update("jax_default_device", jax.devices("cpu")[0])

# # JAX supports single-precisions numbers by default.For double precision, use:
# # https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html
config.update("jax_enable_x64", True)


########################################################################################
@jit
def arb_pulse(tval, pulse_coef, tcap):
    """
    tval: 1d array of size greater than zero (it could be a scalar or an array). Contains
          the time values at which the pulse is evaluated.
    pulse_coef: 1d array containing real numbers, and it is
            normalized to avg_num_phot. That is, jnp.sum((pulse_coef)**2)=avg_num_phot.
            These are the coefficients of basis functions, {c0,...,cn-1}.
    tcap: non-negative real number, determines the width of the pulses
          (same as T in Appendix D2, see paper). The variance of the
          nth-order Hermite Gaussian is given by sqrt(n+0.5)*T.

    Returns
    -------
    pulse: 1d array (or scalar) whose size is equal to len(tval), which corresponds
           to pulse value at time=tval.(fp(t) = c1 f1(t) + c2 f2(t) + ... + cn f(t))

    """
    ### Number of basis coefficients (same as number of Hermite functions):
    ### We subtract one because of the order of Hermite functions starts from 0.
    nmax = len(pulse_coef) - 1

    ### Evaluating the basis functions at tval:
    ### The following contains basis functions up to nmax along different rows.
    basis_funcs = scaled_hermite(nmax, tval, tcap)

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
          (x(t), z(t), xi(t), (1/alpha**2)*qfi).

    args:(pulse_coefs, tcap)
        pulse_coefs: 1d array of size = nmax+1, it contains real numbers, and
                    it is normalized to avg_num_phot.Hence, jnp.sum(pulse_coef**2)=avg_num_phot.
                    These are the coefficients of the basis functions, {c1, c2, c3,....,cn}.
        tcap: non-negative real number, determines the width of the pulse.
              The variance of the nth-order Hermite Gaussian is given by sqrt(n+0.5)*T.

    Returns
    -------
    drvec_dt: one-dimensional array with length=4.
        Derivative of the state of the system, (dx/dt, dz/dt, dxi/dt, (1/alpha**2)*dqfi/dt )
        This assumes that the detuning of the system is zero.
    """
    # Define the system of ODEs
    x, z, xi, _ = rvec

    ### Pulse:
    pulse_coefs, tcap = args

    ### Zeroth component is taken to avoid array
    ftval = arb_pulse(tval, pulse_coefs, tcap)
    ftval = ftval[0]

    alpha2 = jnp.linalg.norm(pulse_coefs) ** 2

    ### Gamma value is set to 1.0:
    dx_dt = -x / 2 + 2 * z * ftval
    dz_dt = -z - 2 * x * ftval - 1
    dxi_dt = -(xi / 2) + (ftval / 2) + (x / 4)
    dqfi_dt = (0.5 / alpha2) * (1 + z) + (4 * (ftval / alpha2) * xi)

    drvec_dt = jnp.array([dx_dt, dz_dt, dxi_dt, dqfi_dt])

    return drvec_dt


@jit
def glb_finfo(pulse_coefs, tcap):
    """
    Parameters
    ----------
    pulse_coefs: 1d array of size = nmax+1, it contains real numbers, and it is
                normalized to avg_num_phot. That is, jnp.sum(pulse_coef**2)=avg_num_phot.
                These are the coefficients of basis functions, {c1, c2, c3,....,cn}.
    tcap: non-negative real number, determines the width of the pulse.
              The variance of the nth-order Hermite Gaussian is given by sqrt(n+0.5)*T.

    Returns
    -------
    real number, QFI per unit photon of the pulse
    """
    gamma_cap = 1.0  # capital gamma is set to one
    ### Initial conditions (fixed):
    coef_ini = jnp.zeros(4)
    coef_ini = coef_ini.at[1].set(-1.0)

    ### ODE solver:
    term = ODETerm(ode_system)
    solver = Dopri5()

    ### Final time for the pulse
    ### contains two parts: width of the pulse + spontaneous decay time
    nmax = len(pulse_coefs) - 1
    twidth = (jnp.sqrt(2 * nmax + 1) + 5) * tcap
    tfinal = twidth + (10.0 / gamma_cap) * jnp.log(10.0)

    ### Note that we add adjoint=DirectAdjoint() so that the
    ### derivative of the output can be taken:
    sol = diffeqsolve(
        term,
        solver,
        t0=-twidth,
        t1=tfinal,
        dt0=None,
        y0=coef_ini,
        args=(pulse_coefs, tcap),
        stepsize_controller=PIDController(rtol=1e-9, atol=1e-9), #∣error∣≤atol+rtol⋅∣y∣
        max_steps=10**6,
        adjoint=DirectAdjoint(),
    )

    return sol.ys[0, 3]


####################################################################################
#### Optimization functions:
@jit
def objective_func_jax(pulse_coefs, avg_num_phot, tcap):
    """
    Parameters
    ----------
    pulse_coefs: 1d array of size = nmax+1, it contains real numbers, and it may
                 not be normalized to avg_num_phot.
    avg_num_phot: real number, average number of photons.
    tcap: non-negative real number, determines the width of the pulse.
              The variance of the nth-order Hermite Gaussian is given by sqrt(n+0.5)*T.

    Returns
    -------
    fisher_information: real number, fisher information, which is obtained by the
                      derivative at epsilon=0. This is the objective function
                      to be minimized (-1 times the quantity we want to maximize) for
                      all the JAX optimizers (jaxopt, optax solvers)
    """

    ##### Normalizing the pulse to sqrt(avg_num_phot):
    pulse_coefs_normalized = (
        jnp.sqrt(avg_num_phot) / jnp.linalg.norm(pulse_coefs)
    ) * pulse_coefs

    ### Passing the normalized pulse for computing fisher information
    fisher_information = -1 * glb_finfo(pulse_coefs_normalized, tcap)

    return fisher_information


def sing_inst_optimum_lbfgsb(ini_seed, avg_num_phot, tcap):
    """
    Parameters
    ----------
    ini_seed: 1d array of size = nmax+1, containing real numbers. nmax is the
              number of basis functions + 1. This array serves as a random seed
              for the optimizer. Note that this seed may or may not be normalized to
              avg_num_phot.
    avg_num_phot: real number, average number of photons
    tcap: non-negative real number, determines the width of the pulse.
              The variance of the nth-order Hermite Gaussian is given by sqrt(n+0.5)*T.

    Returns
    -------
    Results of the jaxopt optimizer with'BF-BFGS-B' method for a given ini_seed:
    (optimal pulse, res.state.fun_val, res.state.success*1.0)
    where optimal pulse is the
    set of pulse coefficients that optimize the objective function,
    res.state.fun_val is the function value at optimal parameters,
    res.state.success*1.0 is 1.0 if the optimizer is successful and 0.0 otherwise
    """

    ### BFGS
    solver = jaxopt.ScipyMinimize(
        fun=objective_func_jax, method="L-BFGS-B", maxiter=5000, tol=1e-9
    )

    #################################################################
    ### Running the optimizer:
    res = solver.run(ini_seed, avg_num_phot=avg_num_phot, tcap=tcap)

    opt_qfi = -1 * res.state.fun_val

    ### Normalizing the pulse:
    pulse_norm = jnp.linalg.norm(res.params)
    optimal_pulse = jnp.sqrt(avg_num_phot) * (res.params / pulse_norm)

    opt_success = res.state.success * 1.0

    return (optimal_pulse, opt_qfi, opt_success)


### The following function gives the gradient of the objective function
# pylint: disable=invalid-name
fisher_opt_grad = jit(jax.grad(objective_func_jax, argnums=0))

### The following function returns the function value along with the gradient:
# pylint: disable=invalid-name
fisher_opt_value_grad = jit(jax.value_and_grad(objective_func_jax, argnums=0))


def mult_inst_optimum_lbfgsb(num_rseeds, avg_num_phot, num_basis_funcs, tcap):
    """
    Parameters
    ----------
    num_rseeds: number of the random seeds passed to the optimizer.
    avg_num_phot: real number, average number of photons.
    num_basis_funcs: real non-negative number, number of basis functions for the input pulse
    tcap: non-negative real number, determines the width of the pulse.
              The variance of the nth-order Hermite Gaussian is given by sqrt(n+0.5)*T.

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

    # First random seed as zeroth order Hermite Gaussian function:
    pulse_coef_normalized = pulse_coef_normalized.at[0, :].set(
        jnp.zeros(num_basis_funcs).at[0].set(jnp.sqrt(avg_num_phot))
    )

    ##### Collecting the results of the optimizer obtained for various random seeds:
    opt_pulse_coef = np.zeros(num_rseeds * num_basis_funcs)
    opt_fisher = np.zeros(num_rseeds)
    opt_grad = np.zeros(num_rseeds)

    for ind in range(num_rseeds):
        (
            opt_pulse_coef[ind * num_basis_funcs : (ind + 1) * num_basis_funcs],
            opt_fisher[ind],
            opt_grad[ind],
        ) = sing_inst_optimum_lbfgsb(pulse_coef_normalized[ind], avg_num_phot, tcap)

    return (opt_pulse_coef, opt_fisher, opt_grad)


def sing_inst_optimum_adam(ini_seed, avg_num_phot, tcap):
    """
    Parameters
    ----------
    ini_seed: 1d array containing real numbers. Size of the array equals the
              number of basis functions. This array serves as a random seed
              for the optimizer. Note that this seed may or may not be normalized to
              avg_num_phot.
    avg_num_phot: real number, average number of photons
    tcap: non-negative real number, determines the width of the pulse.
              The variance of the nth-order Hermite Gaussian is given by sqrt(n+0.5)*T.

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
        return objective_func_jax(theta, avg_num_phot=avg_num_phot, tcap=tcap)

    ### auto step size
    opt = optax.adafactor(learning_rate=None)

    solver = jaxopt.OptaxSolver(
        fun=fun,
        opt=opt,
        maxiter=10**4,
        tol=1e-5,  # this can be decreased, but very time consuming
        implicit_diff=False,  # <- big compile-time saver
    )

    # compile once per (avg_num_phot, tcap):
    run_once = jax.jit(solver.run)
    res = run_once(ini_seed)  # pylint: disable=not-callable

    pulse_norm = jnp.linalg.norm(res.params)
    opt_pulse_coef = jnp.sqrt(avg_num_phot) * (res.params / pulse_norm)

    opt_qfi = -1 * res.state.value
    opt_success = (res.state.error <= solver.tol) * 1.0

    return (opt_pulse_coef, opt_qfi, opt_success)
