"""
This script has all the functions for computing quantum Fisher information
of a complex pulse, objective function for optimizing, optimizer that uses Jaxopt.
It is primarily written for the optimization of QFI
in the following basis: exp(-i detuning t)*sin((n*pi/T)*t)
"""

import jax
import numpy as np
from jax import numpy as jnp
from jax import config, jit, random
import jaxopt
from diffrax import diffeqsolve, ODETerm, Dopri5, PIDController, DirectAdjoint
from functions_sines_complex import sine_basis_complex

### Restricting the usage to CPU:
jax.config.update("jax_default_device", jax.devices("cpu")[0])

### JAX supports single-precisions numbers by default.For double precision, use:
### https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html
config.update("jax_enable_x64", True)


###########################################################################################
@jit
def arb_pulse_complex(tval, pulse_coef, detuning, twidth):
    """
    tval: real number. The time value at which the pulse is evaluated.
    pulse_coef: 1d (even-sized) array containing real numbers, and it is
            normalized to avg_num_phot. That is, jnp.sum((pulse_coef)**2)=avg_num_phot.
            These are the coefficients of basis functions, {d1,...,dn, g1,...., gn}.
            where each coefficient is given by ci = di + 1j* gi and
            f(t) = c1 f1(t) + c2 f2(t) + ... + cn fn(t) where {fi|i=1,2,...n} are the basis funcs
    detuning: real number, value of detuning
    twidth: real non-negative number. Width of the pulse.

    Returns
    -------
    (real_part, imag_part): The real and the imaginary part of the function is outputted,
      f(t) = c1 f1(t) + c2 f2(t) + ... + cn fn(t) where {fi|i=1,2,...n} are the basis funcs
      The output consists of two scalars which correspond to
      real and imaginary part of the pulse at time=tval.

    """
    ### Number of basis coefficients (twice the number of harmonic functions):
    ### nmax is half of pulse_coef because we have a complex basis: half of them
    ### correspond to the real components and the other half correpond to the imaginary
    ### components
    nmax = int(len(pulse_coef) / 2)

    ### Evaluating the basis functions at tval:
    ### The following contains basis functions up to nmax along the vector.
    (basis_funcs_real, basis_funcs_imag) = sine_basis_complex(
        tval, nmax, detuning, twidth
    )
    basis_funcs = basis_funcs_real + 1j * basis_funcs_imag

    ### Separating out the real and the imaginary coefficients:
    coef_real = pulse_coef[0:nmax]
    coef_imag = pulse_coef[nmax:]
    coef = coef_real + 1j * coef_imag

    ### pulse: sum_{n} pulse_coef[n]*basis_funcs[n,:]
    pulse = jnp.dot(basis_funcs, coef)

    ### Real and the imaginary part of the function:
    (real_part, imag_part) = (jnp.real(pulse), jnp.imag(pulse))

    return (real_part, imag_part)


@jit
def ode_system(tval, rvec, args):
    """
    Parameters
    ----------
    tval: non-negative number, time at which the function is evaluated
    rvec: one dimensional array with len = 8, describes the state of the system
        x1, y1, z1, w1, x2, y2, z2, qfi/avg_num_phot = rvec, where qfi is the
        global Fisher information
    args:(pulse_coefs, detuning, twidth)
        pulse_coefs: 1d (even-sized) array containing real numbers, and it is
                    normalized to avg_num_phot.
                    That is, jnp.sum((pulse_coef)**2)=avg_num_phot.
                    These are the coefficients of basis functions, {d1,...,dn, g1,...., gn}
                    where each coefficient is given by ci = di + 1j* gi and
                    f(t) = c1 f1(t) + c2 f2(t) + ... + cn fn(t) where {fi|i=1,2,...n}
                    are the basis funcs.
        detuning: real number, detuning
        twidth: positive real number, width of the pulse

    Returns
    -------
    drvec_dt: one-dimensional array with length=8.
                Derivative of the state of the system,
                d(rvec)/dt = (dx1/dt, dy1/dt, dz1/dt,
                              dw1/dt, dx2/dt, dy2/dt, dz2/dt,
                              (1/avg_num_phot)*dqfi/dt)
    """
    pulse_coefs, detuning, twidth = args

    # Define the system of ODEs
    (x1, y1, *_) = rvec

    ### Pulse:
    (ftr, fti) = arb_pulse_complex(tval, pulse_coefs, detuning, twidth)
    avg_num_phot = jnp.linalg.norm(pulse_coefs) ** 2

    dw1_dt = 0.5 * (x1 * fti + y1 * ftr)
    ### Note that w2 can be eliminated easily because
    ### the factor that appears in the derivative of the
    ### QFI associated with w2 is 8*dw2_dt = 4 * (x2 * fpI + y2 * fpR)
    ### and we add this to the QFI derivative without introducing
    ### a new variable (w2), which reduces the dimension of the system to 8.

    ### ODEs of the following variables: (x1, y1, z1, w1, x2, y2, z2, qfi/avg_num_phot)
    ### Note that Gamma =1.0 in the following ODEs.
    coef_mat = jnp.array(
        [
            [-0.5, -detuning, 2 * ftr, 0, 0, 0, 0, 0],
            [detuning, -0.5, -2 * fti, 0, 0, 0, 0, 0],
            [-2 * ftr, 2 * fti, -1, 0, 0, 0, 0, 0],
            [0.5 * fti, 0.5 * ftr, 0, 0, 0, 0, 0, 0],
            [0, -0.25, 0, 0, -0.5, -detuning, 2 * ftr, 0],
            [0.25, 0, 0, 0, detuning, -0.5, -2 * fti, 0],
            [0, 0, 0, -1, -2 * ftr, 2 * fti, -1, 0],
            [
                0,
                0,
                0.5 / avg_num_phot,
                -8.0 * (dw1_dt / avg_num_phot),
                4 * (fti / avg_num_phot),
                4 * (ftr / avg_num_phot),
                0,
                0,
            ],
        ]
    )

    inhomogenous_part = jnp.array(
        [0, 0, -1, 0, 0.5 * fti, 0.5 * ftr, 0, (0.5 / avg_num_phot)]
    )
    drvec_dt = jnp.dot(coef_mat, rvec) + inhomogenous_part

    return drvec_dt


@jit
def glb_finfo(pulse_coefs, detuning, twidth):
    """
    Parameters
    ----------
    pulse_coefs: 1d (even-sized) array containing real numbers, and it is
                    normalized to avg_num_phot.
                    That is, jnp.sum((pulse_coef)**2)=avg_num_phot.
                    These are the coefficients of basis functions, {d1,...,dn, g1,...., gn}
                    where each coefficient is given by ci = di + 1j* gi and
                    f(t) = c1 f1(t) + c2 f2(t) + ... + cn fn(t) where {fi|i=1,2,...n}
                    are the basis funcs.
    detuning: real number, detuning
    twidth: positive real number, width of the pulse

    Returns
    -------
    real number, global Fisher information at the final time (after pulse
    has passed and the atom decayed to its ground state).
    """
    ### Initial conditions (fixed):
    coef_ini = jnp.zeros(8)
    coef_ini = coef_ini.at[2].set(-1.0)

    ### ODE solver:
    term = ODETerm(ode_system)
    solver = Dopri5()

    #########################################################################
    ### Final time for the pulse
    ### contains two parts: width of the pulse + spontaneous decay time
    gamma_cap = 1.0  ### Gamma
    tfinal = twidth + (10.0 / gamma_cap) * jnp.log(10.0)
    #########################################################################
    ### Note that we add adjoint=DirectAdjoint() so that the
    ### derivative of the output can be taken:
    sol = diffeqsolve(
        term,
        solver,
        t0=0,
        t1=tfinal,
        dt0=None,
        y0=coef_ini,
        args=(pulse_coefs, detuning, twidth),
        stepsize_controller=PIDController(rtol=1e-9, atol=1e-9),
        max_steps=10**6,
        adjoint=DirectAdjoint(),
    )

    return sol.ys[0, 7]


####################################################################################
#### Optimization functions:
@jit
def objective_func_jax(pulse_coefs, avg_num_phot, detuning, twidth):
    """
    Parameters
    ----------
    pulse_coefs: 1d (even-sized) array containing real numbers, and it is
                    normalized to avg_num_phot.
                    That is, jnp.sum((pulse_coef)**2)=avg_num_phot.
                    These are the coefficients of basis functions, {d1,...,dn, g1,...., gn}
                    where each coefficient is given by ci = di + 1j* gi and
                    f(t) = c1 f1(t) + c2 f2(t) + ... + cn fn(t) where {fi|i=1,2,...n}
                    are the basis funcs.
    avg_num_phot: real number, average number of photons.
    detuning: real number, value of detuning.
    twidth: positive real number, width of the pulse

    Returns
    -------
    qfi: real number, -1*fisher information at the final time.
                        This is the objective function to be minimized.
    """

    ##### Normalizing the pulse to sqrt(avg_num_phot):
    pulse_coefs_normalized = (
        jnp.sqrt(avg_num_phot) / jnp.linalg.norm(pulse_coefs)
    ) * pulse_coefs

    ### Passing the normalized pulse for computing fisher information
    qfi = -1 * glb_finfo(pulse_coefs_normalized, detuning, twidth)

    return qfi


### The following function gives the gradient of the objective function
# pylint: disable=invalid-name
fisher_opt_grad = jit(jax.grad(objective_func_jax, argnums=0))


def sing_inst_optimum_lbfgsb(ini_seed, avg_num_phot, detuning, twidth):
    """
    Parameters
    ----------
    ini_seed: 1d array of size = 2*nmax, containing real numbers. nmax is the
              number of basis functions. This array serves as a random seed
              for the optimizer. Note that this seed may or may not be normalized
              to avg_num_phot. These are the coefficients of basis functions,
              {d1,...,dn, g1,...., gn} where each coefficient is given by
              ci = di + 1j* gi and f(t) = c1 f1(t) + c2 f2(t) + ... + cn fn(t)
              where {fi|i=1,2,...n} are the basis funcs.
    avg_num_phot: positive real number, average number of photons.
    detuning: real number, detuning.
    twidth: positive real number, width of the pulse

    Returns
    -------
    (opt_coef_pulse, opt_qfi, opt_success): results of the optimizer
                                              for a given ini_seed.
        opt_coef_pulse: 1d array with length = len(ini_seed),
                         set of pulse coefficients that optimize the objective function,
        opt_qfi: scalar, it is the function value at optimal parameters,
        opt_success: scalar(one or zero), one if the optimizer is successful else zero
    """

    ### LBFGSB optimizer:
    solver = jaxopt.ScipyMinimize(
        fun=objective_func_jax, method="L-BFGS-B", maxiter=10**4, tol=1e-9
    )

    #################################################################
    ### Running the optimizer:
    res = solver.run(
        ini_seed, avg_num_phot=avg_num_phot, detuning=detuning, twidth=twidth
    )

    opt_qfi = -1 * res.state.fun_val

    ### Normalizing the pulse:
    pulse_norm = jnp.linalg.norm(res.params)
    opt_coef_pulse = jnp.sqrt(avg_num_phot) * (res.params / pulse_norm)

    opt_success = res.state.success * 1.0

    return (opt_coef_pulse, opt_qfi, opt_success)


def multi_inst_optimum_lbfgsb(
    num_rseeds, avg_num_phot, num_basis_funcs, detuning, twidth
):
    """
    Parameters
    ----------
    num_rseeds: positive integer, number of the random seeds passed to the optimizer
    avg_num_phot: positive real number, average number of photons
    num_basis_funcs: positive integer, number of basis functions for the input pulse
    detuning: real number, detuning.
    twidth: positive real number, width of the pulse

    Returns
    -------
    (opt_pulse_coef, opt_qfi, opt_success) where
    opt_pulse_coef: 2d array with len = num_rseeds*2*num_basis_funcs,
                        with optimal pulse coeffs,
    opt_fisher: 1d array with len = num_rseeds, quantum Fisher information stored
                    for a number of random seeds.
    opt_success: 1d array with len = num_rseeds, stores where the optimizer
                is successful or not.
    """
    #### Obtaining random seeds:########################################################
    #### Key for random Seeds:
    key = random.PRNGKey(4391)
    ### Picking random pulse coefficients between 0.0 and 1.0:
    pulse_coef = random.uniform(
        key, shape=(num_rseeds, 2 * num_basis_funcs), minval=0.0, maxval=1.0
    )
    row_sums = jnp.linalg.norm(pulse_coef, axis=1, keepdims=True)
    ### Following step not necessaay, but we normalize the random seeds
    pulse_coef_normalized = jnp.sqrt(avg_num_phot) * (pulse_coef / row_sums)

    ### Choosing first random seed as the harmonic at omega =1/2:
    ind = int((twidth / jnp.pi) * 0.5)
    ### setting the whole row to zero:
    pulse_coef_normalized = pulse_coef_normalized.at[0].set(0.0)
    pulse_coef_normalized = pulse_coef_normalized.at[0, ind].set(jnp.sqrt(avg_num_phot))

    #######################################################################################
    ##### Collecting the results of the optimizer obtained for various random seeds:
    opt_pulse_coef = np.zeros(num_rseeds * 2 * num_basis_funcs)
    opt_qfi = np.zeros(num_rseeds)
    opt_success = np.zeros(num_rseeds)

    for ind in range(num_rseeds):
        # print(ind)
        (
            opt_pulse_coef[ind * 2 * num_basis_funcs : (ind + 1) * 2 * num_basis_funcs],
            opt_qfi[ind],
            opt_success[ind],
        ) = sing_inst_optimum_lbfgsb(
            pulse_coef_normalized[ind], avg_num_phot, detuning, twidth
        )

    return (opt_pulse_coef, opt_qfi, opt_success)
