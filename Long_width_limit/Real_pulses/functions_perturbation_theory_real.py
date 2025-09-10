"""
This contains functions that output the QFI per unit of a coherent state
in the long-width limit and QFI of a single photon pulse for a real pulse.
"""

import jax
from jax import config
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Dopri5, PIDController

### Restricting the usage to the CPU
jax.config.update("jax_default_device", jax.devices("cpu")[0])

# # JAX supports single-precisions numbers by default.For double precision, use:
# # https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html
config.update("jax_enable_x64", True)


################################################################################
### Example pulses:
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


def harmonic_pulse(ind, tval, twidth, avg_num_phot):
    """
    Parameters
    ----------
    ind: positive integer corresponding to the order of harmonic function
    tval: non-negative number, time at which pulse amplitude is required
    twidth: positive number, width of the pulse
    avg_num_phot: positive real number, average number of photons in the pulse

    Returns
    -------
    pulse: real number, sqrt(2*num_phot/T)*sin(ind*pi*(t/Tpulse)) 0<t<T
                        and zero for t<0 or t>Tpulse
    """
    prefac = jnp.sqrt((2 * avg_num_phot) / twidth)
    pulse = jnp.piecewise(
        tval,
        [(tval >= 0) & (tval <= twidth), tval > twidth],
        [prefac * jnp.sin(ind * (jnp.pi / twidth) * tval), 0.0],
    )
    return pulse


##########################################################################################
### ODEs of a coherent state:
def ode_system_coherent(tval, rvec, args):
    """
    Parameters
    ----------
    tval: non-negative number, time at which pulse amplitude is required
    rvec: (pt, qt, qfi_long)
        pt: real number, variable defined in Eq. 63 (also Gamma=1 here)
        qt: real number, variable defined in Eq. 65
        qfi_long: real number, QFI of a coherent state in the long-width limit (see Appendix F)

    args: (twidth(tcap), num_phot)
        twidth(tcap): positive number, twidth of the harmonic(tcap of the Gaussian pulse)
        avg_num_phot: positive real number, average number of photons in the pulse

    Returns
    -------
    state_final: [dp_dt, dq_dt, dqfi_long_dt]
                derivatives of the variables pt, qt, qfi_long
    """
    pt, qt, _ = rvec

    ########################################################
    ## Two choices for the pulse:
    ## 1. Gaussian Pulse:
    twidth, avg_num_phot = args
    ft = gauss_pulse(tval, twidth, avg_num_phot)

    ##2. Harmonics:
    # twidth, avg_num_phot = args
    # ind = 7 ### chnage ind in ode_system_single if changed here
    # ft = harmonic_pulse(ind, tval, twidth, avg_num_phot)
    ########################################################

    ### ODEs (assuming Gamma=1) (see Appendix F):
    dp_dt = -(pt / 2) + ft
    dq_dt = -(qt / 2) - (pt / 2)
    dqfi_long_dt = (
        pt**2 + (2 * ft * pt) + (4 * ft * qt)
    )  ### qfi of coherent state in the long-width

    state_final = jnp.array([dp_dt, dq_dt, dqfi_long_dt])

    return state_final


def qfi_coherent(twidth, avg_num_phot):
    """
    Parameters
    ----------
    twidth(tcap): positive number, twidth of the harmonic(tcap of the Gaussian pulse)
    avg_num_phot: positive real number, average number of photons in the pulse

    Returns
    -------
    qfi_final: real number, final QFI per unit photon obtained using the ODEs
            for a coherent state in the perturbation theory up to second order.
    """
    gamma_cap = 1.0

    # Initial conditions:
    coef_ini = jnp.zeros(3)

    ## ODE solver:
    term = ODETerm(ode_system_coherent)
    solver = Dopri5()

    ### Two choices for final time depending on the pulse:
    ### (select depending on the pulse chosen in ODEs)
    ###1.  Gaussian pulse:
    tfinal = 12*(twidth/jnp.sqrt(2)) + (10/gamma_cap)*jnp.log(10)

    ###2.  Harmonics:
    # tfinal = twidth + (10 / gamma_cap) * jnp.log(10)

    sol = diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=tfinal,
        dt0=None,
        y0=coef_ini,
        args=(twidth, avg_num_phot),
        stepsize_controller=PIDController(rtol=1e-14, atol=1e-14),
        max_steps=10**6,
    )

    qfi_final = sol.ys[:, 2] / avg_num_phot

    return qfi_final


##########################################################################################
### ODEs of a single photon pulse:
def ode_system_single(tval, rvec, twidth):
    """
    Parameters
    ----------
    tval: non-negative number, time at which pulse amplitude is required
    rvec: (pt, qt, qfi_single)
        pt: real number, variable defined in Eq. 63 (also Gamma=1 here)
        qt: real number, variable defined in Eq. 65
        qfi_single: real number, QFI of a single photon pulse (see Appendix G)

    (twidth(tcap): positive real number, twidth of the harmonic(tcap of the Gaussian pulse)

    Returns
    -------
    state_final: [dp_dt, dq_dt, dqfi_single_dt]
                derivatives of the variables pt, qt, qfi_single
    """
    pt, qt, _ = rvec
    avg_num_phot = 1.0  # single photon pulse
    ########################################################
    ## Two choices for the pulse:
    ## 1. Gaussian Pulse:
    ft = gauss_pulse(tval, twidth, avg_num_phot)

    ##2. Harmonics:
    # ind = 7 ### chnage ind in ode_system_coherent if changed here
    # ft = harmonic_pulse(ind, tval, twidth, avg_num_phot)

    ########################################################
    ### ODEs (assuming Gamma=1) (see Appendix F and G):
    dp_dt = -(pt / 2) + ft
    dq_dt = -(qt / 2) - (pt / 2)
    dqfi_single_dt = 16 * (dq_dt) ** 2  ### qfi of single photon

    state_final = jnp.array([dp_dt, dq_dt, dqfi_single_dt])

    return state_final


def qfi_single(twidth):
    """
    Parameters
    ----------
    twidth(tcap): positive real number, twidth of the harmonic(tcap of the Gaussian pulse)

    Returns
    -------
    qfi_final: positive real number, the final QFI obtained for a single-photon pulse
    """
    gamma_cap = 1.0

    # Initial conditions:
    coef_ini = jnp.zeros(3)

    ## ODE solver:
    term = ODETerm(ode_system_single)
    solver = Dopri5()

    ### Two choices for final time depending on the pulse:
    ### (select depending on the pulse chosen in ODEs)
    ##1.  Gaussian pulse:
    tfinal = 12*(twidth/jnp.sqrt(2)) + (10/gamma_cap)*jnp.log(10)

    ##2.  Harmonics:
    # tfinal = twidth + (10 / gamma_cap) * jnp.log(10)

    sol = diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=tfinal,
        dt0=None,
        y0=coef_ini,
        args=(twidth),
        stepsize_controller=PIDController(rtol=1e-14, atol=1e-14),
        max_steps=10**6,
    )

    qfi_final = sol.ys[:, 3]

    return qfi_final
