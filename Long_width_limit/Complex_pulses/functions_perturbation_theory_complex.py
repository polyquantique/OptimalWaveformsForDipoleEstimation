"""
This contains functions that output the QFI per unit of a coherent state
in the long-width limit and QFI of a single photon pulse for complex pulses.
"""

import jax
from jax import config, jit
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


def harmonic_pulse(tval, ind, twidth, avg_num_phot):
    """
    Parameters
    ----------
    tval: non-negative number, time at which pulse amplitude is required
    ind: positive integer corresponding to the order of harmonic function
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


def sqr_plane_pulse(tval, freq, twidth, avg_num_phot):
    """
    Parameters
    ----------
    tval: non-negative number, time at which pulse amplitude is required
    freq: positive scalar, angular frequency of the cosine function
    twidth: positive scalar, it is the width of the square pulse
    avg_num_phot: positive real number, average number of pump photons

    Returns
    -------
    (jnp.real(pulse), jnp.imag(pulse)): tuple containing two scalars,
    the values correspond to the function prefac*exp(i*freq*t),
    where prefac is chosen such that the L2 norm of the function is avg_num_phot.
    """
    prefac = jnp.sqrt(avg_num_phot / twidth)

    mask = (tval >= 0) & (tval <= twidth)
    complex_pulse = prefac * jnp.exp(1j * freq * tval)
    pulse = jnp.where(mask, complex_pulse, 0.0)

    return (jnp.real(pulse), jnp.imag(pulse))


#######################################################################################
### ODEs of a coherent state:
@jit
def ode_system_coherent(tval, rvec, args):
    """
    Parameters
    ----------
    tval: non-negative number, time at which pulse amplitude is required
    rvec: [r1_1r, r1_1i, r2_1r, r2_1i,
           z1_2, z2_2, w1_2, w2_2,
           r1_3r, r1_3i, r2_3r, r2_3i,
           z1_4, z2_4, w1_4, w2_4,
           qfi2, qfi4], 18-d array of real numbers

        See Appendix I for more details

        r1_1r, r1_1i, r2_1r, r2_1i: first-order real and imaginary parts of
                                                r1 and r2.
        z1_2, z2_2, w1_2, w2_2: second-order z1, z2, w1 and w2.
        r1_3r, r1_3i, r2_3r, r2_3i: third-order real and imaginary parts of
                                                r1 and r2.
        z1_4, z2_4, w1_4, w2_4: fourth-order z1, z2, w1 and w2.
        qfi2, qfi4: second-order and the fourth-order qfi respectively

    args: (twidth, avg_num_phot, delta)
        twidth: positive number, width of the pulse
        avg_num_phot: positive real number, average number of photons in the pulse
        delta: real number, detuning

    Returns
    -------
    state_final: 18 dimensional real array with the derivatives of the variables
                defined in rvec (d(rvec)/dt)
    """

    gamma_cap = 1.0
    twidth, avg_num_phot, delta = args
    kval = 0.5 * gamma_cap - 1j * delta

    #############################################
    ### Three choices for the pulse:
    ##############################################
    ### 1. Gaussian (real):
    # ft = gauss_pulse(tval, twidth, avg_num_phot)

    ### 2. Harmonics (complex):
    ind1=3
    ind2=13
    ftr = harmonic_pulse(tval, ind1, twidth, avg_num_phot/2)
    fti = harmonic_pulse(tval, ind2, twidth, avg_num_phot/2)
    ft = ftr + 1j*fti

    ### 3. Plane wave
    # freq = delta - 0.5
    # # freq = delta + 0.5
    # (ftr, fti) = sqr_plane_pulse(tval, freq, twidth, avg_num_phot)
    # ft = ftr - 1j * fti
    ##############################################

    ft_conj = jnp.conjugate(ft)

    (
        r1_1r,
        r1_1i,
        r2_1r,
        r2_1i,
        z1_2,
        z2_2,
        w1_2,
        w2_2,
        r1_3r,
        r1_3i,
        r2_3r,
        r2_3i,
        z1_4,
        z2_4,
        w1_4,
        w2_4,
        qfi2,
        qfi4,
    ) = rvec

    r1_1 = r1_1r + 1j * r1_1i
    r2_1 = r2_1r + 1j * r2_1i
    r1_3 = r1_3r + 1j * r1_3i
    r2_3 = r2_3r + 1j * r2_3i

    # First-order equations
    dr1_1 = -kval * r1_1 - 2 * ft_conj
    dr2_1 = -kval * r2_1 + (1j / 4) * r1_1 + (1j / 2) * ft_conj

    # Second-order equations
    dz1_2 = -z1_2 - 2 * jnp.real(ft * r1_1)
    dw1_2 = (1 / 2) * jnp.imag(ft * r1_1)
    dz2_2 = -z2_2 - w1_2 - 2 * jnp.real(ft * r2_1)
    dw2_2 = (1 / 2) * jnp.imag(ft * r2_1)
    dqfi_2 = (1 / 2) * z1_2 + 8 * dw2_2

    # Third-order equations
    dr1_3 = -kval * r1_3 + 2 * ft_conj * z1_2
    dr2_3 = -kval * r2_3 + 2 * ft_conj * z2_2 + (1j / 4) * r1_3

    # Fourth-order equations
    dz1_4 = -z1_4 - 2 * jnp.real(ft * r1_3)
    dw1_4 = (1 / 2) * jnp.imag(ft * r1_3)
    dz2_4 = -z2_4 - w1_4 - 2 * jnp.real(ft * r2_3)
    dw2_4 = (1 / 2) * jnp.imag(ft * r2_3)
    dqfi_4 = (1 / 2) * z1_4 + 8 * dw2_4 - 8 * w1_2 * dw1_2

    drvec_dt = jnp.array(
        [
            jnp.real(dr1_1),
            jnp.imag(dr1_1),
            jnp.real(dr2_1),
            jnp.imag(dr2_1),
            dz1_2,
            dz2_2,
            dw1_2,
            dw2_2,
            jnp.real(dr1_3),
            jnp.imag(dr1_3),
            jnp.real(dr2_3),
            jnp.imag(dr2_3),
            dz1_4,
            dz2_4,
            dw1_4,
            dw2_4,
            dqfi_2,
            dqfi_4,
        ]
    )

    return drvec_dt

@jit
def qfi_coherent(twidth, avg_num_phot, delta):
    """
    Parameters
    ----------
    twidth(tcap): positive number, twidth of the harmonic(or plane-wave 
                                            or tcap of the Gaussian pulse)
    avg_num_phot: positive real number, average number of photons in the pulse
    delta: real number, detuning

    Returns
    -------
    output: [qfi_2, qfi_4] where
            qfi_2 is the final QFI per unit photon at the second order 
            qfi_4 is the final QFI per unit photon at the fourth order
    """
    gamma_cap = 1.0
    coef_ini = jnp.zeros(18)  # Initial conditions
    term = ODETerm(ode_system_coherent)  ## ODEs
    solver = Dopri5()  ## ODE solver

    ################################################################
    ### Two choices for final time depending on the pulse:
    ### (select depending on the pulse chosen in ODEs)
    ##1.  Gaussian pulse:
    # tfinal = 12*(twidth/jnp.sqrt(2)) + (10/gamma_cap)*jnp.log(10)

    ##2.  Harmonics or plane-wave:
    tfinal = twidth + (10.0 / gamma_cap) * jnp.log(10.0)
    #################################################################
    ### Solving ODEs
    sol = diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=tfinal,
        dt0=None,
        y0=coef_ini,
        args=(twidth, avg_num_phot, delta),
        stepsize_controller=PIDController(rtol=1e-14, atol=1e-14),
        max_steps=10**6,
    )

    output = (
        sol.ys[:, 16]/avg_num_phot,
        sol.ys[:, 17]/avg_num_phot,
    )

    return output

#######################################################################################
### ODEs for single photons from Elnaz et al.:

def odes_single(tval, rvec, args):
    """
    Parameters
    ----------
    tval: non-negative number, time at which the function is evaluated
    rvec:  (preal, pimag, qreal, qimag, rreal, rimag, qfi2_single, qfi4_single)
        preal, pimag: real numbers corresponding to the real and the imaginary parts
                        of P(t) defined in Eq. 81 (or I8)
        qreal, qimag: real numbers corresponding to the real and the imaginary parts
                        of Q(t) defined in Eq. 80 (or I9)
        rreal, rimag: real numbers corresponding to the real and the imaginary parts
                        of R(t) defined in Eq. I60 
        qfi2_single: real number, variable defined in the first part of I57
                                (QFI upto second order in pulse shape for a single-photon pulse)
        qfi4_single: real number, variable defined in the second part of I57
                                (QFI upto fourth order in pulse shape for a single-photon pulse)

    args: (twidth, delta)
        twidth: positive number, width of the pulse
        delta: real number, detuning

    Returns
    -------
    state_final: 9 dimensional real array with the derivatives of the variables
                defined in rvec (d(rvec)/dt)
    """
    (preal, pimag, qreal, qimag, rreal, rimag, qfi2_single, qfi4_single) = rvec
    
    twidth, delta = args
    avg_num_phot = 1.0  ## single-photon pulse
    #############################################
    ### Three choices for the pulse:
    ##############################################
    ### 1. Gaussian (real):
    # fp = gauss_pulse(tval, twidth, avg_num_phot)

    ### 2. Harmonics (complex):
    ind1=3
    ind2=13
    fpr = harmonic_pulse(tval, ind1, twidth, avg_num_phot/2)
    fpi = harmonic_pulse(tval, ind2, twidth, avg_num_phot/2)
    fp = fpr + 1j*fpi

    ### 3. Plane wave
    # freq = delta + 0.5
    # freq = delta - 0.5
    # (fpr, fpi) = sqr_plane_pulse(tval, freq, twidth, avg_num_phot)
    # fp = fpr - 1j * fpi
    ##############################################

    # Reconstruct complex scalars
    p = preal + 1j * pimag
    q = qreal + 1j * qimag
    r = rreal + 1j * rimag

    # Original complex derivatives
    dp_dt = -0.5 * p + jnp.exp(-1j * delta * tval) * jnp.conj(fp)
    dq_dt = -0.5 * q - 0.5 * p
    dr_dt = jnp.conj(p) * dp_dt
    dqfi2_dt = 16.0 * jnp.abs(dq_dt) ** 2
    dqfi4_dt = -8.0 * jnp.real(jnp.conj(r) * dr_dt)

    # Split into real parts for the solver
    return jnp.array(
        [
            jnp.real(dp_dt),
            jnp.imag(dp_dt),
            jnp.real(dq_dt),
            jnp.imag(dq_dt),
            jnp.real(dr_dt),
            jnp.imag(dr_dt),
            dqfi2_dt,
            dqfi4_dt,
        ]
    )


def qfi_single(twidth, delta):
    """
    Parameters
    ----------
    twidth(tcap): positive number, twidth of the harmonic(or plane-wave 
                                            or tcap of the Gaussian pulse)
    delta: real number, detuning

    Returns
    -------
    output: [qfi2(t), qfi4(t)]
        qfi2(t): real number, qfi second order term defined in Eq. I57 (single photon)
        qfi4(t): real number, qfi fourth order term defined in Eq. I57 (single photon)
    """
    coef_ini = jnp.zeros(8)  # Initial conditions
    term = ODETerm(odes_single)  ## ODEs
    solver = Dopri5()  ## ODE solver
    Gamma = 1.0
    ################################################################
    ### Two choices for final time depending on the pulse:
    ### (select depending on the pulse chosen in ODEs)
    ##1.  Gaussian pulse:
    # tfinal = 12*(twidth/jnp.sqrt(2)) + (10/Gamma)*jnp.log(10)

    ##2.  Harmonics or plane-wave:
    tfinal = twidth + (10.0 / Gamma) * jnp.log(10.0)
    #################################################################
    ## Solving ODEs:
    sol = diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=tfinal,
        dt0=None,
        y0=coef_ini,
        args=(twidth, delta),
        stepsize_controller=PIDController(rtol=1e-14, atol=1e-14),
        max_steps=10**6,
    )

    output = (sol.ys[:, 6], sol.ys[:, 7])

    return output
