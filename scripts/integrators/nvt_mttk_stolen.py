#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def harmonic_pot(x, m=1, omega=1):
    """
    The harmonic potential
    """

    return 0.5 * m * omega ** 2 * x ** 2


def harmonic_for(x, m=1, omega=1):
    """
    The force of harmonic potential
    """

    return - m * omega ** 2 * x


def e_kin(v, m=1):
    """
    The force of harmonic potential
    """

    return 0.5 * m * v ** 2


def log(x, v, m=1, omega=1):
    """
    """

    print("{:20.8E} {:20.8E} {:20.8E} {:20.8E} {:20.8E}".format(
        x, v,
        e_kin(v, m),
        harmonic_pot(x, m, omega),
        harmonic_pot(x, m, omega) + e_kin(v, m)
    ))


def nose_hoover_chain(
        x_0,
        v_0,
        temperature,
        q_1,
        mm=2,
        n_c=1,
        nys=3,
        f0=None,
        m=1,
        omega=1,
        time_step=0.01,
        n_steps=1000,
):
    """
    G. J. Martyna, M. E. Tuckerman, D. J. Tobias, and Michael L. Klein,
    "Explicit reversible integrators for extended systems dynamics"
    Molecular Physics, 87, 1117-1157 (1996) 
    """
    x_array = []
    v_array = []

    assert mm >= 1
    assert n_c >= 1
    assert nys == 3 or nys == 5

    if nys == 3:
        tmp = 1 / (2 - 2 ** (1. / 3))
        wdti = np.array([tmp, 1 - 2 * tmp, tmp]) * time_step / n_c
    else:
        tmp = 1 / (4 - 4 ** (1. / 3))
        wdti = np.array([tmp, tmp, 1 - 4 * tmp, tmp, tmp]) * time_step / n_c

    q_mass = np.ones(mm) * q_1
    # if M > 1: Qmass[1:] /= 2
    v_logs = np.zeros(mm)
    x_logs = np.zeros(mm)
    g_logs = np.zeros(mm)
    # for ii in range(1, M):
    #     Glogs[ii] = (Qmass[ii-1] * Vlogs[ii-1]**2 - T) / Qmass[ii]

    if f0 is None:
        f0 = harmonic_for(x_0, m, omega)

    log(x_0, v_0, m, omega)
    x_array.append(x_0)
    v_array.append(v_0)

    def nhc_step(v, Glogs, Vlogs, Xlogs):
        """
        """

        scale = 1.0
        M = Glogs.size
        K = e_kin(v, m)
        K2 = 2 * K
        Glogs[0] = (K2 - temperature) / q_mass[0]

        for inc in range(n_c):
            for iys in range(nys):
                wdt = wdti[iys]
                # update the thermostat velocities
                Vlogs[-1] += 0.25 * Glogs[-1] * wdt

                for kk in range(M - 1):
                    AA = np.exp(-0.125 * wdt * Vlogs[M - 1 - kk])
                    Vlogs[M - 2 - kk] = Vlogs[M - 2 - kk] * AA * AA \
                                        + 0.25 * wdt * Glogs[M - 2 - kk] * AA

                # update the particle velocities
                AA = np.exp(-0.5 * wdt * Vlogs[0])
                scale *= AA
                # update the forces
                Glogs[0] = (scale * scale * K2 - temperature) / q_mass[0]
                # update the thermostat positions
                Xlogs += 0.5 * Vlogs * wdt
                # update the thermostat velocities
                for kk in range(M - 1):
                    AA = np.exp(-0.125 * wdt * Vlogs[kk + 1])
                    Vlogs[kk] = Vlogs[kk] * AA * AA \
                                + 0.25 * wdt * Glogs[kk] * AA
                    Glogs[kk + 1] = (q_mass[kk] * Vlogs[
                        kk] ** 2 - temperature) / q_mass[
                                        kk + 1]
                Vlogs[-1] += 0.25 * Glogs[-1] * wdt

        return v * scale

    # 0, 1, 2 represents t, t + 0.5*dt, and t + dt, respectively
    for ii in range(n_steps):
        vnhc = nhc_step(v_0, g_logs, v_logs, x_logs)
        v1 = vnhc + 0.5 * time_step * f0 / m
        x2 = x_0 + v1 * time_step
        f2 = harmonic_for(x2, m, omega)
        v2p = v1 + 0.5 * time_step * f2 / m
        v2 = nhc_step(v2p, g_logs, v_logs, x_logs)

        log(x2, v2, m, omega)
        x_0, v_0, f0 = x2, v2, f2
        x_array.append(x_0)
        v_array.append(v_0)
    return x_array, v_array


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    T0 = 0.1
    v0 = np.sqrt(4 * T0) * 2
    x0 = 0.0
    dt = 0.1
    N = 10000

    X, V = nose_hoover_chain(
        x0, v0, temperature=T0,
        q_1=0.1, mm=1,
        # # q_1=0.1 * 16, mm=2, n_c=1,
        # # q_1=0.1, mm=2, n_c=1,
        # q_1=0.1 / 16, mm=2, n_c=1,
        time_step=0.1, n_steps=N
    )

    plt.plot(X, V)
    plt.show()
