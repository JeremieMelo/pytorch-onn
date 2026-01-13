"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-07-18 00:03:04
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-07-18 00:03:05
"""

import numpy as np

__all__ = [
    "MORRConfig_20um_MQ",
    "MRRConfig_5um_HQ",
    "MRRConfig_5um_MQ",
    "MRRConfig_5um_LQ",
    "MORRConfig_10um_MQ",
]


class MORRConfig_20um_MQ:
    attenuation_factor = 0.8578
    coupling_factor = 0.8985
    radius = 20000  # nm
    group_index = 2.35316094
    effective_index = 2.35
    resonance_wavelength = 1554.252  # nm
    bandwidth = 0.67908  # nm
    quality_factor = 2288.7644639


class MRRConfig_5um_HQ:
    attenuation_factor = 0.987
    coupling_factor = 0.99
    radius = 5000  # nm
    group_index = 2.35316094
    effective_index = 2.4
    resonance_wavelength = 1538.739  # nm
    bandwidth = 0.2278  # nm
    quality_factor = 6754.780509


class MRRConfig_5um_MQ:
    attenuation_factor = 0.925
    coupling_factor = 0.93
    radius = 5000  # nm
    group_index = 2.35316094
    effective_index = 2.4
    resonance_wavelength = 1538.739  # nm
    bandwidth = 1.5068  # nm
    quality_factor = 1021.1965755


class MRRConfig_5um_LQ:
    attenuation_factor = 0.845
    coupling_factor = 0.85
    radius = 5000  # nm
    group_index = 2.35316094
    effective_index = 2.4
    resonance_wavelength = 1538.739  # nm
    bandwidth = 2.522  # nm
    quality_factor = 610.1265


class MORRConfig_10um_MQ:
    attenuation_factor = 0.8578
    coupling_factor = 0.8985
    radius = 10000  # nm
    group_index = 2.35316094
    effective_index = 2.4
    resonance_wavelength = 1538.739  # nm
    bandwidth = 1.6702  # nm
    quality_factor = 1213.047


def plot_curve(config):
    import matplotlib.pyplot as plt

    lambda0 = config.resonance_wavelength
    lambda_vec = np.linspace(1546, lambda0, 9400)
    aa = config.attenuation_factor  # attenuation a

    t = config.coupling_factor  # self-coupling
    # r = np.sqrt(1 - t**2) # cross coupling coef

    R = config.radius  # radius
    neff = config.effective_index  # refractive index
    phi = -4 * np.pi * np.pi * R * neff / lambda_vec

    phase_shift = np.linspace(phi[0], phi[-1], len(phi))
    phase_shift = phase_shift - np.min(phase_shift)
    print(phase_shift)
    tr = (t - aa * np.exp(1j * phi)) / (1 - t * aa * np.exp(1j * phi))
    energy = abs(tr) ** 2
    print(energy)
    plt.figure()
    plt.plot(lambda_vec, energy)
    plt.savefig("mrr_tr_wl.png")
    plt.figure()
    plt.plot(phase_shift, energy)
    plt.savefig("mrr_tr_ps.png")

    for i, e in enumerate(energy[:-1]):
        if energy[i] >= 0.5 and energy[i + 1] <= 0.5:
            print(i, i + 1)
            print(energy[i], energy[i + 1])
            print(lambda_vec[i], lambda_vec[i + 1])
            exit(1)


if __name__ == "__main__":
    plot_curve(MRRConfig_5um_MQ)
