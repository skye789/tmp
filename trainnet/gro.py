from math import ceil, sqrt

import numpy as np

from rounding import away_from_zero_numpy as round_away_from_zero


def GRO(n_in: int, FR_in: int, PE_in: int, E_in: int = 1, PF_in: int = 0, offset: float = 0) -> np.ndarray:
    """
    Python implementation of the GRO (Golden Ratio offset) sampling pattern,
    based on the MatLab implementation of the original authors.

    offset added by Marc Vornehm. Should be between 0 and 1.

    Reference:
        Rizwan Ahmad, Ning Jin, Orlando Simonetti, Yingmin Liu, and Adam
        Rich. “Cartesian sampling for dynamic magnetic resonance imaging
        (MRI)”, U.S. Patent Application No. 16/984,351 (pub. February 4th
        2021). https://patents.justia.com/patent/20210033689.

    Original MatLab implementation:
        https://github.com/OSU-CMR/GRO-CAVA
    """
    n = n_in  # Number of phase encoding (PE) lines per frame
    FR = FR_in  # Frames
    PE = PE_in  # Size of of PE grid
    E = E_in  # Number of encoding, E=1 for cine, E=2 for flow (phase-contrast MRI)
    PF = PF_in  # for partial fourier; discards PF samples from one side (default: 0, range: 0-floor(n/2), precision: 1);
    ir = 1  # ir = 1 or 2 for golden angle, ir > 2 for tiny golden angles; default value: 1
    k = 3  # k>=1. k=1 uniform; k>1 variable density profile; larger k means flatter top (default: 3)
    s = 2  # s>=0; % largers s means higher sampling density in the middle (default: 2, range: 0-10, precision: 0.1)

    gr = (1 + sqrt(5)) / 2  # golden ratio
    ga = 1 / (gr + ir - 1)  # golden angle, sqrt(2) works equally well

    # Size of the smaller pseudo-grid which after stretching gives the true grid size
    PES = ceil(PE * 1 / s)  # Size of shrunk PE grid
    vd = (PE / 2 - PES / 2) / ((PES / 2) ** k)  # location specific displacement
    samp = np.zeros((PE, FR, E))  # sampling on PE-t grid
    PEInd = np.zeros(((n - PF) * FR, E))  # The ordered sequence of PE indices

    eps = 1e-10
    v0 = np.arange(1 / 2 + eps, PES + 1 / 2 - eps, PES / (n + PF))  # Start with uniform sampling for each frame
    v0 += offset * PES / (n + PF)
    for e in range(E):
        v0 += 1 / E * PES / (n + PF)  # Start with uniform sampling for each frame
        kk = E - e
        for j in range(FR):
            v = ((v0 + (j - 1) * PES / (n + PF) * ga) - 1) % PES + 1  # In each frame, shift by golden shift of PES/TR*ga
            v -= PES * (v >= PES + 0.5)

            if PE % 2 == 0:  # if even, shift by 1/2 pixel
                vC = v - vd * np.sign((PES / 2 + 1 / 2) - v) * abs((PES / 2 + 1 / 2) - v) ** k + (PE - PES) / 2 + 1 / 2
                vC -= PE * (vC >= PE + 0.5)
            elif PE % 2 == 1:  # if odd don't shift
                vC = v - vd * np.sign((PES / 2 + 1 / 2) - v) * abs((PES / 2 + 1 / 2) - v) ** k + (PE - PES) / 2
            vC.sort()
            vC = round_away_from_zero(vC)
            vC = vC[PF:]

            if j % 2 == 1:
                PEInd[j * n:(j + 1) * n, e] = vC
            else:
                PEInd[j * n:(j + 1) * n, e] = vC[::-1]

            samp[vC.astype(int) - 1, j, e] += kk

    return samp
