import sys

import subprocess
import numpy as np
from matplotlib import pyplot as plt
import os
# from drawarrow import fig_arrow

FONT_LARGE = 27
WIDTH = 12
HEIGHT = 6
LINE_WIDTH = 2

def get_Delta_from_string(vec: str, N_sys: int, w:np.ndarray) -> Keldysh:

    env = os.environ.copy()
    env["PYTHONPATH"] = "/home/daniel/gitlab/amea_all/"

    with open("./output.dat", "w") as f:
        f.write(vec)

    kwargs_convert_fit_output = [
        "--loadpars",
        "./output.dat",
        "--outgam",
        "./gam_mat.dat",
    ]

    subprocess.run(
        [
            "python",
            "/home/daniel/gitlab/amea_all/Fit/getparameters.py",
        ]
        + kwargs_convert_fit_output,
        check=True,
        env=env,
    )

    full_matrix = np.loadtxt("./gam_mat.dat", comments=["h", "R", "I"])

    os.remove("./gam_mat.dat")
    os.remove("./output.dat")

    E_temp = full_matrix[:N_sys, :]

    Gam_1_re = full_matrix[N_sys : 2 * N_sys, :]
    Gam_1_im = full_matrix[2 * N_sys : 3 * N_sys, :]
    Gam_1_temp = Gam_1_re + 1j * Gam_1_im

    Gam_2_re = full_matrix[3 * N_sys : 4 * N_sys, :]
    Gam_2_im = full_matrix[4 * N_sys : 5 * N_sys, :]
    Gam_2_temp = Gam_2_re + 1j * Gam_2_im

    N_true = 4 * N_sys

    L = np.zeros((N_true // 2, N_true // 2), dtype=complex)

    omega = Gam_2_temp - Gam_1_temp
    L[:N_sys, :N_sys] = E_temp + 1j * omega
    L[:N_sys, N_sys:] = 2 * Gam_2_temp
    L[N_sys:, :N_sys] = -2 * Gam_1_temp
    L[N_sys:, N_sys:] = E_temp - 1j * omega

    eigen_L, V_L = np.linalg.eig(L)
    V_L_inv = np.linalg.inv(V_L)
    D_L = np.heaviside(np.imag(eigen_L), -np.inf)
    D_L_mat = D_L * np.identity(N_true // 2)
    D_L_mat_ = (1 - D_L) * np.identity(N_true // 2)

    eigen_L_mat = eigen_L * np.identity(N_true // 2)

    g_0_greater_mat = np.array(
        [V_L @ (D_L_mat_ @ np.linalg.inv(w * np.identity(N_true // 2) - eigen_L_mat) @ V_L_inv) for w in w]
    )

    g_0_lesser_mat = np.array(
        [V_L @ (D_L_mat @ np.linalg.inv(w * np.identity(N_true // 2) - eigen_L_mat) @ V_L_inv) for w in w]
    )

    keldysh_mat = (
        g_0_greater_mat
        + g_0_lesser_mat
        - (np.swapaxes(g_0_greater_mat.conj(), -1, -2) + np.swapaxes(g_0_lesser_mat.conj(), -1, -2))
    )
    retarded_mat = g_0_greater_mat + np.swapaxes(g_0_lesser_mat.conj(), -1, -2)

    GF_0 = Keldysh(
        R=retarded_mat[:, N_sys // 2, N_sys // 2],
        K=keldysh_mat[:, N_sys // 2, N_sys // 2],
    )

    Delta_R_aux = -1 / GF_0.R + w
    Delta_K_aux = GF_0.R**-1 * GF_0.R.conj() ** -1 * GF_0.K

    Delta = Keldysh(R=Delta_R_aux, K=Delta_K_aux)

    return Delta

@dataclass
class Keldysh:
    R: np.ndarray | None = None
    R_s: np.ndarray | None = None
    K: np.ndarray | None = None
    K_s: np.ndarray | None = None
    occ_func: np.ndarray | None = None
    occ: np.ndarray | None = None

    def calc_occ_func(self):
        self.occ_func = (1 - np.imag(self.K) / np.imag(self.R) / 2) / 2

    def calc_occ(self):
        self.occ = self.occ_func * (-np.imag(self.R)) / np.pi
