import numpy as np

from xlrp import PointLensModel
from xlrp import Data
from xlrp import Event


##################
# Physics params #
##################


class PhyParams(object):
    def __init__(self) -> None:
        return

    @staticmethod
    def cal_velocity(m_s, m_c, p):
        """
        Calculate the source velocity through binary physics parameters.

        The orbit is circular.
        All units are in solar unit (Msun, AU and year).

        Parameters
        ----------
        m_s: float, source mass in solar mass.
        m_p: float, companion mass in solar mass.
        p: float, orbital period in years.

        Returns
        -------
        v: float, source velocity in km/s.
        """
        # The Kepler's third law
        r_s = ((p**2 * (m_s + m_c)) / (1 + m_s / m_c) ** 3) ** (1 / 3)

        # uniform circular motion, v is in unit of AU/yr
        v = (2 * np.pi * r_s) / (p)
        # Convert the unit to km/s
        v = v * 149597871 / (365.25 * 24 * 3600)

        return v

    @staticmethod
    def cal_xi_E(q, P_xi, theta_E=0.55, M_S=1.0, d_s=8.3):
        """
        Calculate xi_E through M_P, P_xi and theta_E.
        M_P should be in M_sun.
        P_xi should be in days.
        theta_E should be in mas.
        A typical value is 0.55 mas for bulge events with a lens mass of 0.3 Msun.
        A typical value is 2 mas for black-hole-lens bulge events.
        """
        p_year = P_xi / 365.25
        r_s_AU = q * M_S ** (1 / 3) * (p_year / (1 + q)) ** (2 / 3)
        return r_s_AU / (d_s * theta_E)

    @staticmethod
    def compute_ABFG(xi_E_N, xi_E_E, i_xi, phi_xi):
        xi_E = np.sqrt(xi_E_N**2 + xi_E_E**2)
        theta_xi = np.arctan2(xi_E_E, xi_E_N)
        Omega = -theta_xi
        a = xi_E
        omega = phi_xi
        i = i_xi
        A = a* (np.cos(omega)*np.cos(Omega) - np.sin(omega)*np.sin(Omega)*np.cos(i))  # fmt:skip
        B = a* (np.cos(omega)*np.sin(Omega) + np.sin(omega)*np.cos(Omega)*np.cos(i))  # fmt:skip
        F = a* (-np.sin(omega)*np.cos(Omega) - np.cos(omega)*np.sin(Omega)*np.cos(i))  # fmt:skip
        G = a* (-np.sin(omega)*np.sin(Omega) + np.cos(omega)*np.cos(Omega)*np.cos(i))  # fmt:skip

        return A, B, F, G

    @staticmethod
    def compute_campbell(A, B, F, G):
        omega_p_Omega = np.arctan2((B - F), (A + G))
        omega_n_Omega = np.arctan2((-B - F), (A - G))
        omega = (omega_p_Omega + omega_n_Omega) / 2
        Omega = (omega_p_Omega - omega_n_Omega) / 2

        q1 = (A + G) / np.cos(omega_p_Omega)
        q2 = (A - G) / np.cos(omega_n_Omega)
        i = 2 * np.arctan(np.sqrt(q2 / q1))
        a = 1 / 2 * (q1 + q2)

        theta = -Omega
        phi = omega

        if type(theta) == np.ndarray:
            theta_neg_mask = theta < 0
            theta_large_mask = theta > np.pi
            theta[theta_neg_mask] += np.pi
            phi[theta_neg_mask] += np.pi
            theta[theta_large_mask] -= np.pi
            phi[theta_large_mask] -= np.pi
        else:
            if theta < 0:
                theta += np.pi
                phi += np.pi
            if theta > np.pi:
                theta -= np.pi
                phi -= np.pi
        return phi, theta, i, a

    @staticmethod
    def compute_Ml_from_piE(pi_E, D_l_arr, D_s=8.2):
        """
        pi_E is a dimensionless quantity,
        D_l_arr is in kpc,
        M_l is in Msun,
        D_s is in kpc
        """
        pi_rel = 1 * (1 / D_l_arr - 1 / D_s)  # AU/kpc = mas
        kappa = 8.144  # mas/M_sun

        M_l_arr = pi_rel / (kappa * pi_E**2)
        return M_l_arr

    @staticmethod
    def compute_Ml_from_thetaE(theta_E, D_l_arr, D_s=8.2):
        """
        theta_E is in mas,
        D_l_arr is in kpc,
        M_l is in Msun
        """
        pi_rel = 1 * (1 / D_l_arr - 1 / D_s)  # AU/kpc = mas
        kappa = 8.144  # mas/M_sun

        M_l_arr = theta_E**2 / (kappa * pi_rel)
        return M_l_arr

    @staticmethod
    def compute_as(q, M_s, p):
        """
        q is the mass ratio,
        M_s is in Msun,
        p is in days,
        a_s is in AU
        """
        p_year = p / 365.25
        a_s = q / (1 + q) ** (2 / 3) * M_s ** (1 / 3) * p_year ** (2 / 3)
        return a_s

    @staticmethod
    def compute_theta_E(a_s, xi_E, D_s):
        """
        Parameters
        ----------
        a_s: in AU,
        xi_E: a dimensionless quantity.
        D_s: in kpc

        NOTE: these three parameters can be either scalars or arrays
        theta_E is in mas
        """
        theta_E = a_s / (D_s * xi_E)
        return theta_E

    @staticmethod
    def compute_mu_rel(theta_E, t_E):
        """
        Compute the relative proper motion in mas/yr.
        theta_E is in mas,
        t_E is in days.
        """
        mu_rel = theta_E / t_E * 365.25
        return mu_rel
