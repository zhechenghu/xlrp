import numpy as np
from pytest import approx
from copy import deepcopy
import os
from xlrp import PointLensModel, Data, Event
from xlrp.utils import ReadInfo, FitUtils
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Ut:
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


event_ri = ReadInfo("ob150845_config.yml")

# standard parameters
std_params = {
    "t_0": 2457199.457,
    "u_0": 0.0581,
    "t_E": 36.37,
}

# parallax parameters
ra, dec = event_ri.get_coords(return_type="str").split(" ")
t_0_par = 2457200
prlx_params = {
    "t_0": 2457199.456,
    "u_0": 0.0571,
    "t_E": 36.91,
    "pi_E_N": -0.00209,
    "pi_E_E": 0.0874,
    "t_0_par": t_0_par,
}

# xallarap paramters
t_ref = 2457200
# Thiele-Innes only
prlx_xlrp_ti_params = {
    "t_0": 2457199.3855,
    "u_0": 0.0500,
    "t_E": 42.0,
    "pi_E_N": 0.0003,
    "pi_E_E": 0.0775,
    "A_xi": -0.0101,
    "B_xi": -0.0120,
    "F_xi": -0.0060,
    "G_xi": -0.0615,
    "p_xi": 40.5,
    "t_0_par": t_0_par,
    "t_ref": t_ref,
}

# campbell only
_phi_xi, _theta_xi, _i_xi, _a_xi = Ut.compute_campbell(
    prlx_xlrp_ti_params["A_xi"],
    prlx_xlrp_ti_params["B_xi"],
    prlx_xlrp_ti_params["F_xi"],
    prlx_xlrp_ti_params["G_xi"],
)
_xi_E_N = _a_xi * np.cos(_theta_xi)
_xi_E_E = _a_xi * np.sin(_theta_xi)
prlx_xlrp_cpb_params = {
    "t_0": prlx_xlrp_ti_params["t_0"],
    "u_0": prlx_xlrp_ti_params["u_0"],
    "t_E": prlx_xlrp_ti_params["t_E"],
    "pi_E_N": prlx_xlrp_ti_params["pi_E_N"],
    "pi_E_E": prlx_xlrp_ti_params["pi_E_E"],
    "xi_E_N": _xi_E_N,
    "xi_E_E": _xi_E_E,
    "i_xi": _i_xi,
    "phi_xi": _phi_xi,
    "p_xi": prlx_xlrp_ti_params["p_xi"],
    "t_0_par": t_0_par,
    "t_ref": t_ref,
}

# Thiele-Innes binary source
q_xi = 0.4
qf_ogle = 0.4**3.5
prlx_xlrp_ti_2s_params = {
    "t_0": 2457199.284,
    "u_0": 0.0484,
    "t_E": 46.8,
    "pi_E_N": -0.0083,
    "pi_E_E": 0.0760,
    "A_xi": -0.061,
    "B_xi": 0.148,
    "F_xi": 0.024,
    "G_xi": 0.072,
    "p_xi": 71.7,
    "q_xi": 0.299,
    "qf_ogle": 0.094,
    "qf_ogleV": 0.107,
    "qf_spitzer": 0.92,
    "t_0_par": t_0_par,
    "t_ref": t_ref,
}

# compbell binary source
# pass for now

# compbell single source eccentric orbit
prlx_xlrp_cpb_ecc_params = {
    "t_0": 2457199.4134,
    "u_0": 0.05644,
    "t_E": 36.572,
    "pi_E_N": 0.00812,
    "pi_E_E": 0.08410,
    "i_xi": 1.5584,
    "phi_xi": 1.602,
    "xi_E_N": 0.03128,
    "xi_E_E": -0.08886,
    "p_xi": 35.3872,
    "e_xi": 0.3,
    "omega_xi": 0.0428,
    "Omega_xi": 0.0,
    "t_0_par": t_0_par,
    "t_ref": t_ref,
}

# prlx_bins_params = {**prlx_params, "t_0_2": t_0_2, "u_0_2": u_0_2}


def test_init_std_model():
    parameters = std_params.copy()
    model = PointLensModel(parameters, obname="ogle")
    assert model.obname is "ogle"
    assert model.parameters == parameters
    assert model.parameter_set_enabled == ["std"]
    assert model.delta_sun is None
    assert model.t_0_par is None
    assert model.ra is None
    assert model.dec is None
    assert model.alpha is None
    assert model.delta is None
    assert model.ephemeris is None
    assert model.neg_delta_sate is None
    assert model.t_ref is None
    assert model.jds.size == 0
    assert model.trajectory.size == 0
    assert model.magnification.size == 0
    assert model.model_flux.size == 0
    assert model.model_mag.size == 0


def test_init_prlx_model():
    parameters = prlx_params.copy()
    model = PointLensModel(parameters, ra=ra, dec=dec, obname="ogle")
    assert model.parameter_set_enabled == ["std", "prlx"]
    assert model.t_0_par == 2457200
    assert model.ra == "18:04:21.29"
    assert model.dec == "-31:34:50.0"
    assert model.alpha == approx(271.0887083)
    assert model.delta == approx(-31.5805556)


def test_init_xlrp_cpb_model():
    parameters = prlx_xlrp_cpb_params.copy()
    model = PointLensModel(parameters, ra=ra, dec=dec, obname="ogle")
    assert model.parameter_set_enabled == ["std", "prlx", "xlrp_circ"]
    assert model.t_ref == 2457200


def test_init_xlrp_ti_model():
    parameters = prlx_xlrp_ti_params.copy()
    model = PointLensModel(parameters, ra=ra, dec=dec, obname="ogle")
    assert model.parameter_set_enabled == ["std", "prlx", "xlrp_circ_ti"]


def test_init_xlrp_ti2s_model():
    parameters = prlx_xlrp_ti_2s_params.copy()
    model = PointLensModel(parameters, ra=ra, dec=dec, obname="ogle")
    assert model.parameter_set_enabled == ["std", "prlx", "xlrp_circ_ti_2s"]


# def test_init_prlx_bins_model():
#    parameters = prlx_bins_params.copy()
#    model = PointLensModel(parameters, ra=ra, dec=dec)
#    assert model.parameter_set_enabled == ["std", "prlx", "bins"]


def test_std_event():
    std_event_ri = deepcopy(event_ri)
    std_event_ri.event_info["Event_Info"]["observatories"] = ["ogle", "ogleV"]
    parameters = std_params.copy()
    event = FitUtils.build_mylens_event(std_event_ri, params_dict=parameters)
    assert event.ground_ob_tup == ("ogle", "ogleV")
    assert event.all_ob_tup == ("ogle", "ogleV")
    assert event.get_chi2() == approx(1292.62, abs=1e-2)
    traj = event.model_dict["ogle"].get_trajectory()
    assert traj[0].sum() == approx(-15372.50076024, abs=1e-2)
    assert traj[1].sum() == approx(52.81290000, abs=1e-2)


def test_prlx_event():
    parameters = prlx_params.copy()
    event = FitUtils.build_mylens_event(event_ri, params_dict=parameters)
    assert event.ground_ob_tup == ("ogle", "ogleV")
    assert event.sapce_ob_tup == ("spitzer",)
    assert event.all_ob_tup == ("ogle", "ogleV", "spitzer")
    assert event.get_chi2() == approx(1531.65, abs=1e-2)
    traj_val_dict = {
        "ogle": (-15953.095015617691, 64.6448898),
        "ogleV": (-1491.2996992655806, 6.290679286),
        "spitzer": (20.7707781948, 6.1947392193),
    }
    for ob in event.all_ob_tup:
        traj = event.model_dict[ob].get_trajectory()
        assert traj[0].sum() == approx(traj_val_dict[ob][0], abs=1e-5)
        assert traj[1].sum() == approx(traj_val_dict[ob][1], abs=1e-5)


def test_xlrp_event():
    parameters_ti = prlx_xlrp_ti_params.copy()
    event_ti = FitUtils.build_mylens_event(event_ri, params_dict=parameters_ti)
    assert event_ti.get_chi2() == approx(1219.54, abs=1e-2)
    parameters_cpb = prlx_xlrp_cpb_params.copy()
    event_cpb = FitUtils.build_mylens_event(event_ri, params_dict=parameters_cpb)
    assert event_cpb.get_chi2() == approx(event_ti.get_chi2(), abs=1e-5)
    traj_val_dict = {
        "ogle": (-13512.8407372212, 5363.9716472117),
        "ogleV": (-1263.208684521062, 501.702897316377),
        "spitzer": (17.552889710780, 2.235481552976),
    }
    for ob in event_ti.all_ob_tup:
        traj_ti = event_ti.model_dict[ob].get_trajectory()
        assert traj_ti[0].sum() == approx(traj_val_dict[ob][0], abs=1e-2)
        assert traj_ti[1].sum() == approx(traj_val_dict[ob][1], abs=1e-2)
        traj_cpb = event_cpb.model_dict[ob].get_trajectory()
        assert traj_cpb[0].sum() == approx(traj_val_dict[ob][0], abs=1e-2)
        assert traj_cpb[1].sum() == approx(traj_val_dict[ob][1], abs=1e-2)


def test_xlrp_2s_event():
    parameters = prlx_xlrp_ti_2s_params.copy()
    event = FitUtils.build_mylens_event(event_ri, params_dict=parameters)
    assert event.get_chi2() == approx(1147.07, abs=1e-2)
    traj_val_dict = {
        "ogle": (
            -13881.225026150,
            -3274.526326952083,
            -13860.568071502576,
            -3304.328090353803,
        ),
        "ogleV": (
            -1297.1591685559247,
            -305.2553999500614,
            -1297.444827083918,
            -309.4423446538905,
        ),
        "spitzer": (
            15.90245362566319,
            10.32776872262049,
            -13.032211892919829,
            83.67394585962177,
        ),
    }
    for ob in event.all_ob_tup:
        traj = event.model_dict[ob].get_trajectory()
        assert traj[0].sum() == approx(traj_val_dict[ob][0], abs=1e-2)
        assert traj[1].sum() == approx(traj_val_dict[ob][1], abs=1e-2)
        assert traj[2].sum() == approx(traj_val_dict[ob][2], abs=1e-2)
        assert traj[3].sum() == approx(traj_val_dict[ob][3], abs=1e-2)


def test_xlrp_cpb_ecc0_event():
    # when eccentricity is zero, the model should be the same as circular orbit
    parameters_ecc0 = prlx_xlrp_cpb_params.copy()
    parameters_ecc0["e_xi"] = 0.0
    parameters_ecc0["omega_xi"] = 0.0
    parameters_ecc0["Omega_xi"] = 0.0
    event_ecc = FitUtils.build_mylens_event(event_ri, params_dict=parameters_ecc0)
    parameters_cir = prlx_xlrp_cpb_params.copy()
    event_cir = FitUtils.build_mylens_event(event_ri, params_dict=parameters_cir)
    assert event_ecc.get_chi2() == approx(event_cir.get_chi2(), abs=1e-2)
    traj_val_dict = {
        "ogle": (-13512.8407372212, 5363.9716472117),
        "ogleV": (-1263.208684521062, 501.702897316377),
        "spitzer": (17.552889710780, 2.235481552976),
    }
    for ob in event_ecc.all_ob_tup:
        traj = event_ecc.model_dict[ob].get_trajectory()
        assert traj[0].sum() == approx(traj_val_dict[ob][0], abs=1e-2)
        assert traj[1].sum() == approx(traj_val_dict[ob][1], abs=1e-2)


def test_xlrp_cpb_ecc_event():
    # Now test when eccentricity is not zero
    parameters_ecc = prlx_xlrp_cpb_ecc_params.copy()
    event_ecc = FitUtils.build_mylens_event(event_ri, params_dict=parameters_ecc)
    assert event_ecc.get_chi2() == approx(1194.34, abs=1e-2)
    traj_val_dict = {
        "ogle": (-13439.429396478256, 7527.865202395117),
        "ogleV": (-1256.4089113477276, 703.6253871550258),
        "spitzer": (18.372760061945097, 1.3829141918687093),
    }
    for ob in event_ecc.all_ob_tup:
        traj = event_ecc.model_dict[ob].get_trajectory()
        assert traj[0].sum() == approx(traj_val_dict[ob][0], abs=1e-2)
        assert traj[1].sum() == approx(traj_val_dict[ob][1], abs=1e-2)
