from xlrp import PointLensModel
from pytest import approx


def test_init_std_model():
    parameters = {"t_0": 2450000, "u_0": 0.1, "t_E": 10}
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
    parameters = {
        "t_0": 2450000,
        "u_0": 0.1,
        "t_E": 10,
        "pi_E_N": 0.1,
        "pi_E_E": 0.1,
        "t_0_par": 2450000,
    }
    dec = "-31:34:50.0"
    ra = "18:04:21.29"
    model = PointLensModel(parameters, ra=ra, dec=dec)
    assert model.parameter_set_enabled == ["std", "prlx"]
    assert model.t_0_par is 2450000
    assert model.ra is ra
    assert model.dec is dec
    assert model.alpha == approx(271.0887083)
    assert model.delta == approx(-31.5805556)


def test_init_xlrp_model():
    parameters = {
        "t_0": 2450000,
        "u_0": 0.1,
        "t_E": 10,
        "pi_E_N": 0.1,
        "pi_E_E": 0.1,
        "p_xi": 100.0,
        "A_xi": 0.1,
        "B_xi": 0.1,
        "F_xi": 0.1,
        "G_xi": 0.1,
        "t_ref": 2450000.0,
        "t_0_par": 2450000.0,
    }
    dec = "-31:34:50.0"
    ra = "18:04:21.29"
    model = PointLensModel(parameters, ra=ra, dec=dec)
    assert model.parameter_set_enabled == ["std", "prlx", "xlrp_circ_ti"]
