import yaml
from copy import deepcopy

from xlrp.utils.config import ReadInfo


class ParamFile(object):
    def __init__(self, param_file_path: str) -> None:
        self.param_file_path = param_file_path

        with open(self.param_file_path, "r", encoding="utf-8") as f:
            self.param_info = yaml.load(f.read(), Loader=yaml.FullLoader)

        return


class ReadParam(ParamFile):
    # fmt: off
    def get_params(
        self,
        event_type: str,
        include_t_0_par: bool = False,
        include_t_ref: bool = False,
        neg_u0: bool = False,
    ):
        if event_type == "std":
            if not neg_u0:
                return self.param_info["Standard_Parameters"]["pos_u_0"]["params_dict"].copy()
            else:
                return self.param_info["Standard_Parameters"]["neg_u_0"]["params_dict"].copy()
        elif event_type == "prlx":
            if not neg_u0:
                params_dict = self.param_info["Parallax_Parameters"]["pos_u_0"]["params_dict"].copy()
            else:
                params_dict = self.param_info["Parallax_Parameters"]["neg_u_0"]["params_dict"].copy()
            if not include_t_0_par:
                return params_dict
            else:
                params_dict["t_0_par"] = self.get_t_0_par()
                return params_dict
        elif event_type == "xlrp_circ":
            if not neg_u0:
                params_dict = self.param_info["Xallarap_Circular_Parameters"]["pos_u_0"]["params_dict"].copy()
            else:
                params_dict = self.param_info["Xallarap_Circular_Parameters"]["neg_u_0"]["params_dict"].copy()
            if not include_t_ref:
                return params_dict
            else:
                params_dict["t_ref"] = self.get_t_ref()
                return params_dict
        elif event_type == "xlrp_circ_ti":
            if not neg_u0:
                params_dict = self.param_info["Xallarap_Circular_Thiele_Innes_Parameters"]["pos_u_0"]["params_dict"].copy()
            else:
                params_dict = self.param_info["Xallarap_Circular_Thiele_Innes_Parameters"]["neg_u_0"]["params_dict"].copy()
            if not include_t_ref:
                return params_dict
            else:
                params_dict["t_ref"] = self.get_t_ref()
                return params_dict
        elif event_type == "xlrp_cpb":
            if not neg_u0:
                params_dict = self.param_info["Xallarap_Campbell_Parameters"]["pos_u_0"]["params_dict"].copy()
            else:
                params_dict = self.param_info["Xallarap_Campbell_Parameters"]["neg_u_0"]["params_dict"].copy()
            if not include_t_ref:
                return params_dict
            else:
                params_dict["t_ref"] = self.get_t_ref()
                return params_dict
        elif event_type == "xlrp_ti":
            if not neg_u0:
                params_dict = self.param_info["Xallarap_Thiele_Innes_Parameters"]["pos_u_0"]["params_dict"].copy()
            else:
                params_dict = self.param_info["Xallarap_Thiele_Innes_Parameters"]["neg_u_0"]["params_dict"].copy()
            if not include_t_ref:
                return params_dict
            else:
                params_dict["t_ref"] = self.get_t_ref()
                return params_dict
        else:
            raise ValueError("event_type must be 'std', 'prlx', 'xlrp_circ', 'xlrp_circ_ti', 'xlrp_cpb', or 'xlrp_ti'")

    def get_params_err(self, event_type: str, neg_u0: bool = False):
        if event_type == "std":
            if not neg_u0:
                return self.param_info["Standard_Parameters"]["pos_u_0"]["params_err_dict"].copy()
            else:
                return self.param_info["Standard_Parameters"]["neg_u_0"]["params_err_dict"].copy()
        elif event_type == "prlx":
            if not neg_u0:
                return self.param_info["Parallax_Parameters"]["pos_u_0"]["params_err_dict"].copy()
            else:
                return self.param_info["Parallax_Parameters"]["neg_u_0"]["params_err_dict"].copy()
        elif event_type == "xlrp_circ":
            if not neg_u0:
                return self.param_info["Xallarap_Circular_Parameters"]["pos_u_0"]["params_err_dict"].copy()
            else:
                return self.param_info["Xallarap_Circular_Parameters"]["neg_u_0"]["params_err_dict"].copy()
        elif event_type == "xlrp_circ_ti":
            if not neg_u0:
                return self.param_info["Xallarap_Circular_Thiele_Innes_Parameters"]["pos_u_0"]["params_err_dict"].copy()
            else:
                return self.param_info["Xallarap_Circular_Thiele_Innes_Parameters"]["neg_u_0"]["params_err_dict"].copy()
        elif event_type == "xlrp_cpb":
            if not neg_u0:
                return self.param_info["Xallarap_Campbell_Parameters"]["pos_u_0"]["params_err_dict"].copy()
            else:
                return self.param_info["Xallarap_Campbell_Parameters"]["neg_u_0"]["params_err_dict"].copy()
        elif event_type == "xlrp_ti":
            if not neg_u0:
                return self.param_info["Xallarap_Thiele_Innes_Parameters"]["pos_u_0"]["params_err_dict"].copy()
            else:
                return self.param_info["Xallarap_Thiele_Innes_Parameters"]["neg_u_0"]["params_err_dict"].copy()
        else:
            raise ValueError("event_type must be 'std', 'prlx', 'xlrp_circ', 'xlrp_circ_ti', 'xlrp_cpb', or 'xlrp_ti'")

    def get_flux_params(self, event_type: str, neg_u0: bool = False):
        if event_type == "std":
            if not neg_u0:
                return self.param_info["Standard_Parameters"]["pos_u_0"]["flux_params_dict"].copy()
            else:
                return self.param_info["Standard_Parameters"]["neg_u_0"]["flux_params_dict"].copy()
        elif event_type == "prlx":
            if not neg_u0:
                return self.param_info["Parallax_Parameters"]["pos_u_0"]["flux_params_dict"].copy()
            else:
                return self.param_info["Parallax_Parameters"]["neg_u_0"]["flux_params_dict"].copy()
        elif event_type == "xlrp_circ":
            if not neg_u0:
                return self.param_info["Xallarap_Circular_Parameters"]["pos_u_0"]["flux_params_dict"].copy()
            else:
                return self.param_info["Xallarap_Circular_Parameters"]["neg_u_0"]["flux_params_dict"].copy()
        elif event_type == "xlrp_circ_ti":
            if not neg_u0:
                return self.param_info["Xallarap_Circular_Thiele_Innes_Parameters"]["pos_u_0"]["flux_params_dict"].copy()
            else:
                return self.param_info["Xallarap_Circular_Thiele_Innes_Parameters"]["neg_u_0"]["flux_params_dict"].copy()
        elif event_type == "xlrp_cpb":
            if not neg_u0:
                return self.param_info["Xallarap_Campbell_Parameters"]["pos_u_0"]["flux_params_dict"].copy()
            else:
                return self.param_info["Xallarap_Campbell_Parameters"]["neg_u_0"]["flux_params_dict"].copy()
        elif event_type == "xlrp_ti":
            if not neg_u0:
                return self.param_info["Xallarap_Thiele_Innes_Parameters"]["pos_u_0"]["flux_params_dict"].copy()
            else:
                return self.param_info["Xallarap_Thiele_Innes_Parameters"]["neg_u_0"]["flux_params_dict"].copy()
        else:
            raise ValueError("event_type must be 'std', 'prlx', 'xlrp_circ', 'xlrp_circ_ti', 'xlrp_cpb', or 'xlrp_ti'")

    def get_chi2(self, event_type: str, neg_u0: bool = False):
        if event_type == "std":
            if not neg_u0:
                return self.param_info["Standard_Parameters"]["pos_u_0"]["chi2"]
            else:
                return self.param_info["Standard_Parameters"]["neg_u_0"]["chi2"]
        elif event_type == "prlx":
            if not neg_u0:
                return self.param_info["Parallax_Parameters"]["pos_u_0"]["chi2"]
            else:
                return self.param_info["Parallax_Parameters"]["neg_u_0"]["chi2"]
        elif event_type == "xlrp_circ":
            if not neg_u0:
                return self.param_info["Xallarap_Circular_Parameters"]["pos_u_0"]["chi2"]
            else:
                return self.param_info["Xallarap_Circular_Parameters"]["neg_u_0"]["chi2"]
        elif event_type == "xlrp_circ_ti":
            if not neg_u0:
                return self.param_info["Xallarap_Circular_Thiele_Innes_Parameters"]["pos_u_0"]["chi2"]
            else:
                return self.param_info["Xallarap_Circular_Thiele_Innes_Parameters"]["neg_u_0"]["chi2"]
        elif event_type == "xlrp_cpb":
            if not neg_u0:
                return self.param_info["Xallarap_Campbell_Parameters"]["pos_u_0"]["chi2"]
            else:
                return self.param_info["Xallarap_Campbell_Parameters"]["neg_u_0"]["chi2"]
        elif event_type == "xlrp_ti":
            if not neg_u0:
                return self.param_info["Xallarap_Thiele_Innes_Parameters"]["pos_u_0"]["chi2"]
            else:
                return self.param_info["Xallarap_Thiele_Innes_Parameters"]["neg_u_0"]["chi2"]
        else:
            raise ValueError("event_type must be 'std', 'prlx', 'xlrp_circ', 'xlrp_circ_ti', 'xlrp_cpb', or 'xlrp_ti'")

    def get_t_0_par(self):
        return self.param_info["Parallax_Parameters"]["t_0_par"]

    def get_t_ref(self, event_type="xlrp_cpb"):
        if event_type == "xlrp_cpb":
            return self.param_info["Xallarap_Campbell_Parameters"]["t_ref"]
        return self.param_info["Xallarap_Campbell_Parameters"]["t_ref"]


class WriteParam(ParamFile):
    def save_yaml(self):
        with open(self.param_file_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                self.param_info, f, default_flow_style=False, allow_unicode=True
            )

    def _check_params_dict(self, event_type: str, params_dict: dict):
        if event_type == "std":
            for param_name in ["t_0", "u_0", "t_E"]:
                assert param_name in params_dict, f"{param_name} is not in params_dict"
            assert len(params_dict) == 3, "params_dict should have 3 keys"
        elif event_type == "prlx":
            param_names = ["t_0", "u_0", "t_E", "pi_E_N", "pi_E_E"]
            for param_name in param_names:
                assert param_name in params_dict, f"{param_name} is not in params_dict"
            assert len(params_dict) == 5, "params_dict should have 5 keys"
        elif event_type == "xlrp_circ":
            param_names = [
                "t_0",
                "u_0",
                "t_E",
                "p_xi",
                "phi_xi",
                "i_xi",
                "xi_E_N",
                "xi_E_E",
            ]
            for param_name in param_names:
                assert param_name in params_dict, f"{param_name} is not in params_dict"
            assert len(params_dict) == 8, "params_dict should have 8 keys"
        elif event_type == "xlrp_circ_ti":
            param_names = ["t_0", "u_0", "t_E", "p_xi", "A_xi", "B_xi", "F_xi", "G_xi"]
            for param_name in param_names:
                assert param_name in params_dict, f"{param_name} is not in params_dict"
            assert len(params_dict) == 8, "params_dict should have 8 keys"
        elif event_type == "xlrp_cpb":
            param_names = ["t_0", "u_0", "t_E", 
                           "e_xi", "p_xi", "phi_xi", "i_xi", "omega_xi", "Omega_xi",
                           "xi_E_N", "xi_E_E"]  # fmt: skip
            for param_name in param_names:
                assert param_name in params_dict, f"{param_name} is not in params_dict"
            assert len(params_dict) == 11, "params_dict should have 11 keys"
        elif event_type == "xlrp_ti":
            param_names = [
                "t_0",
                "u_0",
                "t_E",
                "e_xi",
                "p_xi",
                "phi_xi",
                "A_xi",
                "B_xi",
                "F_xi",
                "G_xi",
                "theta_xi",
            ]
            for param_name in param_names:
                assert param_name in params_dict, f"{param_name} is not in params_dict"
            assert len(params_dict) == 11, "params_dict should have 11 keys"
        else:
            raise ValueError(
                "event_type must be 'std', 'prlx', 'xlrp_circ', 'xlrp_circ_ti', 'xlrp_cpb', or 'xlrp_ti'"
            )

    def set_params(self, params_dict: dict, event_type: str, neg_u0: bool = False):
        self._check_params_dict(event_type, params_dict)
        if event_type == "std":
            if not neg_u0:
                self.param_info["Standard_Parameters"]["pos_u_0"]["params_dict"].update(
                    params_dict
                )
            else:
                self.param_info["Standard_Parameters"]["neg_u_0"]["params_dict"].update(
                    params_dict
                )
        elif event_type == "prlx":
            if not neg_u0:
                self.param_info["Parallax_Parameters"]["pos_u_0"]["params_dict"].update(
                    params_dict
                )
            else:
                self.param_info["Parallax_Parameters"]["neg_u_0"]["params_dict"].update(
                    params_dict
                )
        elif event_type == "xlrp_circ":
            if not neg_u0:
                self.param_info["Xallarap_Circular_Parameters"]["pos_u_0"][
                    "params_dict"
                ].update(params_dict)
            else:
                self.param_info["Xallarap_Circular_Parameters"]["neg_u_0"][
                    "params_dict"
                ].update(params_dict)
        elif event_type == "xlrp_circ_ti":
            if not neg_u0:
                self.param_info["Xallarap_Circular_Thiele_Innes_Parameters"]["pos_u_0"][
                    "params_dict"
                ].update(params_dict)
            else:
                self.param_info["Xallarap_Circular_Thiele_Innes_Parameters"]["neg_u_0"][
                    "params_dict"
                ].update(params_dict)
        elif event_type == "xlrp_cpb":
            if not neg_u0:
                self.param_info["Xallarap_Campbell_Parameters"]["pos_u_0"][
                    "params_dict"
                ].update(params_dict)
            else:
                self.param_info["Xallarap_Campbell_Parameters"]["neg_u_0"][
                    "params_dict"
                ].update(params_dict)
        elif event_type == "xlrp_ti":
            if not neg_u0:
                self.param_info["Xallarap_Thiele_Innes_Parameters"]["pos_u_0"][
                    "params_dict"
                ].update(params_dict)
            else:
                self.param_info["Xallarap_Thiele_Innes_Parameters"]["neg_u_0"][
                    "params_dict"
                ].update(params_dict)
        else:
            raise ValueError(
                "event_type must be 'std', 'prlx', 'xlrp_circ', 'xlrp_circ_ti', 'xlrp_cpb', or 'xlrp_ti'"
            )
        self.save_yaml()
        return

    def _check_err_dict(self, event_type, err_dict):
        # TODO: Now just for xallarap with Compbell elements
        if event_type == "std":
            for err_name in ["t_0_err", "u_0_err", "t_E_err"]:
                assert err_name in err_dict, f"{err_name} is not in params_err_dict"
            assert len(err_dict) == 3, "params_err_dict should have 3 keys"
        elif event_type == "prlx":
            err_names = ["t_0_err", "u_0_err", "t_E_err", "pi_E_N_err", "pi_E_E_err"]
            for err_name in err_names:
                assert err_name in err_dict, f"{err_name} is not in params_err_dict"
            assert len(err_dict) == 5, "params_err_dict should have 5 keys"
        elif event_type == "xlrp_circ":
            err_names = [
                "t_0_err",
                "u_0_err",
                "t_E_err",
                "p_xi_err",
                "phi_xi_err",
                "i_xi_err",
                "xi_E_N_err",
                "xi_E_E_err",
            ]
            for err_name in err_names:
                assert err_name in err_dict, f"{err_name} is not in params_err_dict"
            assert len(err_dict) == 8, "params_err_dict should have 8 keys"
        elif event_type == "xlrp_circ_ti":
            err_names = [
                "t_0_err",
                "u_0_err",
                "t_E_err",
                "p_xi_err",
                "A_xi_err",
                "B_xi_err",
                "F_xi_err",
                "G_xi_err",
            ]
            for err_name in err_names:
                assert err_name in err_dict, f"{err_name} is not in params_err_dict"
            assert len(err_dict) == 8, "params_err_dict should have 8 keys"
        elif event_type == "xlrp_cpb":
            err_names = ["t_0_err", "u_0_err", "t_E_err", 
                         "e_xi_err", "p_xi_err", "phi_xi_err", "i_xi_err", "omega_xi_err",
                         "Omega_xi_err", "xi_E_N_err", "xi_E_E_err"]  # fmt: skip
            for err_name in err_names:
                assert err_name in err_dict, f"{err_name} is not in params_err_dict"
            assert len(err_dict) == 11, "params_err_dict should have 11 keys"
        elif event_type == "xlrp_ti":
            err_names = [
                "t_0_err",
                "u_0_err",
                "t_E_err",
                "e_xi_err",
                "p_xi_err",
                "phi_xi_err",
                "A_xi_err",
                "B_xi_err",
                "F_xi_err",
                "G_xi_err",
                "theta_xi_err",
            ]
            for err_name in err_names:
                assert err_name in err_dict, f"{err_name} is not in params_err_dict"
            assert len(err_dict) == 11, "params_err_dict should have 11 keys"

        else:
            raise ValueError(
                "event_type must be 'std', 'prlx', 'xlrp_circ', 'xlrp_circ_ti', 'xlrp_cpb', or 'xlrp_ti'"
            )

    def set_err(self, err_dict: dict, event_type: str, neg_u0: bool = False):
        self._check_err_dict(event_type, err_dict)
        if event_type == "std":
            if not neg_u0:
                self.param_info["Standard_Parameters"]["pos_u_0"][
                    "params_err_dict"
                ] = err_dict.copy()
            else:
                self.param_info["Standard_Parameters"]["neg_u_0"][
                    "params_err_dict"
                ] = err_dict.copy()
        elif event_type == "prlx":
            if not neg_u0:
                self.param_info["Parallax_Parameters"]["pos_u_0"][
                    "params_err_dict"
                ] = err_dict.copy()
            else:
                self.param_info["Parallax_Parameters"]["neg_u_0"][
                    "params_err_dict"
                ] = err_dict.copy()
        elif event_type == "xlrp_circ":
            if not neg_u0:
                self.param_info["Xallarap_Circular_Parameters"]["pos_u_0"][
                    "params_err_dict"
                ] = err_dict.copy()
            else:
                self.param_info["Xallarap_Circular_Parameters"]["neg_u_0"][
                    "params_err_dict"
                ] = err_dict.copy()
        elif event_type == "xlrp_circ_ti":
            if not neg_u0:
                self.param_info["Xallarap_Circular_Thiele_Innes_Parameters"]["pos_u_0"][
                    "params_err_dict"
                ] = err_dict.copy()
            else:
                self.param_info["Xallarap_Circular_Thiele_Innes_Parameters"]["neg_u_0"][
                    "params_err_dict"
                ] = err_dict.copy()
        elif event_type == "xlrp_cpb":
            if not neg_u0:
                self.param_info["Xallarap_Campbell_Parameters"]["pos_u_0"][
                    "params_err_dict"
                ] = err_dict.copy()
            else:
                self.param_info["Xallarap_Campbell_Parameters"]["neg_u_0"][
                    "params_err_dict"
                ] = err_dict.copy()
        elif event_type == "xlrp_ti":
            if not neg_u0:
                self.param_info["Xallarap_Thiele_Innes_Parameters"]["pos_u_0"][
                    "params_err_dict"
                ] = err_dict.copy()
            else:
                self.param_info["Xallarap_Thiele_Innes_Parameters"]["neg_u_0"][
                    "params_err_dict"
                ] = err_dict.copy()
        else:
            raise ValueError(
                "event_type must be 'std', 'prlx', 'xlrp_circ', 'xirp_circ_ti', 'xlrp_cpb', or 'xlrp_ti'"
            )
        self.save_yaml()
        return

    def _check_flux_dict(self, flux_dict: dict):
        for flux_name in ["f_s", "f_b"]:
            assert flux_name in flux_dict, f"{flux_name} is not in flux_dict"
        assert len(flux_dict) == 2, "flux_dict should have 2 keys"

    def set_flux_chi2(
        self, flux_dict: dict, chi2: float, event_type: str, neg_u0: bool = False
    ):
        if event_type == "std":
            if not neg_u0:
                self.param_info["Standard_Parameters"]["pos_u_0"][
                    "flux_params_dict"
                ] = flux_dict.copy()
                self.param_info["Standard_Parameters"]["pos_u_0"]["chi2"] = chi2
            else:
                self.param_info["Standard_Parameters"]["neg_u_0"][
                    "flux_params_dict"
                ] = flux_dict.copy()
                self.param_info["Standard_Parameters"]["neg_u_0"]["chi2"] = chi2
        elif event_type == "prlx":
            if not neg_u0:
                self.param_info["Parallax_Parameters"]["pos_u_0"][
                    "flux_params_dict"
                ] = flux_dict.copy()
                self.param_info["Parallax_Parameters"]["pos_u_0"]["chi2"] = chi2
            else:
                self.param_info["Parallax_Parameters"]["neg_u_0"][
                    "flux_params_dict"
                ] = flux_dict.copy()
                self.param_info["Parallax_Parameters"]["neg_u_0"]["chi2"] = chi2
        elif event_type == "xlrp_circ":
            if not neg_u0:
                self.param_info["Xallarap_Circular_Parameters"]["pos_u_0"][
                    "flux_params_dict"
                ] = flux_dict.copy()
                self.param_info["Xallarap_Circular_Parameters"]["pos_u_0"][
                    "chi2"
                ] = chi2
            else:
                self.param_info["Xallarap_Circular_Parameters"]["neg_u_0"][
                    "flux_params_dict"
                ] = flux_dict.copy()
                self.param_info["Xallarap_Circular_Parameters"]["neg_u_0"][
                    "chi2"
                ] = chi2
        elif event_type == "xlrp_circ_ti":
            if not neg_u0:
                self.param_info["Xallarap_Circular_Thiele_Innes_Parameters"]["pos_u_0"][
                    "flux_params_dict"
                ] = flux_dict.copy()
                self.param_info["Xallarap_Circular_Thiele_Innes_Parameters"]["pos_u_0"][
                    "chi2"
                ] = chi2
            else:
                self.param_info["Xallarap_Circular_Thiele_Innes_Parameters"]["neg_u_0"][
                    "flux_params_dict"
                ] = flux_dict.copy()
                self.param_info["Xallarap_Circular_Thiele_Innes_Parameters"]["neg_u_0"][
                    "chi2"
                ] = chi2
        elif event_type == "xlrp_cpb":
            if not neg_u0:
                self.param_info["Xallarap_Campbell_Parameters"]["pos_u_0"][
                    "flux_params_dict"
                ] = flux_dict.copy()
                self.param_info["Xallarap_Campbell_Parameters"]["pos_u_0"][
                    "chi2"
                ] = chi2
            else:
                self.param_info["Xallarap_Campbell_Parameters"]["neg_u_0"][
                    "flux_params_dict"
                ] = flux_dict.copy()
                self.param_info["Xallarap_Campbell_Parameters"]["neg_u_0"][
                    "chi2"
                ] = chi2
        elif event_type == "xlrp_ti":
            if not neg_u0:
                self.param_info["Xallarap_Thiele_Innes_Parameters"]["pos_u_0"][
                    "flux_params_dict"
                ] = flux_dict.copy()
                self.param_info["Xallarap_Thiele_Innes_Parameters"]["pos_u_0"][
                    "chi2"
                ] = chi2
            else:
                self.param_info["Xallarap_Thiele_Innes_Parameters"]["neg_u_0"][
                    "flux_params_dict"
                ] = flux_dict.copy()
                self.param_info["Xallarap_Thiele_Innes_Parameters"]["neg_u_0"][
                    "chi2"
                ] = chi2
        else:
            raise ValueError(
                "event_type must be 'std', 'prlx', 'xlrp_circ', 'xlrp_cpb', or 'xlrp_ti'"
            )
        self.save_yaml()
        return


# fmt: on


def init_param_file(event_ri: ReadInfo, std_params_dict: dict, p_init=20.0):
    ob_name_list = event_ri.get_observatories()

    # basic flux parameters dictionary
    flux_params_dict = {}
    for ob_name in ob_name_list:
        flux_params_dict[ob_name] = {"f_s": 1.0, "f_b": 0.0}

    # standard parameters dictionary
    std_params_err_dict = {}
    for params_name in std_params_dict:
        std_params_err_dict[params_name + "_err"] = 0.0
    std_neg_params_dict = std_params_dict.copy()
    std_neg_params_dict["u_0"] = -std_neg_params_dict["u_0"]
    std_full_dict = {
        "pos_u_0": {
            "chi2": 1e9,
            "flux_params_dict": deepcopy(flux_params_dict),
            "params_dict": std_params_dict,
            "params_err_dict": deepcopy(std_params_err_dict),
        },
        "neg_u_0": {
            "chi2": 1e9,
            "flux_params_dict": deepcopy(flux_params_dict),
            "params_dict": std_neg_params_dict,
            "params_err_dict": deepcopy(std_params_err_dict),
        },
    }

    # parallax parameters dictionary
    prlx_params_dict = {**std_params_dict, "pi_E_N": 0.0, "pi_E_E": 0.0}
    prlx_params_err_dict = {}
    for params_name in prlx_params_dict:
        prlx_params_err_dict[params_name + "_err"] = 0.0
    prlx_neg_params_dict = deepcopy(prlx_params_dict)
    prlx_neg_params_dict["u_0"] = -prlx_neg_params_dict["u_0"]
    prlx_full_dict = {
        "pos_u_0": {
            "chi2": 1e9,
            "flux_params_dict": deepcopy(flux_params_dict),
            "params_dict": prlx_params_dict,
            "params_err_dict": deepcopy(prlx_params_err_dict),
        },
        "neg_u_0": {
            "chi2": 1e9,
            "flux_params_dict": deepcopy(flux_params_dict),
            "params_dict": prlx_neg_params_dict,
            "params_err_dict": deepcopy(prlx_params_err_dict),
        },
        "t_0_par": prlx_params_dict["t_0"],
    }

    # Xallarap Circular parameters dictionary
    xlrp_circ_params_dict = {
        **std_params_dict,
        "xi_E_N": 0.0,
        "xi_E_E": 0.0,
        "i_xi": 0.0,
        "p_xi": p_init,
        "phi_xi": 0.0,
    }
    xlrp_circ_err_dict = {}
    for params_name in xlrp_circ_params_dict:
        xlrp_circ_err_dict[params_name + "_err"] = 0.0
    xlrp_circ_neg_params_dict = deepcopy(xlrp_circ_params_dict)
    xlrp_circ_neg_params_dict["u_0"] = -xlrp_circ_neg_params_dict["u_0"]
    xlrp_circ_full_dict = {
        "pos_u_0": {
            "chi2": 1e9,
            "flux_params_dict": deepcopy(flux_params_dict),
            "params_dict": xlrp_circ_params_dict,
            "params_err_dict": deepcopy(xlrp_circ_err_dict),
        },
        "neg_u_0": {
            "chi2": 1e9,
            "flux_params_dict": deepcopy(flux_params_dict),
            "params_dict": xlrp_circ_neg_params_dict,
            "params_err_dict": deepcopy(xlrp_circ_err_dict),
        },
        "t_ref": xlrp_circ_params_dict["t_0"],
    }

    # Xallarap Circular Thiele-Innes parameters dictionary
    xlrp_circ_ti_params_dict = {
        **std_params_dict,
        "p_xi": p_init,
        "A_xi": 0.0,
        "B_xi": 0.0,
        "F_xi": 0.0,
        "G_xi": 0.0,
    }
    xlrp_circ_ti_err_dict = {}
    for params_name in xlrp_circ_ti_params_dict:
        xlrp_circ_ti_err_dict[params_name + "_err"] = 0.0
    xlrp_circ_ti_neg_params_dict = deepcopy(xlrp_circ_ti_params_dict)
    xlrp_circ_ti_neg_params_dict["u_0"] = -xlrp_circ_ti_neg_params_dict["u_0"]
    xlrp_circ_ti_full_dict = {
        "pos_u_0": {
            "chi2": 1e9,
            "flux_params_dict": deepcopy(flux_params_dict),
            "params_dict": xlrp_circ_ti_params_dict,
            "params_err_dict": deepcopy(xlrp_circ_ti_err_dict),
        },
        "neg_u_0": {
            "chi2": 1e9,
            "flux_params_dict": deepcopy(flux_params_dict),
            "params_dict": xlrp_circ_ti_neg_params_dict,
            "params_err_dict": deepcopy(xlrp_circ_ti_err_dict),
        },
        "t_ref": xlrp_circ_ti_params_dict["t_0"],
    }

    # Xallarap Campbell parameters dictionary
    xlrp_cpb_params_dict = {
        **std_params_dict,
        "xi_E_N": 0.0,
        "xi_E_E": 0.0,
        "e_xi": 0.1,
        "p_xi": p_init,
        "omega_xi": 0.0,
        "Omega_xi": 0.0,
        "i_xi": 0.0,
        "phi_xi": 0.0,
    }
    xlrp_cpb_params_err_dict = {}
    for params_name in xlrp_cpb_params_dict:
        xlrp_cpb_params_err_dict[params_name + "_err"] = 0.0
    xlrp_cpb_neg_params_dict = deepcopy(xlrp_cpb_params_dict)
    xlrp_cpb_neg_params_dict["u_0"] = -xlrp_cpb_neg_params_dict["u_0"]
    xlrp_cpb_full_dict = {
        "pos_u_0": {
            "chi2": 1e9,
            "flux_params_dict": deepcopy(flux_params_dict),
            "params_dict": xlrp_cpb_params_dict,
            "params_err_dict": deepcopy(xlrp_cpb_params_err_dict),
        },
        "neg_u_0": {
            "chi2": 1e9,
            "flux_params_dict": deepcopy(flux_params_dict),
            "params_dict": xlrp_cpb_neg_params_dict,
            "params_err_dict": deepcopy(xlrp_cpb_params_err_dict),
        },
        "t_ref": xlrp_circ_params_dict["t_0"],
    }

    # Xallarap Thiele-Innes parameters dictionary
    xlrp_ti_params_dict = {
        **std_params_dict,
        "xi_E_N": 0.0,
        "xi_E_E": 0.0,
        "e_xi": 0.1,
        "p_xi": p_init,
        "A_xi": 0.0,
        "B_xi": 0.0,
        "F_xi": 0.0,
        "G_xi": 0.0,
        "theta_xi": 0.0,
    }
    xlrp_ti_params_err_dict = {}
    for params_name in xlrp_ti_params_dict:
        xlrp_ti_params_err_dict[params_name + "_err"] = 0.0
    xlrp_ti_neg_params_dict = deepcopy(xlrp_ti_params_dict)
    xlrp_ti_neg_params_dict["u_0"] = -xlrp_ti_neg_params_dict["u_0"]
    xlrp_ti_full_dict = {
        "pos_u_0": {
            "chi2": 1e9,
            "flux_params_dict": deepcopy(flux_params_dict),
            "params_dict": xlrp_ti_params_dict,
            "params_err_dict": deepcopy(xlrp_ti_params_err_dict),
        },
        "neg_u_0": {
            "chi2": 1e9,
            "flux_params_dict": deepcopy(flux_params_dict),
            "params_dict": xlrp_ti_neg_params_dict,
            "params_err_dict": deepcopy(xlrp_ti_params_err_dict),
        },
        "t_ref": xlrp_circ_params_dict["t_0"],
    }

    full_dict = {
        "Standard_Parameters": std_full_dict,
        "Parallax_Parameters": prlx_full_dict,
        "Xallarap_Circular_Parameters": xlrp_circ_full_dict,
        "Xallarap_Circular_Thiele_Innes_Parameters": xlrp_circ_ti_full_dict,
        "Xallarap_Campbell_Parameters": xlrp_cpb_full_dict,
        "Xallarap_Thiele_Innes_Parameters": xlrp_ti_full_dict,
    }
    with open(event_ri.get_param_path(), "w", encoding="utf-8") as f:
        yaml.safe_dump(full_dict, f, default_flow_style=False, allow_unicode=True)

    return


def init_param_file_full(
    param_file_path: str,
    std_params_dict: dict,
    prlx_params_dict: dict,
    xlrp_circ_params_dict: dict,
    xlrp_circ_ti_params_dict: dict,
    xlrp_cpb_params_dict: dict,
    xlrp_ti_params_dict: dict,
    std_params_err: dict | None = None,
    std_flux_dict: dict | None = None,
    std_chi2: float | None = None,
    prlx_params_err: dict | None = None,
    prlx_flux_dict: dict | None = None,
    prlx_chi2: float | None = None,
    xlrp_circ_params_err: dict | None = None,
    xlrp_circ_flux_dict: dict | None = None,
    xlrp_circ_chi2: float | None = None,
    xlrp_circ_ti_params_err: dict | None = None,
    xlrp_circ_ti_flux_dict: dict | None = None,
    xlrp_circ_ti_chi2: float | None = None,
    xlrp_cpb_params_err: dict | None = None,
    xlrp_cpb_flux_dict: dict | None = None,
    xlrp_cpb_chi2: float | None = None,
    xlrp_ti_params_err: dict | None = None,
    xlrp_ti_flux_dict: dict | None = None,
    xlrp_ti_chi2: float | None = None,
):
    event_params_info = {
        "Standard_Parameters": {
            "pos_u_0": {
                "params_dict": {"t_0": None, "u_0": None, "t_E": None},
                "params_err_dict": {"t_0_err": None, "u_0_err": None, "t_E_err": None},
                "flux_params_dict": {"f_s": None, "f_b": None},
                "chi2": None,
            },
            "neg_u_0": {
                "params_dict": {"t_0": None, "u_0": None, "t_E": None},
                "params_err_dict": {"t_0_err": None, "u_0_err": None, "t_E_err": None},
                "flux_params_dict": {"f_s": None, "f_b": None},
                "chi2": None,
            }
        },
        "Parallax_Parameters": {
            "pos_u_0": {
                "params_dict": {"t_0": None, "u_0": None, "t_E": None,
                                "pi_E_N": None, "pi_E_E": None},
                "params_err_dict": {"t_0_err": None, "u_0_err": None, "t_E_err": None,
                                    "pi_E_N_err": None, "pi_E_E_err": None},
                "flux_params_dict": {"f_s": None, "f_b": None},
                "chi2": None,
            },
            "neg_u_0": {
                "params_dict": {"t_0": None, "u_0": None, "t_E": None,
                                "pi_E_N": None, "pi_E_E": None},
                "params_err_dict": {"t_0_err": None, "u_0_err": None, "t_E_err": None,
                                    "pi_E_N_err": None, "pi_E_E_err": None},
                "flux_params_dict": {"f_s": None, "f_b": None},
                "chi2": None,
            },
            "t_0_par": None,
        },
        "Xallarap_Circular_Parameters": {
            "pos_u_0": {
                "params_dict": {"t_0": None, "u_0": None, "t_E": None,
                                "p_xi": None, "phi_xi": None, "i_xi": None,
                                "xi_E_N": None, "xi_E_E": None},
                "params_err_dict": {"t_0_err": None, "u_0_err": None, "t_E_err": None,
                                    "p_xi_err": None, "phi_xi_err": None, "i_xi_err": None,
                                    "xi_E_N_err": None, "xi_E_E_err": None},
                "flux_params_dict": {"f_s": None, "f_b": None},
                "chi2": None,
            },
            "neg_u_0": {
                "params_dict": {"t_0": None, "u_0": None, "t_E": None,
                                "p_xi": None, "phi_xi": None, "i_xi": None,
                                "xi_E_N": None, "xi_E_E": None},
                "params_err_dict": {"t_0_err": None, "u_0_err": None, "t_E_err": None,
                                    "p_xi_err": None, "phi_xi_err": None, "i_xi_err": None,
                                    "xi_E_N_err": None, "xi_E_E_err": None},
                "flux_params_dict": {"f_s": None, "f_b": None},
                "chi2": None,
            },
            "t_ref": None,
        },
        "Xallarap_Circular_Thiele_Innes_Parameters": {
            "pos_u_0": {
                "params_dict": {"t_0": None, "u_0": None, "t_E": None,
                                "p_xi": None, "phi_xi": None, "A_xi": None,
                                "B_xi": None, "F_xi": None},
                "params_err_dict": {"t_0_err": None, "u_0_err": None, "t_E_err": None,
                                    "p_xi_err": None, "phi_xi_err": None, "A_xi_err": None,
                                    "B_xi_err": None, "F_xi_err": None},
                "flux_params_dict": {"f_s": None, "f_b": None},
                "chi2": None,
            },
            "neg_u_0": {
                "params_dict": {"t_0": None, "u_0": None, "t_E": None,
                                "p_xi": None, "phi_xi": None, "A_xi": None,
                                "B_xi": None, "F_xi": None},
                "params_err_dict": {"t_0_err": None, "u_0_err": None, "t_E_err": None,
                                    "p_xi_err": None, "phi_xi_err": None, "A_xi_err": None,
                                    "B_xi_err": None, "F_xi_err": None},
                "flux_params_dict": {"f_s": None, "f_b": None},
                "chi2": None,
            },
            "t_ref": None,
        },
        "Xallarap_Campbell_Parameters": {
            "pos_u_0": {
                "params_dict": {"t_0": None, "u_0": None, "t_E": None,
                                "e_xi": None, "p_xi": None, "phi_xi": None, "i_xi": None,
                                "omega_xi": None, "Omega_xi":None, "xi_E_N": None, "xi_E_E": None},
                "params_err_dict": {"t_0_err": None, "u_0_err": None, "t_E_err": None,
                                    "e_xi_err": None, "p_xi_err": None, "phi_xi_err": None,
                                    "i_xi_err": None, "omega_xi_err": None, "Omega_xi_err": None,
                                    "xi_E_N_err": None, "xi_E_E_err": None},
                "flux_params_dict": {"f_s": None, "f_b": None},
                "chi2": None,
            },
            "neg_u_0": {
                "params_dict": {"t_0": None, "u_0": None, "t_E": None,
                                "e_xi": None, "p_xi": None, "phi_xi": None, "i_xi": None,
                                "omega_xi": None, "Omega_xi":None, "xi_E_N": None, "xi_E_E": None},
                "params_err_dict": {"t_0_err": None, "u_0_err": None, "t_E_err": None,
                                    "e_xi_err": None, "p_xi_err": None, "phi_xi_err": None,
                                    "i_xi_err": None, "omega_xi_err": None, "Omega_xi_err": None,
                                    "xi_E_N_err": None, "xi_E_E_err": None},
                "flux_params_dict": {"f_s": None, "f_b": None},
                "chi2": None,
            },
            "t_ref": None,
        },
        "Xallarap_Thiele_Innes_Parameters": {
            "pos_u_0": {
                "params_dict": {"t_0": None, "u_0": None, "t_E": None,
                                "e_xi": None, "p_xi": None, "phi_xi": None,
                                "A_xi": None, "B_xi": None, "F_xi": None, "G_xi": None,
                                "theta_xi":None},
                "params_err_dict": {"t_0_err": None, "u_0_err": None, "t_E_err": None,
                                    "e_xi_err": None, "p_xi_err": None, "phi_xi_err": None,
                                    "A_xi_err": None, "B_xi_err": None, "F_xi_err": None,
                                    "G_xi_err": None, "theta_xi_err": None},
                "flux_params_dict": {"f_s": None, "f_b": None},
                "chi2": None,
            },
            "neg_u_0": {
                "params_dict": {"t_0": None, "u_0": None, "t_E": None,
                                "e_xi": None, "p_xi": None, "phi_xi": None,
                                "A_xi": None, "B_xi": None, "F_xi": None, "G_xi": None,
                                "theta_xi":None},
                "params_err_dict": {"t_0_err": None, "u_0_err": None, "t_E_err": None,
                                    "e_xi_err": None, "p_xi_err": None, "phi_xi_err": None,
                                    "A_xi_err": None, "B_xi_err": None, "F_xi_err": None,
                                    "G_xi_err": None, "theta_xi_err": None},
                "flux_params_dict": {"f_s": None, "f_b": None},
                "chi2": None,
            },
            "t_ref": None,
        },
    }  # fmt: skip

    # fmt: off
    # Build constant parameters at first
    event_params_info["Parallax_Parameters"]["t_0_par"] = std_params_dict["t_0"]
    event_params_info["Xallarap_Circular_Parameters"]["t_ref"] = std_params_dict["t_0"]
    event_params_info["Xallarap_Circular_Thiele_Innes_Parameters"]["t_ref"] = std_params_dict["t_0"]
    event_params_info["Xallarap_Campbell_Parameters"]["t_ref"] = std_params_dict["t_0"]
    event_params_info["Xallarap_Thiele_Innes_Parameters"]["t_ref"] = std_params_dict["t_0"]
    # Build pos_u_0 dict and write
    event_params_info["Standard_Parameters"]["pos_u_0"]["params_dict"] = std_params_dict
    event_params_info["Standard_Parameters"]["pos_u_0"]["params_err_dict"] = std_params_err
    event_params_info["Standard_Parameters"]["pos_u_0"]["flux_params_dict"] = std_flux_dict
    event_params_info["Standard_Parameters"]["pos_u_0"]["chi2"] = std_chi2

    event_params_info["Parallax_Parameters"]["pos_u_0"]["params_dict"] = prlx_params_dict
    event_params_info["Parallax_Parameters"]["pos_u_0"]["params_err_dict"] = prlx_params_err
    event_params_info["Parallax_Parameters"]["pos_u_0"]["flux_params_dict"] = prlx_flux_dict
    event_params_info["Parallax_Parameters"]["pos_u_0"]["chi2"] = prlx_chi2

    event_params_info["Xallarap_Circular_Parameters"]["pos_u_0"]["params_dict"] = xlrp_circ_params_dict
    event_params_info["Xallarap_Circular_Parameters"]["pos_u_0"]["params_err_dict"] = xlrp_circ_params_err
    event_params_info["Xallarap_Circular_Parameters"]["pos_u_0"]["flux_params_dict"] = xlrp_circ_flux_dict
    event_params_info["Xallarap_Circular_Parameters"]["pos_u_0"]["chi2"] = xlrp_circ_chi2

    event_params_info["Xallarap_Circular_Thiele_Innes_Parameters"]["pos_u_0"]["params_dict"] = xlrp_circ_ti_params_dict
    event_params_info["Xallarap_Circular_Thiele_Innes_Parameters"]["pos_u_0"]["params_err_dict"] = xlrp_circ_ti_params_err
    event_params_info["Xallarap_Circular_Thiele_Innes_Parameters"]["pos_u_0"]["flux_params_dict"] = xlrp_circ_ti_flux_dict
    event_params_info["Xallarap_Circular_Thiele_Innes_Parameters"]["pos_u_0"]["chi2"] = xlrp_circ_ti_chi2

    event_params_info["Xallarap_Campbell_Parameters"]["pos_u_0"]["params_dict"] = xlrp_cpb_params_dict
    event_params_info["Xallarap_Campbell_Parameters"]["pos_u_0"]["params_err_dict"] = xlrp_cpb_params_err
    event_params_info["Xallarap_Campbell_Parameters"]["pos_u_0"]["flux_params_dict"] = xlrp_cpb_flux_dict
    event_params_info["Xallarap_Campbell_Parameters"]["pos_u_0"]["chi2"] = xlrp_cpb_chi2

    event_params_info["Xallarap_Thiele_Innes_Parameters"]["pos_u_0"]["params_dict"] = xlrp_ti_params_dict
    event_params_info["Xallarap_Thiele_Innes_Parameters"]["pos_u_0"]["params_err_dict"] = xlrp_ti_params_err
    event_params_info["Xallarap_Thiele_Innes_Parameters"]["pos_u_0"]["flux_params_dict"] = xlrp_ti_flux_dict
    event_params_info["Xallarap_Thiele_Innes_Parameters"]["pos_u_0"]["chi2"] = xlrp_ti_chi2

    # Build neg_u_0 dict and write
    std_params_dict_neg = std_params_dict.copy()
    std_params_dict_neg["u_0"] = -std_params_dict["u_0"]
    prlx_params_dict_neg = prlx_params_dict.copy()
    prlx_params_dict_neg["u_0"] = -prlx_params_dict["u_0"]
    xlrp_circ_params_dict_neg = xlrp_circ_params_dict.copy()
    xlrp_circ_params_dict_neg["u_0"] = -xlrp_circ_params_dict["u_0"]
    xlrp_circ_ti_params_dict_neg = xlrp_circ_ti_params_dict.copy()
    xlrp_circ_ti_params_dict_neg["u_0"] = -xlrp_circ_ti_params_dict["u_0"]
    xlrp_cpb_params_dict_neg = xlrp_cpb_params_dict.copy()
    xlrp_cpb_params_dict_neg["u_0"] = -xlrp_cpb_params_dict["u_0"]
    xlrp_ti_params_dict_neg = xlrp_ti_params_dict.copy()
    xlrp_ti_params_dict_neg["u_0"] = -xlrp_ti_params_dict["u_0"]

    event_params_info["Standard_Parameters"]["neg_u_0"]["params_dict"] = std_params_dict_neg
    event_params_info["Standard_Parameters"]["neg_u_0"]["params_err_dict"] = std_params_err
    event_params_info["Standard_Parameters"]["neg_u_0"]["flux_params_dict"] = std_flux_dict
    event_params_info["Standard_Parameters"]["neg_u_0"]["chi2"] = std_chi2

    event_params_info["Parallax_Parameters"]["neg_u_0"]["params_dict"] = prlx_params_dict_neg
    event_params_info["Parallax_Parameters"]["neg_u_0"]["params_err_dict"] = prlx_params_err
    event_params_info["Parallax_Parameters"]["neg_u_0"]["flux_params_dict"] = prlx_flux_dict
    event_params_info["Parallax_Parameters"]["neg_u_0"]["chi2"] = prlx_chi2

    event_params_info["Xallarap_Circular_Parameters"]["neg_u_0"]["params_dict"] = xlrp_circ_params_dict_neg
    event_params_info["Xallarap_Circular_Parameters"]["neg_u_0"]["params_err_dict"] = xlrp_circ_params_err
    event_params_info["Xallarap_Circular_Parameters"]["neg_u_0"]["flux_params_dict"] = xlrp_circ_flux_dict
    event_params_info["Xallarap_Circular_Parameters"]["neg_u_0"]["chi2"] = xlrp_circ_chi2

    event_params_info["Xallarap_Circular_Thiele_Innes_Parameters"]["neg_u_0"]["params_dict"] = xlrp_circ_ti_params_dict_neg
    event_params_info["Xallarap_Circular_Thiele_Innes_Parameters"]["neg_u_0"]["params_err_dict"] = xlrp_circ_ti_params_err
    event_params_info["Xallarap_Circular_Thiele_Innes_Parameters"]["neg_u_0"]["flux_params_dict"] = xlrp_circ_ti_flux_dict
    event_params_info["Xallarap_Circular_Thiele_Innes_Parameters"]["neg_u_0"]["chi2"] = xlrp_circ_ti_chi2

    event_params_info["Xallarap_Campbell_Parameters"]["neg_u_0"]["params_dict"] = xlrp_cpb_params_dict_neg
    event_params_info["Xallarap_Campbell_Parameters"]["neg_u_0"]["params_err_dict"] = xlrp_cpb_params_err
    event_params_info["Xallarap_Campbell_Parameters"]["neg_u_0"]["flux_params_dict"] = xlrp_cpb_flux_dict
    event_params_info["Xallarap_Campbell_Parameters"]["neg_u_0"]["chi2"] = xlrp_cpb_chi2

    event_params_info["Xallarap_Thiele_Innes_Parameters"]["neg_u_0"]["params_dict"] = xlrp_ti_params_dict_neg
    event_params_info["Xallarap_Thiele_Innes_Parameters"]["neg_u_0"]["params_err_dict"] = xlrp_ti_params_err
    event_params_info["Xallarap_Thiele_Innes_Parameters"]["neg_u_0"]["flux_params_dict"] = xlrp_ti_flux_dict
    event_params_info["Xallarap_Thiele_Innes_Parameters"]["neg_u_0"]["chi2"] = xlrp_ti_chi2
    # fmt: on

    # Write inro files
    with open(param_file_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            event_params_info, f, default_flow_style=False, allow_unicode=True
        )
    return
