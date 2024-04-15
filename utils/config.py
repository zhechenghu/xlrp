import yaml
import os
import numpy as np


class ConfigFile(object):
    def __init__(self, cfg_file_path: str):
        self.cfg_file_path = cfg_file_path

        with open(self.cfg_file_path, "r", encoding="utf-8") as f:
            self.event_info = yaml.load(f.read(), Loader=yaml.FullLoader)

        return


class ReadInfo(ConfigFile):
    def get_short_name(self):
        return self.event_info["Event_Info"]["short_name"]

    def get_official_name(self):
        return self.event_info["Event_Info"]["official_name"]

    def get_coords(self, return_type="str"):
        """
        Parameters
        ----------
        return_type: str, optional
            'str' returns a string of the form '17:51:40.19 -29:53:26.3'
            'dict' returns a dictionary of the form {'ra': '17:51:40.19', 'dec': '-29:53:26.3'}
        """
        ra = self.event_info["Event_Info"]["ra_j2000"]
        dec = self.event_info["Event_Info"]["dec_j2000"]
        coords = f"{ra} {dec}"
        if return_type == "str":
            return coords
        elif return_type == "dict":
            return {"ra": ra, "dec": dec}
        else:
            raise ValueError("return_type must be 'str' or 'dict'")

    def get_observatories(self):
        return self.event_info["Event_Info"]["observatories"]

    def get_event_dir(self):
        return self.event_info["Data_File"]["event_dir"]

    def get_data_path(self, observatory="ogle"):
        return self.event_info["Data_File"][observatory]

    def get_bad_path(self, observatory="ogle"):
        return self.event_info["Data_File"][f"{observatory}_bad"]

    def get_data_path_dict(self):
        data_path_dict = {}
        for observatory in self.get_observatories():
            data_path_dict[observatory] = self.get_data_path(observatory)
        return data_path_dict

    def get_param_path(self):
        return self.event_info["Result_File"]["param_file"]

    def get_bad_path_dict(self):
        bad_path_dict = {}
        for observatory in self.get_observatories():
            bad_path_dict[observatory] = self.get_bad_path(observatory)
        return bad_path_dict

    def get_emcee_path(self, neg_u0=False):
        if neg_u0:
            return self.event_info["Result_File"]["mcmc_neg_chain"]
        else:
            return self.event_info["Result_File"]["mcmc_pos_chain"]

    def get_dynesty_result(self, neg_u0=False):
        # TODO: check where this function has been used
        if neg_u0:
            return self.event_info["Data_File"]["dynesty_neg_result"]
        else:
            return self.event_info["Data_File"]["dynesty_pos_result"]

    def get_zero_blend_dict(self):
        return self.event_info["Control_Parameter"]["zero_blend"]

    def get_t_range(self):
        return self.event_info["Control_Parameter"]["t_range"]

    def get_fitting_method(self):
        return self.event_info["Control_Parameter"]["fitting_method"]

    def get_emcee_opt_dict(self):
        return self.event_info["Control_Parameter"]["emcee_opt_dict"]

    def get_dynesty_opt_dict(self):
        return self.event_info["Control_Parameter"]["dynesty_opt_dict"]

    def get_errfac(self, observatory="ogle"):
        return self.event_info["Error_Rescaling"][f"errfac_{observatory}"]

    def get_errfac_dict(self):
        errfac_dict = {}
        for observatory in self.get_observatories():
            errfac_dict[observatory] = self.get_errfac(observatory)
        return errfac_dict

    def get_base_ob_id(self):
        return self.event_info["Control_Parameter"]["base_ob_id"]

    def get_color_dict(self):
        return self.event_info["Control_Parameter"]["color_dict"]


class WriteInfo(ConfigFile):
    def save_yaml(self):
        with open(self.cfg_file_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                self.event_info, f, default_flow_style=False, allow_unicode=True
            )

    def set_phot_file_path(self, phot_file_path, observatory="ogle"):
        self.event_info["Data_File"][observatory] = phot_file_path
        self.save_yaml()

    def set_bad_file_path(self, bad_file_path, observatory="ogle"):
        self.event_info["Data_File"][f"{observatory}_bad"] = bad_file_path
        self.save_yaml()

    def set_mcmc_chain_path(self, mcmc_chain_path: str, neg_u0: bool):
        """
        Parameters
        ----------
        mcmc_chain_path: str
            path/to/mcmc/chain/path
        u0_sign: str
            'pos' or 'neg', the sign of u0.
        """
        u0_sign = "neg" if neg_u0 else "pos"
        self.event_info["Result_File"][f"mcmc_{u0_sign}_chain"] = mcmc_chain_path
        self.save_yaml()

    def set_dynesty_result_path(self, dynesty_result_path: str, neg_u0: bool):
        # TODO: check where this function has been used
        u0_sign = "neg" if neg_u0 else "pos"
        self.event_info["Result_File"][
            f"dynesty_{u0_sign}_result"
        ] = dynesty_result_path
        self.save_yaml()

    def reset_control_params(self):
        """
        Reset all control params to default values,
        which is fit standard light curve with downhill method.
        """
        initial_control_params = {
            "zero_blend": False,
            "fitting_method": "downhill",
            "emcee_opt_dict": {"nwalkers": 30, "nstep": 1000, "nburn": 1000},
            "dynesty_opt_dict": {"nlive": 500, "bound": "multi", "sample": "rwalk"},
        }

        # self.event_info should be changed in the same time
        self.event_info["Control_Parameter"] = initial_control_params
        self.save_yaml()

    def set_zero_blend_dict(self, zero_blend_dict=False):
        """
        Set the status of zero blend.
        Note that fs and fb should always be 'locked', which means it should be fitted by weight linear fit.
        """

        # self.event_info should be changed in the same time
        self.event_info["Control_Parameter"]["zero_blend"] = zero_blend_dict
        self.save_yaml()

    def set_t_range(self, t_range: list | None = None):
        """
        Set the time range of good time.
        """
        self.event_info["Control_Parameter"]["t_range"] = t_range
        self.save_yaml()

    def set_fitting_method(self, method: str):
        """Change the cfg file to set fitting method.

        :param method: str
                'downhill' or 'emcee' or 'dynesty'
        """
        method_list = ["downhill", "emcee", "dynesty"]
        if method not in method_list:
            raise ValueError(f"Method should be one of {method_list}")
        self.event_info["Control_Parameter"]["fitting_method"] = method
        self.save_yaml()
        return

    def set_emcee_options(self, n_walkers=30, n_burn=1000, n_steps=1000):
        self.event_info["Control_Parameter"]["emcee_opt_dict"] = {
            "nwalkers": n_walkers,
            "nstep": n_steps,
            "nburn": n_burn,
        }
        self.save_yaml()

    def set_dynesty_options(self, nlive=500, bound="multi", sample="rwalk"):
        self.event_info["Control_Parameter"]["dynesty_opt_dict"] = {
            "nlive": nlive,
            "bound": bound,
            "sample": sample,
        }
        self.save_yaml()

    def set_errfac(self, errfac: list, observatory="ogle"):
        ## err_correct = k*sqrt(err^2+a0^2) ##
        self.event_info["Error_Rescaling"][f"errfac_{observatory}"] = errfac
        self.save_yaml()

    def set_base_ob_id(self, base_ob_id: str):
        assert base_ob_id in self.event_info["Event_Info"]["observatories"]
        self.event_info["Control_Parameter"]["base_ob_id"] = base_ob_id
        self.save_yaml()

    def set_color(self, ob_id, color):
        assert ob_id in self.event_info["Event_Info"]["observatories"]
        self.event_info["Control_Parameter"]["color_dict"][ob_id] = color
        self.save_yaml()


def init_datafile_dict(data_dir, ob_id_list):
    def datafile_item(data_dir, file_id):
        file_list = os.listdir(data_dir)
        target_file_list = [
            file_name
            for file_name in file_list
            if file_id in file_name and ".bad" not in file_name
        ]
        if len(target_file_list) == 0:
            raise ValueError(f"No file found with id {file_id} in {data_dir}")
        elif len(target_file_list) > 1:
            raise ValueError(f"Multiple files found with id {file_id} in {data_dir}")
        return os.path.join(data_dir, target_file_list[0])

    return {file_id: datafile_item(data_dir, file_id) for file_id in ob_id_list}


def init_bad_datafile_dict(datafile_dict):
    return {k + "_bad": v + ".bad" for k, v in datafile_dict.items()}


def init_errfac_dict(ob_id_list):
    return {"errfac_" + observatory: [0.003, 1.0] for observatory in ob_id_list}


def init_base_ob_id(ob_id_list):
    if "ogle" in ob_id_list:
        return "ogle"
    elif "OB" in ob_id_list:
        return "OB"
    else:
        for ob_id in ob_id_list:
            if ob_id.startswith("C") and "I" in ob_id:
                return ob_id


def init_color_dict(ob_id_list):
    default_color_dict = {
        "OGLE": "black",
        "KMT_C": "C3",  # red
        "KMT_A": "C0",  # blue
        "KMT_S": "C2",  # green
    }
    color_dict = {}
    for ob_id in ob_id_list:
        if "ogle" in ob_id.lower():
            color_dict[ob_id] = default_color_dict["OGLE"]
        elif "kmtc" in ob_id.lower():
            color_dict[ob_id] = default_color_dict["KMT_C"]
        elif "kmta" in ob_id.lower():
            color_dict[ob_id] = default_color_dict["KMT_A"]
        elif "kmts" in ob_id.lower():
            color_dict[ob_id] = default_color_dict["KMT_S"]
    return color_dict


def init_cfg_file(
    star_name: str,
    event_name: str,
    official_name: str,
    event_ra: str,
    event_dec: str,
    event_dir: str,
    ob_id_list: list,
):
    """It initializes a config file from given parameters

    Parameters
    ----------
    star_name
        str, star name
    event_name
        str, The name of the event in my catalog.
    official_name
        The name of the event in the official catalog.
    event_ra
        str, right ascension of the event
    event_dec
        str, declination of the event
    data_prefix
        str, the path to the directory where the data files are stored.
    """

    event_info = {
        "Event_Info": {
            "official_name": None,
            "short_name": None,
            "star_name": None,
            "ra_j2000": None,
            "dec_j2000": None,
            "observatories": ob_id_list,
            "bands": "I",
        },
        "Data_File": {},
        "Result_File": {
            "param_file": os.path.join(event_dir, f"{event_name}_param.yaml"),
        },
        "Control_Parameter": {
            "zero_blend": False,
            "t_range": None,
            "fitting_method": "downhill",
            "base_ob_id": None,
            "emcee_opt_dict": {"nwalkers": 30, "nstep": 1000, "nburn": 1000},
            "dynesty_opt_dict": {"nlive": 500, "bound": "multi", "sample": "rwalk"},
            "color_dict": {},
        },
        "Error_Rescaling": {},  # In general, it is [0.003, 1.0]
    }

    event_info["Event_Info"]["official_name"] = official_name
    event_info["Event_Info"]["short_name"] = event_name
    event_info["Event_Info"]["star_name"] = star_name
    event_info["Event_Info"]["ra_j2000"] = event_ra
    event_info["Event_Info"]["dec_j2000"] = event_dec

    data_dict = init_datafile_dict(event_dir, ob_id_list)
    bad_dict = init_bad_datafile_dict(data_dict)
    event_info["Data_File"] = {**data_dict, **bad_dict}
    event_info["Error_Rescaling"] = init_errfac_dict(ob_id_list)
    event_info["Control_Parameter"]["base_ob_id"] = init_base_ob_id(ob_id_list)
    event_info["Control_Parameter"]["color_dict"] = init_color_dict(ob_id_list)

    event_config_path = os.path.join(event_dir, f"{event_name}_config.yml")
    with open(event_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(event_info, f, default_flow_style=False, allow_unicode=True)
    return event_config_path
