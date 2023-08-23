import numpy as np
import matplotlib.pyplot as plt
import os

from xlrp.model import PointLensModel
from xlrp.data import Data


class Event(object):
    """
    This is a private module which mainly comes from MulensModel and sfitpy by Wei Zhu.

    Combines a microlensing model with data. Allows calculating chi^2.

    Args:
        datasets: :py:class:`~MylensModel.data.Data

        model: :py:class:`~MylensModel.model.Model`, which should **only for the
            ground-based data, i.e. ephemeris=None**.
            A space model will be created automatically for the satellite data.

        zero_blend: bool
    """

    def __init__(
        self,
        model: PointLensModel,
        dataset: Data,
        zero_blend_dict: dict = False,
    ):
        # Initialize self._datasets (and check that datasets is defined).
        if isinstance(dataset, Data):
            self.dataset = dataset
            self.ground_ob_tup = dataset.ground_ob_tup
            self.sapce_ob_tup = dataset.space_ob_tup
            self.all_ob_tup = (*self.ground_ob_tup, *self.sapce_ob_tup)
        else:
            raise TypeError("incorrect argument datasets of class Event()")

        # Every observatory needs its own model.
        self.model_dict: dict(str, PointLensModel) = {}
        self.magn_dict: dict(str, np.ndarray) = {}
        for obname in self.all_ob_tup:
            full_params_dict = model.parameters.copy()
            if "pi_E_N" in full_params_dict:
                full_params_dict["t_0_par"] = model.t_0_par
            if "p_xi" in full_params_dict:
                full_params_dict["t_ref"] = model.t_ref
            if obname in self.sapce_ob_tup:
                ephemeris_rel_path = f"ephemeris/{obname}.ephemeris"
                dir_name = os.path.dirname(__file__)
                ephemeris_full_path = os.path.join(dir_name, ephemeris_rel_path)
                ephemeris = np.loadtxt(ephemeris_full_path)
            else:
                ephemeris = None
            self.model_dict[obname] = PointLensModel(
                parameters=full_params_dict,
                ra=model.ra,
                dec=model.dec,
                ephemeris=ephemeris,
                obname=obname,
            )
            temp_date = dataset.data_dict[obname]["date"]
            self.model_dict[obname].set_times(temp_date)

        if zero_blend_dict is False:
            self.zero_blend_dict = {obname: False for obname in self.all_ob_tup}
        else:
            self.zero_blend_dict = zero_blend_dict

        # Properties related to FitData
        self.chi2_dict = {}
        self.flux_dict = {}
        for obname in self.all_ob_tup:
            self.chi2_dict[obname] = None
            self.flux_dict[obname] = {
                "f_s": None,
                "f_s_err": None,
                "f_b": None,
                "f_b_err": None,
            }
        self.chi2: float = 9e9
        self.best_chi2: float = 9e9  # This is a large number. Don't use np.inf to avoid error when converting to float.
        self.best_params: dict | None = None
        return

    def set_parameters(self, params_free, params_to_fit: list):
        for model in self.model_dict.values():
            for idx, name in enumerate(params_to_fit):
                model.parameters[name] = params_free[idx]
        return

    def fit_fsfb_single(
        self,
        flux,
        ferr,
        magnification,
        zero_blend,
        f_s: float | None = None,
        f_b: float | None = None,
    ):
        """> The function fits the blend factor fs and fb with weighted least square

        The function takes two arguments, `f_s` and `f_b`, and returns four values, `fs`, `fb`, `fserr`,
        and `fberr`

        The function has two parts:

        1. If `f_s` and `f_b` are setted, then the function sets `self.fs` and `self.fb` to `f_s`
        and `f_b` and returns `f_s` and `f_b`
        2. If `f_s` and `f_b` are not setted, then the function calculates `fs`, `fb`, `fserr`, and `fberr`

        Parameters
        ----------
        f_s : float|None
            the flux of the source
        f_b : float|None
            the blend factor

        Returns
        -------
            fs, fb, fserr, fberr

        """
        if f_s is not None and f_b is not None:
            return f_s, f_b, 0.0, 0.0

        if zero_blend:
            c00 = 1.0 / np.sum(magnification**2 / ferr**2)
            d0 = np.sum(flux * magnification / ferr**2)
            fs = c00 * d0
            fserr = np.sqrt(c00)
            fb, fberr = 0.0, 0.0
        else:
            sig2 = ferr**2
            wght = flux / sig2
            d = np.ones(2)
            d[0] = np.sum(wght * magnification)
            d[1] = np.sum(wght)
            b = np.zeros((2, 2))
            b[0, 0] = np.sum(magnification**2 / sig2)
            b[0, 1] = np.sum(magnification / sig2)
            b[1, 0] = b[0, 1]
            b[1, 1] = np.sum(1.0 / sig2)
            c = np.linalg.inv(b)
            fs = np.sum(c[0] * d)
            fb = np.sum(c[1] * d)
            fserr = np.sqrt(c[0, 0])
            fberr = np.sqrt(c[1, 1])

        return fs, fb, fserr, fberr

    def fit_fsfb_all(self, flux_dict: dict | None = None):
        # First fit for the ground-based data
        # TODO: flux dict may should be removed
        if flux_dict is not None:
            return flux_dict
        else:
            for obname in self.all_ob_tup:
                self.magn_dict[obname] = self.model_dict[obname].get_magnification()
                magnification = self.magn_dict[obname]
                flux = self.dataset.data_dict[obname]["flux"]
                ferr = self.dataset.data_dict[obname]["ferr"]
                f_s = f_b = None
                flux_tuple = self.fit_fsfb_single(
                    flux, ferr, magnification, self.zero_blend_dict[obname], f_s, f_b
                )
                self.flux_dict[obname]["f_s"] = flux_tuple[0]
                self.flux_dict[obname]["f_b"] = flux_tuple[1]
                self.flux_dict[obname]["f_s_err"] = flux_tuple[2]
                self.flux_dict[obname]["f_b_err"] = flux_tuple[3]
        return flux_dict

    @staticmethod
    def get_chi2_single(magnification, flux, ferr, fs, fb):
        """
        The function `get_chi2` is used to calculate the chi-square value of the model
        """
        model_flux = fs * magnification + fb
        chi2 = np.sum((flux - model_flux) ** 2 / ferr**2)
        return chi2

    def get_chi2(self, flux_dict=None):
        try:
            self.fit_fsfb_all(flux_dict)
        # TODO: this fuctioin should be in fit_fsfb_all
        except np.linalg.LinAlgError:
            # When failed to fit for fs and fb,
            # use the value from the previous run, i.e., do nothing.
            # If previous values do not exsist, set a default value.
            if None in list(self.flux_dict.values())[0].values():
                for obname in self.all_ob_tup:
                    self.flux_dict[obname]["f_s"] = 0.8
                    self.flux_dict[obname]["f_b"] = 0.2
                    self.flux_dict[obname]["f_s_err"] = 0.0
                    self.flux_dict[obname]["f_b_err"] = 0.0

        for obname in self.all_ob_tup:
            magnification = self.magn_dict[obname]
            flux = self.dataset.data_dict[obname]["flux"]
            ferr = self.dataset.data_dict[obname]["ferr"]
            f_s = self.flux_dict[obname]["f_s"]
            f_b = self.flux_dict[obname]["f_b"]
            self.chi2_dict[obname] = self.get_chi2_single(magnification, flux, ferr, f_s, f_b)  # fmt: skip

        this_chi2 = np.sum(list(self.chi2_dict.values()))
        if this_chi2 < self.best_chi2:
            self.best_chi2 = this_chi2
            self.best_params = self.model_dict[obname].parameters.copy()
        self.chi2 = this_chi2
        return this_chi2
