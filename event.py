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

    # TODO: Modify the function to suits multiple observations
    def get_lc(self, jds: np.ndarray | None = None, data_type="mag"):
        """> get Event model light curve data.

        ! Caution! This function is suggested to only used to plot events rather than fitting.
        Because it will calculate qn qe every time when be called.

        Parameters
        ----------
        jds : np.ndarray|None
            The time array to evaluate the model at. If None, the original time array is used.
        data_type, optional
            'mag' or 'flux'

        Returns
        -------
        return_lc: np.ndarray
        """

        if (self.fs is None) or (self.fb is None):
            raise AttributeError("You should fit for fs and fb first.")

        original_jds = self.model.jds
        if jds is not None:
            self.model.set_times(jds)

        if data_type == "mag":
            return_lc = self.model.get_light_curve(self.fs, self.fb, return_type="mag")
        elif data_type == "flux":
            return_lc = self.model.get_light_curve(self.fs, self.fb, return_type="flux")
        else:
            raise ValueError("data_type must be flux or mag.", data_type)

        if jds is not None:
            self.model.set_times(original_jds)
        return return_lc

    def get_parameters(self):
        """> The function `get_parameters` returns the parameters of the model

        Returns
        -------
            The parameters of the model.

        """
        return self.model.parameters

    def plot_events(
        self,
        model_kwargs: dict = None,
        data_kwargs: dict = None,
        bad_kwargs: dict = None,
        subtract_2450000=True,
    ):
        """It plots the data include bad data and model light curves

        Parameters
        ----------
        model_kwargs : dict
            a dictionary of keyword arguments for the model light curve
        data_kwargs : dict
            The keyword arguments for the data points.
            If 'bin_size' is in the dictionary, the data will be binned.
        bad_kwargs : dict
            keyword arguments for plotting bad data points

        """

        def get_binned_data(flux, ferr, bin_size):
            len_after_bin = len(flux) // bin_size + 1
            remainder = len(flux) % bin_size
            flux_binned, ferr_binned = np.zeros(len_after_bin), np.zeros(len_after_bin)
            for i in range(int(bin_size)):
                if i < remainder:
                    flux_binned += flux[i::bin_size]
                    ferr_binned += ferr[i::bin_size] ** 2 / bin_size**2
                else:
                    flux_binned[:-1] += flux[i::bin_size]
                    ferr_binned[:-1] += ferr[i::bin_size] ** 2 / remainder**2
            flux_binned[:-1] /= bin_size
            flux_binned[-1] /= remainder
            ferr_binned[:-1] = np.sqrt(ferr_binned[:-1])
            ferr_binned[-1] = np.sqrt(ferr_binned[-1])
            mag = self.dataset._zero_flux_mag - np.log10(flux) / 0.4
            merr = ferr * 2.5 / flux / np.log(10.0)
            return mag, merr

        # default plot style
        if model_kwargs is None:
            model_kwargs = {
                "color": "k",
                "linewidth": 1,
                "alpha": 0.5,
            }  # , "label":build_param_string(event_rp, "xlrp_circ", include_err=False)}
        if data_kwargs is None:
            data_kwargs = {
                "fmt": "o",
                "alpha": 0.5,
                "color": "royalblue",
                "markersize": 2,
            }
        if bad_kwargs is None:
            bad_kwargs = {"marker": "x", "color": "r", "alpha": 0.8, "s": 10}

        # Initialize
        hjds = self.dataset.date
        model_hjds = np.linspace(hjds[0], hjds[-1], 10000)
        if self.chi2 is None:
            self.get_chi2()

        # Dataset
        if "bin_size" in data_kwargs.keys():
            bin_size = data_kwargs.pop("bin_size")
            mag, merr = get_binned_data(self.dataset.flux, self.dataset.ferr, bin_size)
        else:
            mag, merr = self.dataset.mag, self.dataset.merr
        date_bad, mag_bad, merr_bad = (
            self.dataset.date_bad,
            self.dataset.mag_bad,
            self.dataset.merr_bad,
        )

        # Model
        self.model.set_times(model_hjds)
        ogle_fs = self.flux_dict["ogle"]["f_s"]
        ogle_fb = self.flux_dict["ogle"]["f_b"]
        model_lc = self.model.get_light_curve(fs=ogle_fs, fb=ogle_fb)
        self.model.set_times(hjds)

        # Plot
        plt.gca().invert_yaxis()
        if subtract_2450000:
            plt.errorbar(x=hjds - 2450000, y=mag, yerr=merr, **data_kwargs)
            plt.scatter(x=date_bad - 2450000, y=mag_bad, **bad_kwargs)
            plt.plot(model_hjds - 2450000, model_lc, **model_kwargs)
            plt.xlabel("JD - 2450000")
        else:
            plt.errorbar(x=hjds, y=mag, yerr=merr, **data_kwargs)
            plt.scatter(x=date_bad, y=mag_bad, **bad_kwargs)
            plt.plot(model_hjds, model_lc, **model_kwargs)
            plt.xlabel("JD")
        plt.ylabel("Magnitude")

    def plot_trajectory(self, traj_kwargs: dict):
        """It plots the trajectory of the source star in the sky plane

        Parameters
        ----------
        traj_kwargs : dict
            a dictionary of keyword arguments to be passed to the plot function.

        """
        # Initialize
        hjds = self.dataset.date
        model_hjds = np.linspace(hjds[0], hjds[-1], 2000)
        if self.chi2 is None:
            self.get_chi2()

        # Model
        self.model.set_times(model_hjds)
        model_traj = self.model.get_trajectory()
        self.model.set_times(hjds)
        self.model.get_light_curve(
            fs=self.fs, fb=self.fb
        )  # This line is only for resetting delta s_n delta s_e

        # Plot
        plt.plot(model_traj[0], model_traj[1], **traj_kwargs)
        plt.xlabel(r"$\beta/\theta_{\mathrm{E}}$")
        plt.ylabel(r"$\tau/\theta_{\mathrm{E}}$")

    def plot_residual(self, model_kwargs, data_kwargs, bad_kwargs):
        """This function plots the residual of the model light curve.

        Parameters
        ----------
        model_kwargs
            keyword arguments for the model line
        data_kwargs
            the kwargs for the data points
        bad_kwargs
            keyword arguments for plotting the bad data points

        """
        # TODO: This function is broken, 1. self.fs no longer exists.
        # Initialize
        hjds = self.dataset.date
        model_hjds = np.linspace(hjds[0], hjds[-1], 2000)
        if self.chi2 is None:
            self.get_chi2()

        # Dataset
        mag, merr = self.dataset.mag, self.dataset.merr
        date_bad, mag_bad, merr_bad = (
            self.dataset.date_bad,
            self.dataset.mag_bad,
            self.dataset.merr_bad,
        )

        # Model
        self.model.set_times(model_hjds)
        model_lc = self.model.get_light_curve(fs=self.fs, fb=self.fb)
        self.model.set_times(hjds)
        model_data_points = self.model.get_light_curve(
            fs=self.fs, fb=self.fb
        )  # This line is only for resetting delta s_n delta s_e

        residual_data_points = mag - model_data_points
        # Consider the residual of the bad data points
        # residual_baddata_points = mag_bad - model_data_points[date_bad-hjds[0]]

        # Plot
        plt.gca().invert_yaxis()
        plt.errorbar(x=hjds, y=residual_data_points, yerr=merr, **data_kwargs)
        # plt.scatter(x=date_bad, y=mag_bad, **bad_kwargs)
        plt.plot(model_hjds, np.zeros(len(model_hjds)), **model_kwargs)
        plt.xlabel("JD - 2450000")
        plt.ylabel("Magnitude Residual")
