import numpy as np


class Data(object):
    """
    This is a private module which mainly comes from MulensModel and sfitpy by Wei Zhu.

    A set of photometric measurements for a microlensing event.

    Examples of how to define a MulensData object:
        data = MulensData(file_name=SAMPLE_FILE_01)

    Parameters:
        data_file_dict: dict[str, str]
            The dictionary with keys of observatories and values of data files
            with columns: Date, Magnitude/Flux, Err.
            Loaded using :py:func:`numpy.loadtxt()`.

        data_arr_dict: dict[str, np.ndarray]
            The dictionary with keys of observatories and values of arraies
            with columns: Date, Magnitude, Err.
            It is equivalent to `data_file_dict[key] = np.loadtxt(data_arr_dict[key])`

        obname: list[str]
            The list of names of the observatory.
            Used to define the usage of each columns

        bad_file_dict: dict[str, str]
            The dictionary with keys of observatories and values of bad data files
            with columns: Date, Magnitude/Flux, Err.

        t_range: list[float]
            The range of time to cut the data.
            The default is None.

        sub_2450000: bool
            Whether to subtract 2450000 days from the data.
            Current the module only works when this parameter set as True.

        zero_flux_mag: float
            The magnitude of zero flux.
            It is not important when the errfac is correct.
    """

    def __init__(
        self,
        data_file_dict=None,
        data_arr_dict=None,
        errfac_dict=None,
        obname_list=["ogle"],
        bad_file_dict=None,
        t_range: list | None = None,
        sub_2450000=False,
        zero_flux_mag=18.0,
    ):
        self._t_range = t_range
        self._sub_2450000 = sub_2450000
        self._zero_flux_mag = zero_flux_mag
        # Initialize observatory list
        self.ground_ob_list = []
        self.space_ob_list = []
        self.errfac_dict = errfac_dict
        for ob in obname_list:
            if ob not in ["spitzer", "kepler"]:
                self.ground_ob_list.append(ob)
            else:
                self.space_ob_list.append(ob)
        self.ground_ob_tup = tuple(self.ground_ob_list)  # use tuple to not modify
        self.space_ob_tup = tuple(self.space_ob_list)
        self.all_ob_tup = (*obname_list,)
        # Initialize data dict
        if data_file_dict is not None:
            self._data_file_dict = data_file_dict
            if bad_file_dict is None:
                self._bad_file_dict = {}
                for obname in obname_list:
                    self._bad_file_dict[obname] = None
            else:
                self._bad_file_dict = bad_file_dict
            self._get_all_data()
        elif data_arr_dict is not None:
            self._data_arr_dict = data_arr_dict
            self._build_data_dict_from_arr()
        else:
            raise ValueError("No data is provided.")
        return

    def _build_obname_list(self):
        self.obname_list = []
        for obname in self._data_file_dict.keys():
            self.obname_list.append(obname)
        return

    def _build_data_dict_from_arr(self):
        self.data_dict = {}
        for obname in self.all_ob_tup:
            data_arr = self._data_arr_dict[obname]
            date = data_arr[:, 0]
            mag, merr = data_arr[:, 1], data_arr[:, 2]
            mag_bad, merr_bad = np.array([]), np.array([])
            flux, ferr = self._compute_f_ferr_from_m_merr(mag, merr)
            date_bad, flux_bad, ferr_bad = (
                np.array([]),
                np.array([]),
                np.array([]),
            )
            data_tuple_full = (date, mag, merr, flux, ferr, 
                               date_bad, mag_bad, merr_bad, flux_bad, ferr_bad)  # fmt: skip
            self._build_sub_data_dict(obname, data_tuple_full)

    def _get_all_data(self):
        self.data_dict = {}
        for obname in self.all_ob_tup:
            data_file = self._data_file_dict[obname]
            errfac = self.errfac_dict[obname]
            bad_file = self._bad_file_dict[obname]
            data_tuple_full = self.get_one_data(data_file, errfac, bad_file)

            self._build_sub_data_dict(obname, data_tuple_full)

        return

    def _build_sub_data_dict(self, obname, data_tuple_full):
        date, mag, merr, flux, ferr = data_tuple_full[:5]
        date_bad, mag_bad, merr_bad, flux_bad, ferr_bad = data_tuple_full[5:]

        # Full date here
        # For calculating the full magnification curve with numpy array in the future,
        # which is much faster than the for loop.

        self.data_dict[obname] = {}
        self.data_dict[obname]["date"] = date.copy()
        self.data_dict[obname]["flux"] = flux.copy()
        self.data_dict[obname]["ferr"] = ferr.copy()
        self.data_dict[obname]["mag"] = mag.copy()
        self.data_dict[obname]["merr"] = merr.copy()
        self.data_dict[obname]["date_bad"] = date_bad.copy()
        self.data_dict[obname]["flux_bad"] = flux_bad.copy()
        self.data_dict[obname]["ferr_bad"] = ferr_bad.copy()
        self.data_dict[obname]["mag_bad"] = mag_bad.copy()
        self.data_dict[obname]["merr_bad"] = merr_bad.copy()
        return

    def get_one_data(
        self,
        data_file: str,
        errfac: list[float],
        bad_file=None,
    ):
        """The function reads in all data from the data file."""
        date, mag, merr = self._load_date_mag_merr(data_file, errfac)

        # The following code need to be improved when the
        # bad data file is improved.
        if bad_file is not None:
            data_and_bad = self._get_bad_data(bad_file, date, mag, merr)
            date, mag, merr, date_bad, mag_bad, merr_bad = data_and_bad
        else:
            date_bad, mag_bad, merr_bad = np.array([]), np.array([]), np.array([])

        if self._t_range is not None:
            date, mag, merr = self._cut_data(date, mag, merr)

        # The default zero point is 18 mag.
        flux, ferr = self._compute_f_ferr_from_m_merr(mag, merr)
        flux_bad, ferr_bad = self._compute_f_ferr_from_m_merr(mag_bad, merr_bad)
        return (date, mag, merr, flux, ferr, 
                date_bad, mag_bad, merr_bad, flux_bad, ferr_bad)  # fmt: skip

    def _load_date_mag_merr(self, data_file, errfac):
        """The function reads in date, mag, merr from the data file."""
        data = np.loadtxt(data_file, usecols=(0, 1, 2))

        date = data[:, 0]
        mag = data[:, 1]
        merr = np.sqrt(errfac[1] ** 2 * data[:, 2] ** 2 + errfac[0] ** 2)

        if self._sub_2450000:
            if date[0] > 2450000.0:
                date -= 2450000.0
        elif date[0] < 2450000.0:
            date += 2450000.0

        return date, mag, merr

    def _cut_data(self, date, mag, merr):
        """The function cuts the data based on the time range."""
        cut_mask_1 = date > self._t_range[0]
        date = date[cut_mask_1]
        mag = mag[cut_mask_1]
        merr = merr[cut_mask_1]
        cut_mask_2 = date < self._t_range[1]
        date = date[cut_mask_2]
        mag = mag[cut_mask_2]
        merr = merr[cut_mask_2]

        return date, mag, merr

    def _compute_f_ferr_from_m_merr(self, mag, merr):
        """The function computes the flux and flux error from the magnitude and magnitude error."""
        flux = 10.0 ** (0.4 * (self._zero_flux_mag - mag))
        ferr = merr * flux * np.log(10.0) / 2.5
        return flux, ferr

    def _get_bad_data(self, bad_file, date, mag, merr):
        """It reads in the bad data file, and then finds the corresponding data points in the original data
        file.

        The bad data file is a list of dates, and the original data file is a list of dates and fluxes.

        The function finds the dates in the original data file that are closest to the dates in the bad data
        file, and then removes those data points from the original data file.

        The function returns nothing.

        The function is called in the `get_all_data` function.

        Parameters
        ----------
        date
            the time of each observation
        flux
            the flux of the star
        ferr
            the error of the flux

        """
        bad_data = np.loadtxt(bad_file)
        if len(bad_data) == 0:
            # If the bad data file is empty, return the original data.
            return date, mag, merr, np.array([]), np.array([]), np.array([])

        bad_date = (
            bad_data[:, 0] if len(bad_data.shape) > 1 else np.array([bad_data[0]])
        )
        if self._sub_2450000:
            if bad_date[0] > 2450000.0:
                bad_date -= 2450000.0
        elif bad_date[0] < 2450000.0:
            bad_date += 2450000.0
        loc = []
        for i_bad in bad_date:
            # Locate the time which is most close to each bad date.
            iloc = np.abs(date - i_bad) < 1e-5  # 1e-5 day = 1s
            # there is more than one entry for this date:
            # must be something wrong
            if len(np.ones_like(iloc)[iloc]) > 1:
                print("wrong baddata file: ", bad_file, i_bad, len(np.ones_like(iloc)[iloc]))  # fmt: skip
                return date, mag, merr, np.array([]), np.array([]), np.array([])
            loc.append(iloc)
        bad = np.sum(np.array(loc), axis=0).astype(bool)
        good = ~bad

        dataset = np.vstack((date, mag, merr))
        dataset_bad = (dataset.T)[bad].T
        dataset = (dataset.T)[good].T

        date, mag, merr = dataset[:3]
        date_bad, mag_bad, merr_bad = dataset_bad[:3]
        return date, mag, merr, date_bad, mag_bad, merr_bad

    def convert_to_ogle(self, flux, ferr, fsfb, fsfb_ogle, use_mag=True):
        """
        Convert flux to ogle flux. If `use_mag` is True, return magnitude.
        """

        fs, fb = fsfb
        fs_ogle, fb_ogle = fsfb_ogle
        flux_ogle = (flux - fb) / fs * fs_ogle + fb_ogle
        ferr_ogle = ferr / fs * fs_ogle

        if use_mag == False:
            return flux_ogle, ferr_ogle
        mag_ogle = self._zero_flux_mag - 2.5 * np.log10(flux_ogle)
        merr_ogle = ferr_ogle * 2.5 / np.log(10) / flux_ogle
        return mag_ogle, merr_ogle

    def set_all_data(self, flux_dict):
        """Set all data."""
        self.date = self.date_bad = np.array([])
        self.flux = self.ferr = np.array([])
        self.flux_bad = self.ferr_bad = np.array([])
        self.mag = self.merr = np.array([])
        self.mag_bad = self.merr_bad = np.array([])
        for obname in self.ground_ob_tup:
            date = self.data_dict[obname]["date"]
            flux = self.data_dict[obname]["flux"]
            ferr = self.data_dict[obname]["ferr"]
            fsfb = [flux_dict[obname]["f_s"], flux_dict[obname]["f_b"]]
            fsfb_ogle = [flux_dict["ogle"]["f_s"], flux_dict["ogle"]["f_b"]]
            flux_ogle, ferr_ogle = self.convert_to_ogle(
                flux, ferr, fsfb, fsfb_ogle, False
            )
            mag_ogle = self._zero_flux_mag - 2.5 * np.log10(flux_ogle)
            merr_ogle = ferr_ogle * 2.5 / np.log(10) / flux_ogle
            self.date = np.append(self.date, date)
            self.flux = np.append(self.flux, flux_ogle)
            self.ferr = np.append(self.ferr, ferr_ogle)
            self.mag = np.append(self.mag, mag_ogle)
            self.merr = np.append(self.merr, merr_ogle)
            date_bad = self.data_dict[obname]["date_bad"]
            flux_bad = self.data_dict[obname]["flux_bad"]
            ferr_bad = self.data_dict[obname]["ferr_bad"]
            flux_ogle_bad, ferr_ogle_bad = self.convert_to_ogle(
                flux_bad, ferr_bad, fsfb, fsfb_ogle, False
            )
            mag_ogle_bad = self._zero_flux_mag - 2.5 * np.log10(flux_ogle_bad)
            merr_ogle_bad = ferr_ogle_bad * 2.5 / np.log(10) / flux_ogle_bad
            self.date_bad = np.append(self.date_bad, date_bad)
            self.flux_bad = np.append(self.flux_bad, flux_ogle_bad)
            self.ferr_bad = np.append(self.ferr_bad, ferr_ogle_bad)
            self.mag_bad = np.append(self.mag_bad, mag_ogle_bad)
            self.merr_bad = np.append(self.merr_bad, merr_ogle_bad)

        sort_idx = np.argsort(self.date)
        self.date = self.date[sort_idx]
        self.flux = self.flux[sort_idx]
        self.ferr = self.ferr[sort_idx]
        self.mag = self.mag[sort_idx]
        self.merr = self.merr[sort_idx]
        sort_bad_idx = np.argsort(self.date_bad)
        self.date_bad = self.date_bad[sort_bad_idx]
        self.flux_bad = self.flux_bad[sort_bad_idx]
        self.ferr_bad = self.ferr_bad[sort_bad_idx]
        self.mag_bad = self.mag_bad[sort_bad_idx]
        self.merr_bad = self.merr_bad[sort_bad_idx]
        return
