import numpy as np
import matplotlib.pyplot as plt
from xlrp import Event
from xlrp.utils.config import ReadInfo
from xlrp.utils.fitting import FitUtils
from PyAstronomy.pyasl import binningx0dt
from copy import deepcopy


class PlotLC:
    def __init__(
        self,
        event_ri: ReadInfo,
        params_dict: dict,
        model_start=None,
        model_end=None,
        model_step=0.1,
        date_offset=2450000,
        # base_params_dict=None,
    ):
        self.date_offset = date_offset
        self.event_ri = event_ri
        self.event = FitUtils.build_mylens_event(
            event_ri=event_ri, params_dict=params_dict
        )
        self.event.get_chi2()
        if model_start is None:
            model_start, model_end = self.get_date_start_end()
        self.model_date = np.arange(model_start, model_end, model_step)
        self.data_dict = self.get_data_model_util()
        self.color_dict = self.event_ri.get_color_dict()
        # if base_params_dict is not None:
        #    self.base_event = FitUtils.build_mylens_event(
        #        event_ri=event_ri, params_dict=base_params_dict
        #    )
        #    self.base_data_dict = self.get_data_model_util(self.base_event)

    def get_date_start_end(self):
        date_start = np.inf
        date_end = -np.inf
        for obname in self.event.all_ob_tup:
            date_start = min(
                date_start, np.min(self.event.dataset.data_dict[obname]["date"])
            )
            date_end = max(
                date_end, np.max(self.event.dataset.data_dict[obname]["date"])
            )
        return date_start, date_end

    def get_data_model_util(self, align_spitzer=False):
        # fmt: off
        this_event = deepcopy(self.event)
        this_event.get_chi2()
        #print(this_event.chi2_dict)

        plot_data_dict = {}
        for obname in this_event.all_ob_tup:
            # Data points, those without prefix "model_" are original data points
            data_dict = this_event.dataset.data_dict[obname]
            date = data_dict["date"]
            mag = data_dict["mag"]
            merr = data_dict["merr"]
            flux = data_dict["flux"]
            ferr = data_dict["ferr"]
            flux_dict = this_event.flux_dict[obname]
            # Model
            model = this_event.model_dict[obname]
            # with "_real_" inside the name are model predictions on original data points
            model_real_mag = model.get_light_curve(fs=flux_dict["f_s"], fb=flux_dict["f_b"])
            model_real_flux = model.get_light_curve(fs=flux_dict["f_s"], fb=flux_dict["f_b"], return_type="flux")
            mag_res = mag - model_real_mag
            model.set_times(self.model_date)
            model_mag = model.get_light_curve(fs=flux_dict["f_s"], fb=flux_dict["f_b"])
            model_flux = model.get_light_curve(fs=flux_dict["f_s"], fb=flux_dict["f_b"], return_type="flux")
            model_magnification_norm = model.get_magnification()

            if obname != "spitzer" or align_spitzer:
                # I need to set a base dataset to align others.
                base_ob_id = self.event_ri.get_base_ob_id()
                base_flux_dict = this_event.flux_dict[base_ob_id]
                magnification = (flux - flux_dict["f_b"]) / flux_dict["f_s"]
                flux = magnification * base_flux_dict["f_s"] + base_flux_dict["f_b"]
                ferr = ferr / flux_dict["f_s"] * base_flux_dict["f_s"]
                mag = -2.5 * np.log10(flux) + 18
                merr = ferr * 2.5 / np.log(10) / flux
                model_real_magnification = (model_real_flux - flux_dict["f_b"]) / flux_dict["f_s"]
                model_real_flux = model_real_magnification * base_flux_dict["f_s"]+ base_flux_dict["f_b"]
                model_real_mag = -2.5 * np.log10(model_real_flux) + 18
                mag_res = mag - model_real_mag
                model_magnification = (model_flux - flux_dict["f_b"]) / flux_dict["f_s"]
                model_flux = model_magnification * base_flux_dict["f_s"] + base_flux_dict["f_b"]
                model_mag = -2.5 * np.log10(model_flux) + 18
                # sometimes there will be negative flux, which is not physical.
                # the reason is some points has so low flux, even much lower than the background (base line)
                # then flux - f_b < 0, which is not physical.
                # such data points should be removed in the .bad file
                # but let me just handle it here for now.
                bad_idx = flux < 0
                date = date[~bad_idx]
                mag = mag[~bad_idx]
                merr = merr[~bad_idx]
                flux = flux[~bad_idx]
                ferr = ferr[~bad_idx]
                mag_res = mag_res[~bad_idx]
                model_real_flux = model_real_flux[~bad_idx]
                model_real_mag = model_real_mag[~bad_idx]
                # and some times the converted merr is too large
                bad_idx = merr > 1.0
                date = date[~bad_idx]
                mag = mag[~bad_idx]
                merr = merr[~bad_idx]
                flux = flux[~bad_idx]
                ferr = ferr[~bad_idx]
                mag_res = mag_res[~bad_idx]
                model_real_flux = model_real_flux[~bad_idx]
                model_real_mag = model_real_mag[~bad_idx]
                
            chi2_seq = []
            for i in range(1, len(flux) + 1):
                chi2_seq.append(
                    np.sum(((flux[:i] - model_real_flux[:i]) ** 2 / ferr[:i] ** 2))
                )
            chi2_seq = np.array(chi2_seq)
            #print(f"Check chi2 for {obname}: {chi2_seq[-1]}")

            plot_data_dict[obname] = {
                "date": date,
                "mag": mag,
                "merr": merr,
                "flux": flux,
                "ferr": ferr,
                "mag_res": mag_res,
                "model_real_flux": model_real_flux,
                "model_real_mag": model_real_mag,
                # below are model predictions on model_date
                "model_date": self.model_date,
                "model_mag": model_mag,
                "model_flux": model_flux,
                "model_magnification_norm": model_magnification_norm,
                "chi2": chi2_seq,
            }

        # fmt: on
        return plot_data_dict

    def plot_data(self, ax1, bin_data=False, **kwargs):
        for obname in self.data_dict:
            if obname == "spitzer":
                y = -2.5 * np.log10(self.data_dict[obname]["flux"])
                ax1.errorbar(
                    self.data_dict[obname]["date"] - self.date_offset,
                    y,
                    yerr=self.data_dict[obname]["merr"],
                    color=self.color_dict[obname],
                    **kwargs,
                )
            elif bin_data:
                data_binned, new_dt = binningx0dt(
                    self.data_dict[obname]["date"],
                    self.data_dict[obname]["mag"],
                    yerr=self.data_dict[obname]["merr"],
                    dt=1.0,
                    useBinCenter=True,
                )
                ax1.errorbar(
                    data_binned[:, 0] - self.date_offset,
                    data_binned[:, 1],
                    yerr=data_binned[:, 2],
                    color=self.color_dict[obname],
                    **kwargs,
                )
            else:
                try:
                    ax1.errorbar(
                        self.data_dict[obname]["date"] - self.date_offset,
                        self.data_dict[obname]["mag"],
                        yerr=self.data_dict[obname]["merr"],
                        color=self.color_dict[obname],
                        **kwargs,
                    )
                except ValueError:
                    print(f"Error event: {self.event_ri.get_short_name()}")
                    print(f"Error dataset: {obname}")
                    # print the merr < 0 points
                    print(self.event.flux_dict)
                    print("date")
                    print(
                        self.data_dict[obname]["date"][
                            self.data_dict[obname]["merr"] < 0
                        ]
                    )
                    print("mag")
                    print(
                        self.data_dict[obname]["mag"][
                            self.data_dict[obname]["merr"] < 0
                        ]
                    )
                    print("merr")
                    print(
                        self.data_dict[obname]["merr"][
                            self.data_dict[obname]["merr"] < 0
                        ]
                    )
                    print("flux")
                    print(
                        self.data_dict[obname]["flux"][
                            self.data_dict[obname]["merr"] < 0
                        ]
                    )
                    print("ferr")
                    print(
                        self.data_dict[obname]["ferr"][
                            self.data_dict[obname]["merr"] < 0
                        ]
                    )
        return

    def plot_res(self, ax2, bin_data=False, **kwargs):
        for obname in self.data_dict:
            if bin_data:
                data_binned, new_dt = binningx0dt(
                    self.data_dict[obname]["date"],
                    self.data_dict[obname]["mag_res"],
                    yerr=self.data_dict[obname]["merr"],
                    dt=1.0,
                    useBinCenter=True,
                )
                ax2.errorbar(
                    data_binned[:, 0] - self.date_offset,
                    data_binned[:, 1],
                    yerr=data_binned[:, 2],
                    color=self.color_dict[obname],
                    **kwargs,
                )
            else:
                ax2.errorbar(
                    self.data_dict[obname]["date"] - self.date_offset,
                    self.data_dict[obname]["mag_res"],
                    yerr=self.data_dict[obname]["merr"],
                    color=self.color_dict[obname],
                    **kwargs,
                )
        return

    def plot_model(self, ax1, obname_list=None, label_dict=None, **kwargs):
        if obname_list is None:
            obname_list = self.data_dict.keys()
        for obname in obname_list:
            if obname == "spitzer":
                y = -2.5 * np.log10(self.data_dict[obname]["model_flux"])
                ax1.plot(
                    self.data_dict[obname]["model_date"] - self.date_offset,
                    y,
                    color=self.color_dict[obname],
                    label=label_dict[obname] if label_dict is not None else None,
                    **kwargs,
                )
            else:
                ax1.plot(
                    self.data_dict[obname]["model_date"] - self.date_offset,
                    self.data_dict[obname]["model_mag"],
                    color=self.color_dict[obname],
                    label=label_dict[obname] if label_dict is not None else None,
                    **kwargs,
                )
        return

    def plot_chi2(self, ax3, obname_list=None, **kwargs):
        if obname_list is None:
            obname_list = self.data_dict.keys()
        for obname in obname_list:
            chi2_cum = self.data_dict[obname]["chi2"]
            ax3.plot(
                self.data_dict[obname]["date"] - self.date_offset,
                chi2_cum,
                color=self.color_dict[obname],
                **kwargs,
            )
        pass

    @staticmethod
    def set_fig_3_axes(fig_scale=1.5, fig_ratio=[5, 4], dpi=300):
        # fmt: off
        fig = plt.figure(figsize=(fig_ratio[0]*fig_scale, fig_ratio[1]*fig_scale), dpi=dpi)

        ax3 = fig.add_subplot(311)
        ax3.set_position([0.1, 0.1, 0.8, 0.2])
        ax3.tick_params(which="major", direction="in", length=5, width=0.5, labelsize=10, top=True, right=True, bottom=True, left=True)
        ax3.tick_params(which="minor", direction="in", length=3, width=0.5, labelsize=10, top=True, right=True, bottom=True, left=True)
        ax3.minorticks_on()
        ax3.set_xlabel("HJD - 2450000")
        ax3.set_ylabel(r"Cum. $\Delta \chi^2$")

        ax2 = fig.add_subplot(312)
        ax2.set_position([0.1, 0.3+0.015, 0.8, 0.15])
        ax2.tick_params(which="major", direction="in", length=5, width=0.5, labelsize=10, top=True, right=True, bottom=True, left=True)
        ax2.tick_params(which="minor", direction="in", length=3, width=0.5, labelsize=10, top=True, right=True, bottom=True, left=True)
        ax2.minorticks_on()
        ax2.invert_yaxis()
        ax2.set_xticklabels([])
        ax2.set_ylabel("Residual [mag]")

        ax1 = fig.add_subplot(313)
        ax1.set_position([0.1, 0.5-0.02, 0.8, 0.4])
        ax1.tick_params(which="major", direction="in", length=5, width=0.5, labelsize=10, top=True, right=True, bottom=True, left=True)
        ax1.tick_params(which="minor", direction="in", length=3, width=0.5, labelsize=10, top=True, right=True, bottom=True, left=True)
        ax1.minorticks_on()
        ax1.set_xticklabels([])
        ax1.invert_yaxis()
        ax1.set_ylabel(r"$\mathrm{I}_{\mathrm{OGLE}}$ [mag]")
        # fmt: on
        return fig, ax1, ax2, ax3

    def plot_all(
        self,
        bin_data=True,
        general_data_fmt={
            "fmt": "o",
            "markerfacecolor": "none",
            "capsize": 3,
            "markersize": 5,
            "markeredgewidth": 0.5,
            "elinewidth": 0.5,
        },
        model_fmt={"ls": "-"},
        fig_dpi=300,
        data_ax=None,
        res_ax=None,
        chi2_ax=None,
    ):
        # fig, ax1, ax2, ax3 = self.set_fig_3_axes(dpi=fig_dpi)
        if data_ax is None and res_ax is None and chi2_ax is None:
            fig, data_ax, res_ax, chi2_ax = self.set_fig_3_axes(dpi=fig_dpi)
            return_full = True
        else:
            return_full = False

        if data_ax is not None:
            self.plot_data(data_ax, bin_data=bin_data, **general_data_fmt)
            self.plot_model(
                data_ax, obname_list=[self.event_ri.get_base_ob_id()], **model_fmt
            )
            data_ax.set_xlim(
                self.model_date[0] - self.date_offset,
                self.model_date[-1] - self.date_offset,
            )
        if res_ax is not None:
            self.plot_res(res_ax, bin_data=bin_data, **general_data_fmt)
            res_ax.set_xlim(
                self.model_date[0] - self.date_offset,
                self.model_date[-1] - self.date_offset,
            )
        if chi2_ax is not None:
            self.plot_chi2(chi2_ax)
            chi2_ax.set_xlim(
                self.model_date[0] - self.date_offset,
                self.model_date[-1] - self.date_offset,
            )

        if return_full:
            return fig, data_ax, res_ax, chi2_ax
        else:
            return


class PlotLC2:
    """
    Compare two models for a same dataset.
    """

    def __init__(self, plot_lc_base: PlotLC, plot_lc_best: PlotLC) -> None:
        self.plot_lc_base = plot_lc_base
        self.plot_lc_best = plot_lc_best
        assert plot_lc_base.date_offset == plot_lc_best.date_offset
        assert len(plot_lc_base.model_date) == len(plot_lc_best.model_date)
        return

    def plot_model_res(self, ax2, color_dict, obname_list=None, **kwargs):
        if obname_list is None:
            obname_list = self.plot_lc_base.data_dict.keys()
        for obname in obname_list:
            ax2.plot(
                self.plot_lc_base.data_dict[obname]["model_date"]
                - self.plot_lc_base.date_offset,
                self.plot_lc_best.data_dict[obname]["model_mag"]
                - self.plot_lc_base.data_dict[obname]["model_mag"],
                color=color_dict[obname],
                **kwargs,
            )
        return

    # this function is a combination of two models, should be another class.
    def plot_delta_chi2(self, ax3, color_dict, obname_list=None, **kwargs):
        """
        Delta chi2 = chi2(model1) - chi2(model2)
        """
        if obname_list is None:
            obname_list = self.plot_lc_base.data_dict.keys()
        for obname in obname_list:
            chi2_cum = (
                self.plot_lc_base.data_dict[obname]["chi2"]
                - self.plot_lc_best.data_dict[obname]["chi2"]
            )
            ax3.plot(
                self.plot_lc_base.data_dict[obname]["date"]
                - self.plot_lc_base.date_offset,
                chi2_cum,
                color=color_dict[obname],
                **kwargs,
            )
        return

    def plot_all(
        self,
        bin_data=True,
        general_data_fmt={
            "fmt": "o",
            "markerfacecolor": "none",
            "capsize": 3,
            "markersize": 5,
            "markeredgewidth": 0.5,
            "elinewidth": 0.5,
        },
        base_model_fmt={"ls": "-"},
        best_model_fmt={"ls": "--"},
        fig_dpi=300,
    ):
        fig, data_ax, res_ax, chi2_ax = self.plot_lc_base.set_fig_3_axes(dpi=fig_dpi)
        self.plot_lc_base.plot_all(
            bin_data=bin_data,
            general_data_fmt=general_data_fmt,
            fig_dpi=fig_dpi,
            model_fmt=base_model_fmt,
            data_ax=data_ax,
            res_ax=res_ax,
        )
        self.plot_lc_best.plot_model(
            data_ax,
            obname_list=[self.plot_lc_best.event_ri.get_base_ob_id()],
            **best_model_fmt,
        )
        self.plot_model_res(
            res_ax,
            self.plot_lc_base.color_dict,
            [self.plot_lc_best.event_ri.get_base_ob_id()],
            **best_model_fmt,
        )
        self.plot_delta_chi2(
            chi2_ax,
            self.plot_lc_base.color_dict,
            **best_model_fmt,
        )

        data_ax.set_xlim(
            self.plot_lc_base.model_date[0] - self.plot_lc_base.date_offset,
            self.plot_lc_base.model_date[-1] - self.plot_lc_base.date_offset,
        )
        res_ax.set_xlim(
            self.plot_lc_base.model_date[0] - self.plot_lc_base.date_offset,
            self.plot_lc_base.model_date[-1] - self.plot_lc_base.date_offset,
        )
        chi2_ax.set_xlim(
            self.plot_lc_base.model_date[0] - self.plot_lc_base.date_offset,
            self.plot_lc_base.model_date[-1] - self.plot_lc_base.date_offset,
        )
        return fig, data_ax, res_ax, chi2_ax

    def __axes_setting(
        self, ax1, ax2, ax3, plot_date, general_model_fmt, zero_line_color="black"
    ):
        ax1.set_xlim(plot_date[0] - self.date_offset, plot_date[-1] - self.date_offset)
        ax1.set_ylim(18.5, 14.5)
        ax1.legend(fontsize=9)

        if "alpha" in general_model_fmt:
            alpha = general_model_fmt.pop("alpha")
        else:
            alpha = 0.3
        ax2.hlines(
            0,
            plot_date[0] - self.date_offset,
            plot_date[-1] - self.date_offset,
            color=zero_line_color,
            **general_model_fmt,
            alpha=alpha,
        )
        ax2.set_ylim(0.25, -0.25)
        ax2.set_xlim(plot_date[0] - self.date_offset, plot_date[-1] - self.date_offset)

        ax3.hlines(
            0,
            plot_date[0] - self.date_offset,
            plot_date[-1] - self.date_offset,
            color=zero_line_color,
            **general_model_fmt,
            alpha=alpha,
        )
        ax3.set_xlim(plot_date[0] - self.date_offset, plot_date[-1] - self.date_offset)
        pass
