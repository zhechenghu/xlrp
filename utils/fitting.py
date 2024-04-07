import numpy as np
import matplotlib.pyplot as plt
import os
import functools
import pickle
from scipy.optimize import fmin
from emcee import EnsembleSampler
import dynesty
import dynesty.utils as dyfunc

from xlrp.utils.config import WriteInfo, ReadInfo
from xlrp.utils.param import WriteParam, ReadParam

from xlrp import Event
from xlrp import PointLensModel
from xlrp import Data

#########################################
# Functions to calculate ln probability #
#########################################


class MCMCLnProb(object):
    @staticmethod
    def ln_like(
        params_free,
        event: Event,
        params_to_fit: list,
        flux_dict: dict,
    ):
        """
        likelihood function

        Parameters
        ----------
        params_free: list
            list of free parameters for mcmc sampling
        event: Event
        params_to_fit: list
            list of name of parameters to be fitted

        Returns
        -------
        An unnormalized log-likelihood function, float
        """
        event.set_parameters(params_free, params_to_fit)

        # TODO: Add flux_dict again
        ln_like_ = -0.5 * event.get_chi2()
        # Reject severely negative blending
        # This should be put into prior
        if "ogle" in event.flux_dict:
            ogle_f_b = event.flux_dict["ogle"]["f_b"]
            # if ogle_f_b < 0.0:  # type:ignore
            #    punish = -np.exp(-(ogle_f_b**2) / (2 * 0.03**2))  # type:ignore
            if ogle_f_b < -0.2:  # type:ignore
                return -np.inf

        return ln_like_

    @staticmethod
    def ln_prior(
        params_free,
        params_to_fit: list,
        params_scale_loc_dict: dict,
    ):
        """
        priors - we only reject obviously wrong models, in other word, uniform prior

        Parameters
        ----------
        params_free: list
            list of free parameters for mcmc sampling
        event: Event
        params_to_fit: list
            list of name of parameters to be fitted
        """
        for params_name in params_to_fit:
            if (
                params_free[params_to_fit.index(params_name)]
                > params_scale_loc_dict[params_name]["scale"]+params_scale_loc_dict[params_name]["loc"]  # fmt: skip
                or params_free[params_to_fit.index(params_name)]
                < params_scale_loc_dict[params_name]["loc"]
            ):
                # print(f"out of bound: {params_name}")
                return -np.inf

        return 0.0

    @staticmethod
    def ln_prob(
        params_free: np.ndarray,
        event: Event,
        params_to_fit: list[str],
        params_scale_loc_dict: dict[str, tuple[float, float]],
        flux_dict: dict[str, dict[str, float]],
        include_blobs: bool = False,
    ):
        """
        combines likelihood and priors

        Parameters
        ----------
        params_free: list
            list of free parameters for mcmc sampling
        event: Event
        params_to_fit: list
            list of name of parameters to be fitted
        """
        ln_prior_ = MCMCLnProb.ln_prior(
            params_free, params_to_fit, params_scale_loc_dict
        )
        if not np.isfinite(ln_prior_):
            if include_blobs:
                blobs = [np.nan, np.nan] * len(event.all_ob_tup)
                return -np.inf, *blobs
            else:
                return -np.inf
        ln_like_ = MCMCLnProb.ln_like(params_free, event, params_to_fit, flux_dict)

        if include_blobs:
            flux_flatten = []
            for obname in event.all_ob_tup:
                flux_flatten.append(event.flux_dict[obname]["f_s"])
                flux_flatten.append(event.flux_dict[obname]["f_b"])
            return ln_prior_ + ln_like_, *flux_flatten
        else:
            return ln_prior_ + ln_like_

    @staticmethod
    def chi2(params_free, event: Event, params_to_fit: list, flux_dict: dict):
        return -2 * MCMCLnProb.ln_like(params_free, event, params_to_fit, flux_dict)


class DNSLnProb(object):
    @staticmethod
    def nested_sampling_prior(
        unit_cube, params_scale_loc_dict: dict, params_to_fit: list
    ):
        """
        priors\n
        t_0 prior is a uniform distribution between all data points.\n
        u_0 prior is a uniform distribution between -2 and 2.\n
        t_E prior is a uniform distribution between 0 and 1000 days.\n
        pi_E_N prior is a uniform distribution between -3 and 3.\n
        pi_E_E prior is a uniform distribution between -3 and 3.\n

        Parameters
        ----------
        unit_cube: A unit cube generated by dynesty
        params_scale_loc_dict: dict, {parameter name: {'scale': scale, 'loc': loc}}
        params_to_fit: list,
            Campbell elements: ["t_0", "u_0", "t_E", 'e_xi', 'p_xi', 'phi_xi', 'i_xi', 'omega_xi', 'Omega_xi', 'xi_E_N', 'xi_E_E']
            Thiele-Innes elements: ["t_0", "u_0", "t_E", 'e_xi', 'p_xi', 'phi_xi', "A_xi", "B_xi", "F_xi", "G_xi", "theta_xi"]

        """
        # Transform unit cube to real parameter space
        prior_transform_list = [
            (
                params_scale_loc_dict[param_name]["loc"]
                + params_scale_loc_dict[param_name]["scale"] * unit_cube[param_i]
            )
            for param_i, param_name in enumerate(params_to_fit)
        ]
        return np.array(prior_transform_list)

    @staticmethod
    def nested_sampling_ln_likelihood(
        params_free,
        event: Event,
        params_to_fit: list,
        flux_dict: dict,
    ):
        """
        This function can calculate unnormalized log-likelihood function,
        as well as the prior probability of the parameters.
        """
        ####################
        ## Log likelihood ##
        ####################
        for idx, name in enumerate(params_to_fit):
            event.model.parameters[name] = params_free[idx]
            for sapce_ob in event.sapce_ob_tup:
                event.space_model_dict[sapce_ob].parameters[name] = params_free[idx]
        # TODO: Add flux_dict again
        ln_like_ = -0.5 * event.get_chi2()

        ######################
        ## Prior likelihood ##
        ######################
        # Typically prior should be in prior(),
        # but prior for nested sampling is just for converting unit cube to real uniform distributions
        # So prior is here.
        # But you may still wonder: why you put prior after ln_like_?
        # The answer is: Only after you calculated chi2 you can know the blending.

        # 1. Reject severely negative blending
        # TODO: This should be put into prior
        # punish_blend = -1.0
        if "ogle" in event.flux_dict:
            ogle_f_b = event.flux_dict["ogle"]["f_b"]
            # if ogle_f_b < 0.0:  # type:ignore
            #    punish_blend = -np.exp(-(ogle_f_b**2) / (2 * 0.03**2))  # type:ignore
            if ogle_f_b < -0.2:  # type:ignore
                return -np.inf

        # 2. Prior of orbital period
        # Currently not used, becauese the real PDF prefers small orbital period.
        # The two number below are from Duchêne & Kraus 2013
        # period = event.model.parameters["P_xi"]
        # log_period = np.log10(period)
        # log_p_avg = 5
        # sigma_log_p = 2.3
        # punish_period = (
        #    10                           # This line is a weight of period punishment
        #    * 1 / (period*np.log(10))    # This line is to convert dlogP to dP
        #    # The following two lines are to calculate the PDF of logP
        #    * 1 / (sigma_log_p * np.sqrt(2 * np.pi))
        #    * np.exp(-((log_period - log_p_avg) ** 2) / (2 * sigma_log_p**2))
        # )  # fmt: skip
        # punish = punish_blend  # + punish_period
        # Sometimes will use the full normalized ln probability below
        # return -np.sum(((model_flux-data_flux)/(2*flux_err))**2+0.5*np.log(2*np.pi)+np.log(flux_err))
        return ln_like_


class FitUtils:
    @staticmethod
    def get_params_to_fit(event_type: str, params_fix: list | None = None):
        if event_type == "std":
            params_to_fit = ["t_0", "u_0", "t_E"]
        elif event_type == "prlx":
            params_to_fit = ["t_0", "u_0", "t_E", "pi_E_N", "pi_E_E"]
        elif event_type == "xlrp_circ":
            params_to_fit = ["t_0", "u_0", "t_E", "p_xi", "phi_xi", "i_xi", "xi_E_N", "xi_E_E"]  # fmt: skip
        elif event_type == "xlrp_circ_ti":
            params_to_fit = ["t_0", "u_0", "t_E", "p_xi", "A_xi", "B_xi", "F_xi", "G_xi"]  # fmt: skip
        elif event_type == "xlrp_cpb":
            params_to_fit = ["t_0", "u_0", "t_E",
                             "e_xi", "p_xi", "phi_xi", "i_xi", "omega_xi", 
                             "xi_E_N", "xi_E_E"]  # fmt: skip
        elif event_type == "xlrp_ti":
            params_to_fit = ["t_0", "u_0", "t_E",
                             "e_xi", "p_xi", "phi_xi",
                             "A_xi", "B_xi", "F_xi", "G_xi", "theta_xi"]  # fmt: skip
        else:
            print(event_type)
            raise ValueError(
                "Event type should be 'std', 'prlx' 'xlrp_circ', 'xlrp_circ_ti', 'xlrp_cpb' or 'xlrp_ti'."
            )

        if params_fix is not None:
            for param in params_fix:
                params_to_fit.remove(param)

        return params_to_fit

    @staticmethod
    def get_result_path_without_suff(
        event_dir: str, event_type: str, fit_method: str, neg_u0: bool = False
    ):
        if neg_u0:
            return os.path.join(event_dir, f"{event_type}_{fit_method}_neg")
        else:
            return os.path.join(event_dir, f"{event_type}_{fit_method}_pos")

    @staticmethod
    def get_full_params_scale_loc_dict(
        t_range,
        params_scale_loc_dict: dict | None = None,
        neg_u0: bool = False,
        free_u0: bool = False,
        neg_u0_2: bool = False,
        free_u0_2: bool = False,
    ):
        try:
            jd_start = t_range[0]
            jd_end = t_range[1]
        except TypeError:
            jd_start = 2450000  # corresponds to 1968-05-23
            jd_end = 2490000  # corresponds to 2105
        default_params_scale_loc_dict = {
            "t_0": {"scale": jd_end - jd_start, "loc": jd_start},
            "t_0_2": {"scale": jd_end - jd_start, "loc": jd_start},
            "u_0": {"scale": 10.0, "loc": 0.0},
            "u_0_2": {"scale": 10.0, "loc": 0.0},
            "t_E": {"scale": 500.0, "loc": 0.0},
            "pi_E_N": {"scale": 20.0, "loc": -10.0},
            "pi_E_E": {"scale": 20.0, "loc": -10.0},
            "xi_E_N": {"scale": 2.0, "loc": -1.0},
            "xi_E_E": {"scale": 2.0, "loc": -1.0},
            "p_xi": {"scale": 500, "loc": 5.0},
            "e_xi": {"scale": 1.0, "loc": 0.0},
            "i_xi": {"scale": np.pi, "loc": 0.0},
            "Omega_xi": {"scale": 2 * np.pi, "loc": 0.0},
            "omega_xi": {"scale": 2 * np.pi, "loc": 0.0},
            "phi_xi": {"scale": 2 * np.pi, "loc": 0.0},
            "A_xi": {"scale": 4.0, "loc": -2.0},
            "B_xi": {"scale": 4.0, "loc": -2.0},
            "F_xi": {"scale": 4.0, "loc": -2.0},
            "G_xi": {"scale": 4.0, "loc": -2.0},
            "q_xi": {"scale": 0.99, "loc": 0.005},
            "eta": {"scale": 99.0, "loc": 1.0},
            "qf_xi": {"scale": 0.99, "loc": 0.005},
            "qf_ogle": {"scale": 0.99, "loc": 0.005},
            "qf_ogleV": {"scale": 0.99, "loc": 0.005},
            "qf_moa": {"scale": 0.99, "loc": 0.005},
            "qf_danishV": {"scale": 0.99, "loc": 0.005},
            "qf_danishZ": {"scale": 0.99, "loc": 0.005},
            "qf_spitzer": {"scale": 0.99, "loc": 0.005},
            "theta_xi": {"scale": 2 * np.pi, "loc": 0.0},
        }
        if neg_u0:
            default_params_scale_loc_dict["u_0"]["loc"] = -1.0
        elif free_u0:
            default_params_scale_loc_dict["u_0"]["loc"] = -1.0
            default_params_scale_loc_dict["u_0"]["scale"] = 2.0
        if neg_u0_2:
            default_params_scale_loc_dict["u_0_2"]["loc"] = -1.0
        elif free_u0_2:
            default_params_scale_loc_dict["u_0_2"]["loc"] = -1.0
            default_params_scale_loc_dict["u_0_2"]["scale"] = 2.0

        if params_scale_loc_dict is None:
            return default_params_scale_loc_dict

        for key in params_scale_loc_dict.keys():
            default_params_scale_loc_dict[key] = params_scale_loc_dict[key].copy()
        return default_params_scale_loc_dict

    @staticmethod
    def build_mylens_event(
        event_ri: ReadInfo,
        neg_u0: bool | None = None,
        event_rp: ReadParam | None = None,
        event_type: str | None = None,
        params_dict: dict | None = None,
    ) -> Event:
        """
        Build mylens event from ReadInfo and ReadParam or params_dict.

        Parameters
        ----------
        event_ri : ReadInfo
            ReadInfo
        event_rp : ReadParam|None
            ReadParam
        event_type : str|None
            needed only when event_rp is specified.
        params_dict : dict|None
            a dictionary of parameters for the model.

        Returns
        -------
            An Event object.
        """
        bad_dict = event_ri.get_bad_path_dict()
        have_bad = False
        for bad_file in bad_dict.values():
            if os.path.exists(bad_file):
                have_bad = True
        if have_bad:
            my_data = Data(
                errfac_dict=event_ri.get_errfac_dict(),
                data_file_dict=event_ri.get_data_path_dict(),
                bad_file_dict=event_ri.get_bad_path_dict(),
                obname_list=event_ri.get_observatories(),
                t_range=event_ri.get_t_range(),
            )
        else:
            my_data = Data(
                errfac_dict=event_ri.get_errfac_dict(),
                data_file_dict=event_ri.get_data_path_dict(),
                obname_list=event_ri.get_observatories(),
                t_range=event_ri.get_t_range(),
            )
        coords_dict = event_ri.get_coords(return_type="dict")
        if params_dict is None:
            params_dict = event_rp.get_params(event_type, include_t_0_par=True, include_t_ref=True, neg_u0=neg_u0)  # type: ignore
        my_model = PointLensModel(params_dict, ra=coords_dict["ra"], dec=coords_dict["dec"])  # type: ignore
        return Event(
            my_model,
            my_data,
            zero_blend_dict=event_ri.get_zero_blend_dict(),
        )

    @staticmethod
    def downhill_fitting(
        event: Event,
        params_to_fit: list,
        downhill_opt_dict={"maxiter": 3000, "maxfun": 10000},
        init_params=None,
        print_info=False,
        flux_dict: dict = {"f_s": None, "f_b": None},
        return_full=False,
    ):
        """
        Use simplex method to fit microlensing events.

        Parameters
        ----------
        event: Event
        params_to_fit: list

        Return
        ------
        (best params, best chi2)
        """
        if init_params is None:
            obname = event.all_ob_tup[0]
            init_params = [
                event.model_dict[obname].parameters[param_name]
                for param_name in params_to_fit
            ]
        parmbest, chi2min, iter_num, funcalls, _warnflag, _allevcs = fmin(  # type: ignore
            MCMCLnProb.chi2,
            init_params,
            args=(event, params_to_fit, flux_dict),
            full_output=True,
            disp=print_info,
            retall=True,
            maxiter=downhill_opt_dict["maxiter"],
            maxfun=downhill_opt_dict["maxfun"],
        )
        if print_info:
            print("best parameters: ")
            for i in range(len(parmbest)):
                print("%s = %.6f" % (params_to_fit[i], parmbest[i]))
            print(f"Iter num: {iter_num}")
            print(f"Function ev: {funcalls}")
        if return_full:
            return parmbest, chi2min, iter_num, funcalls, _warnflag, _allevcs
        else:
            return parmbest, chi2min

    @staticmethod
    def emcee_fitting(
        event: Event,
        params_to_fit: list,
        emcee_opt_dict: dict,
        chain_path: str,
        params_scale_loc_dict: dict | None = None,
        flux_dict: dict = {"f_s": None, "f_b": None},
        include_blobs: bool = False,
        return_chain: bool = False,
        print_info=False,
    ):
        """
        Use emcee to fit microlensing events.
        It seems that emcee does not record all the samples.
        The best samples recorded by the event seems always smaller than the best sample in the dataframe.

        Parameters
        ----------
        event : Event
            the event object
        params_to_fit : list
            list of parameter names to fit
        emcee_opt_dict : dict
            a dictionary of options for the emcee fitting.
        chain_path : str
            The path to save the chain to.
        print_info, optional
            If True, print the best fit parameters and their errors.

        Returns
        -------
            (best params, best params err, best chi2)
        value.

        """
        ndim = len(params_to_fit)
        nwalkers = emcee_opt_dict["nwalkers"]
        # Build the dtype for the blobs
        blob_dtype = []
        for obname in event.all_ob_tup:
            blob_dtype.append((obname + "_fs", float))
            blob_dtype.append((obname + "_fb", float))

        # The initial positions of the walkers should include the init_params
        # The init_params is usually the best fit parameters from other methods.
        # The obname does not matter, it is the same across all models.
        obname = event.all_ob_tup[0]
        init_params = [
            event.model_dict[obname].parameters[param_name]
            for param_name in params_to_fit
        ]
        pos = [init_params + (1e-3 * np.random.randn(ndim)) for _ in range(nwalkers - 1)]  # type: ignore
        pos += [init_params]
        sampler = EnsembleSampler(
            nwalkers,
            ndim,
            MCMCLnProb.ln_prob,
            blobs_dtype=blob_dtype,
            args=(
                event,
                params_to_fit,
                params_scale_loc_dict,
                flux_dict,
                include_blobs,
            ),
        )
        event.get_chi2()  # initialize the event, otherwise the first chi2 is none?
        ## run EMCEE ##
        nburn = emcee_opt_dict["nburn"]
        nstep = emcee_opt_dict["nstep"]
        sampler.run_mcmc(pos, nburn + nstep)
        ## save EMCEE results ##
        sampler.chain.reshape((-1, ndim))
        params_chain = sampler.chain[:, nburn:, :].reshape((-1, ndim), order="F")
        chi2_chain = -2 * sampler.lnprobability[:, nburn:].reshape(-1, order="F")
        header_str = ",".join(params_to_fit) + ",chi2"
        save_chain = np.vstack([params_chain.T, chi2_chain]).T
        if chain_path is not None:
            if include_blobs:
                pass
                # TODO: save the blobs
                # fmt: off
                flux_chain_full = sampler.get_blobs() 
                flux_chain_flat = np.array(flux_chain_full[nburn:, :]).reshape(-1, order="F") # type: ignore
                flux_chain_arr = np.zeros((flux_chain_flat.shape[0], 2 * len(event.all_ob_tup)))
                for dtype in blob_dtype:
                    flux_chain_arr[:, blob_dtype.index(dtype)] = flux_chain_flat[dtype[0]].reshape(-1)
                save_chain = np.vstack([params_chain.T, flux_chain_arr.T, chi2_chain]).T
                header_str = ",".join(params_to_fit) + ","
                header_str += ",".join([f"{obname}_fs,{obname}_fb" for obname in event.all_ob_tup])
                header_str += ",chi2"
                np.savetxt(chain_path, save_chain, delimiter=",", header=header_str, comments="")
                # fmt: on
            else:
                np.savetxt(
                    chain_path,
                    save_chain,
                    delimiter=",",
                    header=header_str,
                    comments="",
                )
        ## Find the best fit ##
        params_best = params_chain[np.argmin(chi2_chain), :]
        params_errs = np.percentile(params_chain, q=[16, 84], axis=0)
        if print_info:
            print("Best parameters: ")
            for i in range(len(params_best)):
                print(
                    f"{params_to_fit[i]} = {params_best[i]:.6f} + {params_errs[0, i] - params_errs[1, i]:.6f}"
                )
        if return_chain:
            return params_best, params_errs, np.min(chi2_chain), save_chain
        return params_best, params_errs, np.min(chi2_chain)

    @staticmethod
    def dynesty_fitting(
        event: Event,
        params_to_fit: list,
        params_scale_loc_dict: dict,
        dynesty_opt_dict: dict,
        result_path: str,
        flux_dict: dict = {"f_s": None, "f_b": None},
        print_progress=False,
    ):
        """Use dynesty to fit microlensing events.

        Parameters
        ----------
        event : Event
            Event
        params_to_fit : list
            list of strings, the parameters to fit.
        params_scale_loc_dict : dict
            a dictionary of the form {param_name: (scale, loc)}
        dynesty_opt_dict : dict
            a dictionary of options for the dynesty sampler.
        result_path : str
            The path to the file where the results will be saved.

        Returns
        -------
            (best params, best params err, best chi2)

        """
        prior_unit_transform_func = functools.partial(
            DNSLnProb.nested_sampling_prior,
            params_scale_loc_dict=params_scale_loc_dict,
            params_to_fit=params_to_fit,
        )
        ln_likelihood = functools.partial(
            DNSLnProb.nested_sampling_ln_likelihood,
            event=event,
            params_to_fit=params_to_fit,
            flux_dict=flux_dict,
        )
        n_dim = len(params_to_fit)
        sampler = dynesty.DynamicNestedSampler(
            ln_likelihood,
            prior_unit_transform_func,
            n_dim,
            nlive=dynesty_opt_dict["nlive"],
            bound=dynesty_opt_dict["bound"],
            sample=dynesty_opt_dict["sample"],
        )
        sampler.run_nested(print_progress=print_progress)
        results = sampler.results
        # Save the results to a file.
        if result_path is not None:
            with open(result_path, "wb") as fp:
                pickle.dump(results, fp)
        # Get the best fit parameters and their uncertainties.
        samples = results.samples
        weights = np.exp(results.logwt - results.logz[-1])
        posterior = dyfunc.resample_equal(samples, weights)
        param_err = np.std(posterior, axis=0)
        params_best = np.array([event.best_params[param_name] for param_name in params_to_fit])  # type: ignore

        return params_best, param_err, event.best_chi2

    @staticmethod
    def _set_ctrl_params(
        event_cfg_path: str,
        fit_method: str,
        zero_blend: bool,
        reset_errfac: bool,
        emcee_nwalkers=30,
        emcee_nstep=1000,
        emcee_nburn=1000,
        dynesty_nlive=500,
        dynesty_bound="multi",
        dynesty_sample="rwalk",
    ):
        ### set control params ###
        event_wi = WriteInfo(event_cfg_path)
        ## Reset every thing to general settings of this fit type and fit method.
        event_wi.reset_control_params()
        event_wi.set_fitting_method(fit_method)
        event_wi.set_emcee_options(
            n_walkers=emcee_nwalkers,
            n_steps=emcee_nstep,
            n_burn=emcee_nburn,
        )
        event_wi.set_dynesty_options(
            nlive=dynesty_nlive,
            bound=dynesty_bound,
            sample=dynesty_sample,
        )
        # If want to start from begining, reset error factor
        if reset_errfac:
            event_wi.set_errfac([0.00, 1.0])
        # Some times I will try zero blend.
        if zero_blend:
            event_wi.set_zero_blend(zero_blend=True)
        else:
            event_wi.set_zero_blend(zero_blend=False)
        # After setting the control params, we can get other params
        return ReadInfo(event_cfg_path)

    @staticmethod
    def _build_and_fit(
        event_ri: ReadInfo,
        event_rp: ReadParam,
        event_type: str,
        fit_method: str,
        emcee_opt_dict: dict,
        dynesty_opt_dict: dict,
        neg_u0: bool,
        flux_dict: dict = {"f_s": None, "f_b": None},
        include_blobs: bool = True,
        params_to_fit: list | None = None,
        params_scale_loc_dict: dict | None = None,
    ):
        # 1.1 Build event
        if params_to_fit is None:
            params_to_fit = FitUtils.get_params_to_fit(event_type)
        event_to_fit = FitUtils.build_mylens_event(
            event_ri, neg_u0, event_rp, event_type
        )
        event_dir = event_ri.get_event_dir()
        # 1.2 Build the params_scale_loc_dict
        params_scale_loc_dict = FitUtils.get_full_params_scale_loc_dict(
            t_range=(event_to_fit.dataset.date[0], event_to_fit.dataset.date[-1]),
            params_scale_loc_dict=params_scale_loc_dict,
            neg_u0=neg_u0,
        )

        # Fit the event and save the result
        res_path_without_suff = FitUtils.get_result_path_without_suff(
            event_dir, event_type, fit_method, neg_u0
        )
        if fit_method == "downhill":
            best_params, best_chi2 = FitUtils.downhill_fitting(
                event_to_fit, params_to_fit
            )
            best_params_err = None
        elif fit_method == "emcee":
            best_params, best_params_err, best_chi2 = FitUtils.emcee_fitting(
                event_to_fit,
                params_to_fit,
                emcee_opt_dict,
                res_path_without_suff + ".dat",
                params_scale_loc_dict,
                flux_dict,
                include_blobs,
            )
        elif fit_method == "dynesty":
            best_params, best_params_err, best_chi2 = FitUtils.dynesty_fitting(
                event_to_fit,
                params_to_fit,
                params_scale_loc_dict,
                dynesty_opt_dict,
                res_path_without_suff + ".pkl",
                flux_dict,
            )
        else:
            raise ValueError("Fit method should be 'downhill', 'emcee' or 'dynesty'.")
        return best_params, best_params_err, best_chi2

    @staticmethod
    def _write_params(
        event_ri: ReadInfo,
        event_rp: ReadParam,
        event_wp: WriteParam,
        event_type: str,
        best_params: np.ndarray,
        params_err: np.ndarray | None,
        best_chi2: float,
        neg_u0: bool,
        params_to_fit: list | None = None,
    ):
        if params_to_fit is None:
            params_to_fit = FitUtils.get_params_to_fit(event_type)
        best_params_dict = {
            key: float(best_params[i]) for i, key in enumerate(params_to_fit)
        }
        if params_err is not None:
            params_err_dict = {
                f"{key}_err": float(params_err[i])
                for i, key in enumerate(params_to_fit)
            }
        else:
            params_err_dict = {f"{key}_err": np.nan for key in params_to_fit}

        # there are some redundant parameters, fill them with 0
        if event_type == "xlrp_cpb":
            best_params_dict["Omega_xi"] = 0.0
            params_err_dict["Omega_xi_err"] = 0.0
        if event_type == "xlrp_ti":
            best_params_dict["theta_xi"] = 0.0
            params_err_dict["theta_xi_err"] = 0.0

        # Maybe not all parameters is fitted, fill the fixed parameters with previous values
        prev_params_dict = event_rp.get_params(event_type, False, False, neg_u0)
        for key in prev_params_dict.keys():
            if key not in best_params_dict.keys():
                best_params_dict[key] = prev_params_dict[key]
                params_err_dict[f"{key}_err"] = 0.0
        # set params and error
        event_wp.set_params(best_params_dict, event_type, neg_u0)
        event_wp.set_err(params_err_dict, event_type, neg_u0)
        best_params_for_event = best_params_dict.copy()
        if event_type == "prlx":
            best_params_for_event["t_0_par"] = event_rp.get_t_0_par()
        elif (
            event_type == "xlrp_circ"
            or event_type == "xlrp_circ_ti"
            or event_type == "xlrp_cpb"
            or event_type == "xlrp_ti"
        ):
            best_params_for_event["t_ref"] = event_rp.get_t_ref()
        temp_event = FitUtils.build_mylens_event(
            event_ri, neg_u0, params_dict=best_params_for_event
        )
        temp_event.get_chi2()
        flux_dict = {}
        for obname in temp_event.obname_list:
            flux_dict[obname] = {}
            flux_dict[obname]["f_s"] = float(temp_event.flux_dict[obname]["f_s"])
            flux_dict[obname]["f_b"] = float(temp_event.flux_dict[obname]["f_b"])
        event_wp.set_flux_chi2(flux_dict, best_chi2, event_type, neg_u0)
        return

    @staticmethod
    def _plot_results(
        event_ri: ReadInfo,
        event_rp: ReadParam,
        event_type: str,
        neg_u0: bool,
        model_kwargs: dict,
        data_kwargs: dict,
        bad_data_kwargs: dict,
    ):
        params_dict = event_rp.get_params(event_type)
        temp_event = FitUtils.build_mylens_event(event_ri, neg_u0, event_rp, event_type)
        chi2 = temp_event.get_chi2()
        fs, fb = temp_event.fs, temp_event.fb

        plt.figure(figsize=(10, 5))
        temp_event.plot_events(model_kwargs, data_kwargs, bad_data_kwargs)
        event_name = event_ri.get_short_name()
        plt.title(f"{event_name} - {event_type} - chi2={chi2:.2f}")
        event_dir = os.path.dirname(event_ri.get_param_path())
        plt.savefig(os.path.join(event_dir, f"{event_name}_{event_type}.png"))

    @staticmethod
    def single_event_fitting_main(
        event_cfg_path: str,
        event_type="prlx",
        fit_method="emcee",
        params_scale_loc_dict: dict | None = None,
        neg_u0=False,
        flux_dict: dict = {"f_s": None, "f_b": None},
        include_blobs=False,
        params_to_fit: list | None = None,
    ):
        """
        A single event fitting function.

        Parameters
        ----------
        all_df: DataFrame, A dataframe containing all events' parameters.
        event_i: int, The index of the event to be fitted.
        fit_type: str, The type of fitting, should be 'std', 'prlx' or 'xlrp'.
        fit_method: str, The fitting method, should be 'downhill', 'emcee' or 'dynesty'.
        emcee_opt_dict: dict, The options for emcee.
        dynesty_opt_dict: dict, The options for dynesty.
        reset_errfac: bool, Whether to reset the errfac to [0.003, 1.0].
        reu0: bool, Whether to reverse the sign of u_0.
        """

        event_ri = ReadInfo(event_cfg_path)
        event_rp = ReadParam(event_ri.get_param_path())
        emcee_opt_dict = event_ri.get_emcee_opt_dict()
        dynesty_opt_dict = event_ri.get_dynesty_opt_dict()
        best_params, best_params_err, best_chi2 = FitUtils._build_and_fit(
            event_ri,
            event_rp,
            event_type,
            fit_method,
            emcee_opt_dict,
            dynesty_opt_dict,
            neg_u0,
            flux_dict,
            include_blobs,
            params_to_fit,
            params_scale_loc_dict,
        )
        event_wp = WriteParam(event_ri.get_param_path())
        FitUtils._write_params(
            event_ri,
            event_rp,
            event_wp,
            event_type,
            best_params,
            best_params_err,
            float(best_chi2),
            neg_u0,
            params_to_fit,
        )
        plot_data_args = {
            "fmt": "o",
            "alpha": 0.5,
            "color": "royalblue",
            "markersize": 2,
        }
        plot_bad_args = {"marker": "x", "color": "r", "alpha": 0.8, "s": 10}
        plot_model_args = {"color": "k", "linewidth": 1, "label": "Best-fit"}
        # FitUtils._plot_results(
        #     event_ri,
        #     event_rp,
        #     event_type,
        #     plot_model_args,
        #     plot_data_args,
        #     plot_bad_args,
        # )
        return
