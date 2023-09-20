import numpy as np
from astropy.coordinates import get_body_barycentric_posvel
from astropy.coordinates import get_body_barycentric
from astropy.time import Time
from scipy.interpolate import interp1d


class PointLensModel(object):
    """
    This is the microlensing model for a point lens and (maybe) multiple source.

    Parameters
    ----------
    parameters : dict, all microlensing parameters needed to model a event.
        t_0_par and t_ref should also be included.
    ra : str, optional, in HH:mm:ss, needed when considering parallax effect.
    dec: str, optional, in DD:mm:ss, needed when considering parallax effect.
    ephe : np.ndarray, optional, ephemeris of the satellite.
        Should contain 4 columns: JD, RA, DEC, geocentric distance; (RA,DEC) in deg
        needed when considering satellite parallax effect.
        In principle, this argument should not be added manually, but should be
        automatically added by the Event class.
    obname : str, optional, name of the observatory, needed when considering 2 source
        xallarap effect. Then the flux ratio qf should be different in different
        observatory(band). The default is None.
    """

    def __init__(
        self,
        parameters: dict,
        ra: str | None = None,
        dec: str | None = None,
        ephemeris: np.ndarray | None = None,
        obname: str | None = None,
    ):
        self.obname = obname
        self.parameters = parameters.copy()
        self.parameter_set_enabled = ["std"]
        self.delta_sun = None
        self.t_0_par = None
        self.ra = None
        self.dec = None
        self.alpha = None
        self.delta = None
        self.ephemeris = None
        self.neg_delta_sate: np.ndarray = None
        self.t_ref = None

        self.__process_ra_dec(ra, dec)
        self.__process_ephemeris(ephemeris)
        self.__process_parameters()

        # Model light curve related
        self.jds: np.ndarray = np.array([])
        self.trajectory = np.array([])
        self.magnification = np.array([])
        self.model_flux = np.array([])
        self.model_mag = np.array([])

    def __process_ra_dec(self, ra, dec):
        self.ra = ra
        self.dec = dec
        if self.ra is None or self.dec is None:
            self.alpha = None
            self.delta = None
            return
        else:
            ra_sep = np.array(ra.split(":")).astype(float)
            self.alpha = (ra_sep[0] + ra_sep[1] / 60.0 + ra_sep[2] / 3600.0) * 15.0
            dec_sep = np.array(dec.split(":")).astype(float)
            if dec_sep[0] < 0:
                self.delta = dec_sep[0] - dec_sep[1] / 60.0 - dec_sep[2] / 3600.0
            else:
                self.delta = dec_sep[0] + dec_sep[1] / 60.0 + dec_sep[2] / 3600.0
        return

    def __process_ephemeris(self, ephemeris):
        if ephemeris is None:
            self.ephemeris = None
        else:
            self.ephemeris = ephemeris.copy()

    def __process_parameters(self):
        """
        Three main categories, parallax, xallarap, and static binary source.
        The xallarap effect and the static binary source effect cannot be
        enabled at the same time.
        """
        if "pi_E_N" in self.parameters and "pi_E_E" in self.parameters:
            self.__process_parallax()
        if "p_xi" in self.parameters.keys():
            self.__process_xallarap()
        elif "t_0_2" in self.parameters.keys() and "u_0_2" in self.parameters.keys():
            self.parameter_set_enabled.append("bins")

    def __process_parallax(self):
        self.parameter_set_enabled.append("prlx")
        try:
            self.t_0_par = self.parameters.pop("t_0_par")
        except KeyError:
            raise ValueError("t_0_par, required for parallax, is not defined.")
        if self.ra is None or self.dec is None:
            raise ValueError("ra and dec, required for parallax, are not defined.")

    def __process_xallarap(self):
        self.__process_t_ref()
        if "e_xi" not in self.parameters.keys():
            if "i_xi" in self.parameters.keys() and "phi_xi" in self.parameters.keys():
                self.parameter_set_enabled.append("xlrp_circ")
            elif (
                "A_xi" in self.parameters.keys()
                and "B_xi" in self.parameters.keys()
                and "F_xi" in self.parameters.keys()
                and "G_xi" in self.parameters.keys()
            ):
                self.__process_xallarap_ti_or_ti_2s()
        elif "e_xi" in self.parameters.keys():
            self.__process_xallarap_cpb()

    def __process_xallarap_ti_or_ti_2s(self):
        if "q_xi" in self.parameters.keys():
            self.parameter_set_enabled.append("xlrp_circ_ti_2s")
        else:
            self.parameter_set_enabled.append("xlrp_circ_ti")

    def __process_xallarap_cpb(self):
        if (
            "i_xi" in self.parameters.keys()
            and "Omega_xi" in self.parameters.keys()
            and "omega_xi" in self.parameters.keys()
        ):
            self.parameter_set_enabled.append("xlrp_cpb")

    def __process_t_ref(self):
        try:
            self.t_ref = self.parameters.pop("t_ref")
        except KeyError:
            raise ValueError("t_ref, required for xallarap, is not defined.")

    def __unit_E_N(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the unit North and East Vector

        This function define the North and East vectors projected on the sky plane
        perpendicular to the line of sight (i.e the line define by RA,DEC of the event).
        """
        # fmt: off
        alpha_rad, delta_rad = np.deg2rad(self.alpha), np.deg2rad(self.delta)
        # this is the target direction in the xyz coordinate system
        target = np.array(
            [
                np.cos(delta_rad) * np.cos(alpha_rad),  # x value, in the direction of the East
                np.cos(delta_rad) * np.sin(alpha_rad),  # y value, in the direction perpendicular to the East
                np.sin(delta_rad),
            ]
        )  # z value, in the direction of the North
        # Then project the east vector to the sky plane
        north = np.array([0.0, 0.0, 1.0])
        east_projected = np.cross(north, target)
        east_projected = east_projected / np.linalg.norm(east_projected)  # is [-np.sin(alpha_rad), np.cos(alpha_rad), 0.0]
        # Then project the north vector to the sky plane
        north_projected = np.cross(target, east_projected)
        # fmt: on
        return east_projected, north_projected

    def __annual_parallax(self, time_format="jd"):
        """
        Compute shift of the Sun at given times.
        During fitting, this function should be called only once.
        """
        # set the reference time to the posterior median t0 from the linear
        # trajectory model
        t_ref, jds = self.t_0_par, self.jds

        east_projected, north_projected = self.__unit_E_N()
        time_ref = Time(t_ref, format=time_format)
        earth_pos_ref, earth_vel_ref = get_body_barycentric_posvel("earth", time_ref)
        sun_pos_ref = -earth_pos_ref.get_xyz().value
        sun_vel_ref = -earth_vel_ref.get_xyz().value

        time_jds = Time(jds, format=time_format)
        earth_pos_arr = get_body_barycentric(
            "earth", time_jds
        )  # This is faster when only calculate the position
        delta_sun_project_N = []
        delta_sun_project_E = []
        for time, earth_pos in zip(jds, earth_pos_arr):
            sun_pos = -earth_pos.get_xyz().value
            delta_sun = sun_pos - sun_pos_ref - (time - t_ref) * sun_vel_ref
            delta_sun_project_N.append(np.dot(delta_sun, north_projected))
            delta_sun_project_E.append(np.dot(delta_sun, east_projected))

        return np.array([delta_sun_project_N, delta_sun_project_E]).T

    def __satellite_parallax(self):
        # fmt: off
        east, north = self.__unit_E_N()
        # find the satelite position ##
        # the satellite orbit file should have four columns:
        # JD,RA,DEC,geocentric distance; (RA,DEC) in deg
        jd_sat = np.array(self.ephemeris[:, 0])
        if jd_sat[0] < 2450000:
            jd_sat = jd_sat + 2450000
        ra_sat = self.ephemeris[:, 1]
        dec_sat = self.ephemeris[:, 2]
        dis_sat = self.ephemeris[:, 3]
        radian = np.pi / 180.0

        # we first compute xyz position of the satellite
        # here negtive is just like the delta sun in the annual parallax
        neg_eph_xyz = np.zeros((len(self.ephemeris), 3))
        neg_eph_xyz[:, 0] = (-dis_sat * np.cos(ra_sat * radian) * np.cos(dec_sat * radian))
        neg_eph_xyz[:, 1] = (-dis_sat * np.sin(ra_sat * radian) * np.cos(dec_sat * radian))
        neg_eph_xyz[:, 2] = -dis_sat * np.sin(dec_sat * radian)

        # and then interpolate the xyz position of the satellite
        interp_kwargs = dict(kind="cubic", fill_value="extrapolate")
        neg_sate_x = interp1d(jd_sat, neg_eph_xyz[:, 0], **interp_kwargs)(self.jds)
        neg_sate_y = interp1d(jd_sat, neg_eph_xyz[:, 1], **interp_kwargs)(self.jds)
        neg_sate_z = interp1d(jd_sat, neg_eph_xyz[:, 2], **interp_kwargs)(self.jds)
        neg_sate_xyz = np.array([neg_sate_x, neg_sate_y, neg_sate_z]).T

        # project the satellite position to the sky plane
        neg_d_sat_n, neg_d_sat_e = [], []
        for jd_i in range(len(self.jds)):
            neg_d_sat_n.append(np.sum(neg_sate_xyz[jd_i] * north))
            neg_d_sat_e.append(np.sum(neg_sate_xyz[jd_i] * east))

        # fmt: on
        return np.array([np.array(neg_d_sat_n), np.array(neg_d_sat_e)]).T

    @staticmethod
    def __compute_eccentirc_anomaly(
        l: np.ndarray | float, e: float, l_ref: float = 0, max_iter=5
    ):
        """
        Compute the true anomaly from the mean anomaly and eccentricity with Newton-Raphson method.
        If l_ref is provided, then the true anomaly of l_ref is also computed.

        Parameters
        ----------
        l: `np.ndarray` or `float`
            The mean anomaly
        e: `float`
            The eccentricity
        l_ref: `float`, optional
            The mean anomaly of the reference time, by default 0.
        max_iter: `int`, optional
            The maximum number of iterations, by default 5, which is typically enough.

        Returns
        -------
        (np.ndarray, float), the eccentric anomaly and the eccentric anomaly of l_ref
        """
        # fmt: off
        u_0 = l  # Initial guess of corresponding eccentric anomaly
        u_0_ref = l_ref  # Initial guess of corresponding eccentric anomaly when t=t0
        for _istep in range(max_iter):  # Newton-Raphson method
            # Kepler's equation: u - e * np.sin(u) - l = 0
            # The derivative of Kepler's equation: 1 - e * np.cos(u)
            u_0 = u_0 - (u_0 - e * np.sin(u_0) - l) / (1 - e * np.cos(u_0))
            u_0_ref = u_0_ref - (u_0_ref-e*np.sin(u_0_ref)-l_ref) / (1-e*np.cos(u_0_ref))
        # fmt: on
        return u_0, u_0_ref

    @staticmethod
    def __compute_position_in_orbital_plane(u: np.ndarray | float, e: float):
        """
        Compute the position of the source in the orbital plane.
        I consider the direction of z axis as the direction of the angular momentum vector.
        If inclination is 0, then the orbital plane is the sky plane,
        and the line of sight is the z axis.

        Parameters
        ----------
        u: `np.ndarray` or `float`
            The eccentric anomaly
        e: `float`
            The eccentricity

        Returns
        -------
        np.ndarray of shape (3, len(u)), the position of the source in the orbital plane
        """
        xlrp_arr_x = np.cos(u) - e
        xlrp_arr_y = np.sqrt(1 - e**2) * np.sin(u)
        xlrp_arr_z = 0 * u
        return np.array([xlrp_arr_x, xlrp_arr_y, xlrp_arr_z])

    @staticmethod
    def __compute_velocity_in_orbital_plane(u: np.ndarray | float, e: float, p: float):
        """
        Compute the velocity of the source in the orbital plane at a given time.
        Now I will compute dot{r} and r*dot{phi} and then project them with
        angle of true anomaly.
        I consider the direction of z axis as the direction of the angular momentum vector.
        If inclination is 0, then the orbital plane is the sky plane,
        and the line of sight is the z axis.

        Parameters
        ----------
        u: `np.ndarray` or `float`
            The eccentric anomaly
        e: `float`
            The eccentricity

        Returns
        -------
        np.ndarray of shape (3, len(u)), the velocity of the source in the orbital plane
        """
        # fmt: off
        v_r = (2 * np.pi / p * e * np.sin(u)) / (1 - e * np.cos(u))  # radial velocity, unit: theta_E/day
        v_psi = (2 * np.pi / p * np.sqrt(1 - e**2) / (1 - e * np.cos(u)))  # azimuthal velocity, unit: theta_E/day
        cos_f = (np.cos(u) - e) / (1 - e * np.cos(u))  # cosine of the true anomaly
        sin_f = (np.sqrt(1 - e**2) * np.sin(u)) / (1 - e * np.cos(u))  # sine of the true anomaly
        v_x = v_r * cos_f - v_psi * sin_f  # x-component of the velocity
        v_y = v_r * sin_f + v_psi * cos_f  # y-component of the velocity
        v_z = 0  # z-component of the velocity
        # fmt: on
        return np.array([v_x, v_y, v_z])  # velocity of the source when t=t0

    @staticmethod
    def __rotate_orbit_campbell(
        pos_vec: np.ndarray, i: float, omega: float, Omega: float
    ):
        """
        Compute the position of the source after rotating by the i, the inclination,
        omega, the argument of periastron, and Omega, the longitude of ascending node,
        which means the attitude of the orbital plane is described in Campbell's notation.

        Parameters
        ----------
        pos_vec: `np.ndarray`, shape (3, len(u)), where u is the eccentric anomaly
            The position of the source, z is the direction of angular momentum.
        i: `float`
            The inclination
        omega: `float`
            The argument of periastron
        Omega: `float`
            The longitude of ascending node
        """
        # Compute the projection of position of the source in the sky plane
        rot_mat_z_Omega = np.array(
            [
                [np.cos(Omega), -np.sin(Omega), 0],
                [np.sin(Omega), np.cos(Omega), 0],
                [0, 0, 1],
            ]
        )
        rot_mat_x_i = np.array(
            [
                [1, 0, 0], 
                [0, np.cos(i), -np.sin(i)], 
                [0, np.sin(i), np.cos(i)]
            ]  # fmt: skip
        )
        rot_mat_z_omega = np.array(
            [
                [np.cos(omega), -np.sin(omega), 0],
                [np.sin(omega), np.cos(omega), 0],
                [0, 0, 1],
            ]
        )
        return np.dot(
            rot_mat_z_Omega, np.dot(rot_mat_x_i, np.dot(rot_mat_z_omega, pos_vec))
        )

    @staticmethod
    def __rotate_orbit_thiele_innes(
        pos_xy_vec: np.ndarray, A: float, B: float, F: float, G: float
    ):
        """
        Compute the position of the source after rotating by the A, B, F, G,
        i.e. the Thiele-Innes parameters.

        Parameters
        ----------
        pos_xy_vec: `np.ndarray`, shape (2, len(u)), where u is the eccentric anomaly
            The position of the source in the orbital plane.
        A: `float`
        B: `float`
        F: `float`
        G: `float`

        Returns
        -------
        np.ndarray of shape (2, len(u)), the position of the source in the sky plane
        """
        Thiele_Innes_mat = np.array([[A, F], [B, G]])
        return np.dot(Thiele_Innes_mat, pos_xy_vec)

    def __get_trajectory_std(self):
        """
        This function compute the standard trajectory of the source in unit of theta_E.
        """
        t_0 = self.parameters["t_0"]
        u_0 = self.parameters["u_0"]
        t_E = self.parameters["t_E"]
        tau, beta = (self.jds - t_0) / t_E, u_0 * np.ones(len(self.jds))
        return tau, beta

    def __get_trajectory_static_2s(self):
        t_0 = self.parameters["t_0"]
        u_0 = self.parameters["u_0"]
        t_E = self.parameters["t_E"]
        tau_1, beta_1 = (self.jds - t_0) / t_E, u_0 * np.ones(len(self.jds))
        t_0_2 = self.parameters["t_0_2"]
        u_0_2 = self.parameters["u_0_2"]
        tau_2, beta_2 = (self.jds - t_0_2) / t_E, u_0_2 * np.ones(len(self.jds))
        return tau_1, beta_1, tau_2, beta_2

    def __get_delta_trajectory_annual_parallax(self):
        pi_E_N, pi_E_E = self.parameters["pi_E_N"], self.parameters["pi_E_E"]
        pi_E_vec = np.array([pi_E_N, pi_E_E])
        delta_tau_prlx = np.dot(pi_E_vec, self.delta_sun.T)
        delta_beta_prlx = np.cross(pi_E_vec, self.delta_sun)
        return delta_tau_prlx, delta_beta_prlx

    def __get_delta_trajectory_satellite_parallax(self):
        pi_E_N, pi_E_E = self.parameters["pi_E_N"], self.parameters["pi_E_E"]
        pi_E_vec = np.array([pi_E_N, pi_E_E])
        tot_qn_qe = self.delta_sun + self.neg_delta_sate
        delta_tau_prlx = np.dot(pi_E_vec, tot_qn_qe.T)
        delta_beta_prlx = np.cross(pi_E_vec, tot_qn_qe)
        return delta_tau_prlx, delta_beta_prlx

    def __get_delta_trajectory_xallarap_circular(self):
        """
        Compute the xallarap trajectory.
        """
        # fmt: off
        p_xi = self.parameters["p_xi"]  # Period of the binary
        phi_xi = self.parameters["phi_xi"]  # Phase of the mean anomaly when t = t_ref
        i = self.parameters["i_xi"]  # Inclination
        xi_E_a, xi_E_b = self.parameters["xi_E_N"], self.parameters["xi_E_E"]
        xi_E_vec = np.array([xi_E_a, xi_E_b])

        # Notice that in circular orbit, the mean anomaly is the eccetirc anomaly 
        # and is the true anomaly.
        l_arr = 2 * np.pi * (self.jds - self.t_ref) / p_xi + phi_xi  # mean anomaly, set the reference time as t0
        l_ref = phi_xi  # mean anomaly when t=t0

        delta_xlrp_arr_x = -(np.cos(l_arr) - np.cos(l_ref)) + (self.jds - self.t_ref) * 2*np.pi/p_xi * (-np.sin(l_ref))
        delta_xlrp_arr_y = -(np.sin(l_arr) - np.sin(l_ref)) + (self.jds - self.t_ref) * 2*np.pi/p_xi * np.cos(l_ref)
        delta_xlrp_arr_z = 0 * l_arr
        xlrp_arr_vec = np.array([delta_xlrp_arr_x, delta_xlrp_arr_y, delta_xlrp_arr_z])

        rot_mat_x_i = np.array(
            [
                [1, 0, 0], 
                [0, np.cos(i), -np.sin(i)], 
                [0, np.sin(i), np.cos(i)]
            ]
        )
        xlrp_arr_full = np.dot(rot_mat_x_i, xlrp_arr_vec)
        xlrp_arr = np.array([xlrp_arr_full[0], xlrp_arr_full[1]]).T
        # The z value, i.e. the position of the source in the direction of the sight, is useless

        # Project the position to the (tau, beta) coordinate
        delta_tau_xlrp_circ = np.dot(xi_E_vec, xlrp_arr.T)
        delta_beta_xlrp_circ = np.cross(xi_E_vec, xlrp_arr)

        # fmt: on
        return delta_tau_xlrp_circ, delta_beta_xlrp_circ

    def __get_delta_trajectory_xallarap_circular_thiele_innes(self):
        """
        Compute the circular xallarap trajectory using the Thiele-Innes elements.
        """
        # fmt: off
        p_xi = self.parameters["p_xi"]
        A_xi = self.parameters["A_xi"]
        B_xi = self.parameters["B_xi"]
        F_xi = self.parameters["F_xi"]
        G_xi = self.parameters["G_xi"]

        l_arr = 2 * np.pi * (self.jds - self.t_ref) / p_xi  # mean anomaly, set the reference time as t0
        # Reason for l_ref is 0: here l_ref should be phi_xi,
        # but phi_xi is already included in the Thiele-Innes elements.
        l_ref = 0  # mean anomaly when t=t0.

        delta_xlrp_arr_a = -(np.cos(l_arr) - np.cos(l_ref)) + (self.jds - self.t_ref) * 2*np.pi/p_xi * (-np.sin(l_ref))
        delta_xlrp_arr_b = -(np.sin(l_arr) - np.sin(l_ref)) + (self.jds - self.t_ref) * 2*np.pi/p_xi * np.cos(l_ref)
        
        Thiele_Innes_mat = np.array(
            [[A_xi, F_xi],
            [B_xi, G_xi]]
        )

        delta_tau_beta_arr = np.dot(Thiele_Innes_mat, np.array([delta_xlrp_arr_a, delta_xlrp_arr_b]))

        # fmt: on
        return delta_tau_beta_arr[0], delta_tau_beta_arr[1]

    def __get_delta_trajectory_xallarap_circular_thiele_innes_2s(self):
        """
        Compute the circular xallarap trajectory using the Thiele-Innes elements.
        """
        # fmt: off
        p_xi = self.parameters["p_xi"]
        A_xi = self.parameters["A_xi"]
        B_xi = self.parameters["B_xi"]
        F_xi = self.parameters["F_xi"]
        G_xi = self.parameters["G_xi"]
        if "eta" in self.parameters:
            eta = self.parameters["eta"]
            if self.obname is None:
                qf_xi = self.parameters["qf_xi"]
            else:
                qf_xi = self.parameters[f"qf_{self.obname}"]
            q_xi = (1+(1+eta)*qf_xi)/(eta-1-qf_xi)
        elif "q_xi" in self.parameters:
            q_xi = self.parameters["q_xi"]

        # Reason for how l is set here please refer to the previous function.
        l1_arr = 2 * np.pi * (self.jds - self.t_ref) / p_xi
        l1_ref = 0
        # Difference 1: is the l1_ref here is 0, while l2_ref should be pi.
        l2_arr = l1_arr.copy() + np.pi
        # Here the velocity is not subtracted, because here we consider the barycenter of the source binary
        delta_xlrp_arr_1_a = -(np.cos(l1_arr) - np.cos(l1_ref)) + (self.jds - self.t_ref) * 2*np.pi/p_xi * (-np.sin(l1_ref))
        delta_xlrp_arr_1_b = -(np.sin(l1_arr) - np.sin(l1_ref)) + (self.jds - self.t_ref) * 2*np.pi/p_xi * np.cos(l1_ref)
        # Difference 2: here the semi-major axis is divided by the mass ratio
        # The secondary source is less massive than the primary source,
        # and q is always smaller than 1, so here should devide it to get a larger
        # semi-major axis.
        delta_xlrp_arr_2_a = -(1/q_xi*np.cos(l2_arr) - np.cos(l1_ref)) + (self.jds - self.t_ref) * 2*np.pi/p_xi * (-np.sin(l1_ref))
        delta_xlrp_arr_2_b = -(1/q_xi*np.sin(l2_arr) - np.sin(l1_ref)) + (self.jds - self.t_ref) * 2*np.pi/p_xi * np.cos(l1_ref)
        Thiele_Innes_mat = np.array(
            [[A_xi, F_xi],
            [B_xi, G_xi]]
        )
        delta_tau_beta_1_arr = np.dot(Thiele_Innes_mat, np.array([delta_xlrp_arr_1_a, delta_xlrp_arr_1_b]))
        delta_tau_beta_2_arr = np.dot(Thiele_Innes_mat, np.array([delta_xlrp_arr_2_a, delta_xlrp_arr_2_b]))

        # fmt: on
        return (
            delta_tau_beta_1_arr[0],
            delta_tau_beta_1_arr[1],
            delta_tau_beta_2_arr[0],
            delta_tau_beta_2_arr[1],
        )

    def __get_delta_trajectory_xallarap_campbell(self):
        """
        Compute the xallarap shift with campbell elements.
        """
        # fmt: off
        e = self.parameters["e_xi"]  # Eccentricity
        p = self.parameters["p_xi"]  # Period of the binary
        phi = self.parameters["phi_xi"]  # Phase of the mean anomaly when t = t_ref
        inclination = self.parameters["i_xi"]  # Inclination
        omega = self.parameters["omega_xi"]  # Argument of periastron
        Omega = self.parameters["Omega_xi"]  # Longitude of ascending node
        xi_E_a, xi_E_b = self.parameters["xi_E_N"], self.parameters["xi_E_E"]
        xi_E_vec = np.array([xi_E_a, xi_E_b])

        l_arr = 2 * np.pi * (self.jds - self.t_ref) / p + phi  # mean anomaly, set the reference time as t0
        l_ref = phi  # mean anomaly when t=t0
        u_arr, u_ref = self.__compute_eccentirc_anomaly(l_arr, e, l_ref)
        # Xallarap trajectory
        xlrp_nonrot_vec = self.__compute_position_in_orbital_plane(u_arr, e)
        xlrp_vec = self.__rotate_orbit_campbell(xlrp_nonrot_vec, inclination, omega, Omega)
        # Xallarap reference position
        xlrp_ref_nonrot_vec = self.__compute_position_in_orbital_plane(u_ref, e)
        xlrp_ref_vec = self.__rotate_orbit_campbell(xlrp_ref_nonrot_vec, inclination, omega, Omega)
        # Xallarap reference velocity
        xlrp_ref_v_nonrot_vec = self.__compute_velocity_in_orbital_plane(u_ref, e, p)
        xlrp_ref_v_vec = self.__rotate_orbit_campbell(xlrp_ref_v_nonrot_vec, inclination, omega, Omega)

        # Compute Delta N, Delta E
        delta_xlrp_arr_a = -(xlrp_vec[0] - xlrp_ref_vec[0]) + (self.jds - self.t_ref) * xlrp_ref_v_vec[0]
        delta_xlrp_arr_b = -(xlrp_vec[1] - xlrp_ref_vec[1]) + (self.jds - self.t_ref) * xlrp_ref_v_vec[1]
        xlrp_arr = np.array([delta_xlrp_arr_a, delta_xlrp_arr_b]).T
        # The z value, i.e. the position of the source in the direction of the sight, is useless

        # Project the position to the (tau, beta) coordinate
        delta_tau_xlrp_cpb = np.dot(xi_E_vec, xlrp_arr.T)
        delta_beta_xlrp_cpb = np.cross(xi_E_vec, xlrp_arr)
        return delta_tau_xlrp_cpb, delta_beta_xlrp_cpb

    def __get_delta_trajectory_xallarap_thiele_innes(self):
        """
        Compute the parallax shift with Thiele-Innes elements.
        """
        # fmt: off
        e = self.parameters["e_xi"]  # Eccentricity
        p = self.parameters["p_xi"]  # Period of the binary
        phi = self.parameters["phi_xi"]  # Phase of the mean anomaly when t = t_ref
        # angle between the trajectory of relative motion and the semi-major axis
        theta_xi = self.parameters["theta_xi"]
        # Thiele-Innes elements
        A = self.parameters["A_xi"]
        B = self.parameters["B_xi"]
        F = self.parameters["F_xi"]
        G = self.parameters["G_xi"]

        l_arr = 2 * np.pi * (self.jds - self.t_ref) / p + phi  # mean anomaly, set the reference time as t0
        l_ref = phi  # mean anomaly when t=t0
        u_arr, u_ref = self.__compute_eccentirc_anomaly(l_arr, e, l_ref)
        # Xallarap trajectory
        xlrp_nonrot_xy_vec = self.__compute_position_in_orbital_plane(u_arr, e)[:2]
        xlrp_xy_vec = self.__rotate_orbit_thiele_innes(xlrp_nonrot_xy_vec, A, B, F, G)
        xlrp_ref_nonrot_xy_vec = self.__compute_position_in_orbital_plane(u_ref, e)[:2]
        xlrp_ref_xy_vec = self.__rotate_orbit_thiele_innes(xlrp_ref_nonrot_xy_vec, A, B, F, G)
        xlrp_ref_v_nonrot_xy_vec = self.__compute_velocity_in_orbital_plane(u_ref, e, p)[:2]
        xlrp_ref_v_xy_vec = self.__rotate_orbit_thiele_innes(xlrp_ref_v_nonrot_xy_vec, A, B, F, G)
        delta_xlrp_arr_a = -(xlrp_xy_vec[0] - xlrp_ref_xy_vec[0]) + (self.jds - self.t_ref) * xlrp_ref_v_xy_vec[0]
        delta_xlrp_arr_b = -(xlrp_xy_vec[1] - xlrp_ref_xy_vec[1]) + (self.jds - self.t_ref) * xlrp_ref_v_xy_vec[1]
        if theta_xi is not None or theta_xi != 0:
            delta_tau_xlrp = delta_xlrp_arr_a * np.cos(theta_xi) + delta_xlrp_arr_b * np.sin(theta_xi)
            delta_beta_xlrp = -delta_xlrp_arr_a * np.sin(theta_xi) + delta_xlrp_arr_b * np.cos(theta_xi)
        else:
            delta_tau_xlrp = delta_xlrp_arr_b
            delta_beta_xlrp = delta_xlrp_arr_a
        return delta_tau_xlrp, delta_beta_xlrp

    def set_times(self, jds: np.ndarray):
        """
        Set times of the light curve and compute the annual-parallax.
        """
        self.jds = jds
        # Only when time is first set or changed the annual parallax is computed
        if "prlx" in self.parameter_set_enabled:
            self.delta_sun = self.__annual_parallax()
            if self.ephemeris is not None:
                # If it is satellite, add the satellite parallax
                self.neg_delta_sate = self.__satellite_parallax()
        return

    def get_trajectory(self):
        """
        Compute the trajectory of the source in unit of theta_E.
        """
        # fmt: off
        if self.jds is None:
            raise ValueError("Please set the times of the light curve first.")

        # First compute standard trajectory
        tau, beta = self.__get_trajectory_std()

        # Compute the parallax shift
        if "prlx" in self.parameter_set_enabled:
            if self.ephemeris is None:
                delta_tau_prlx, delta_beta_prlx = self.__get_delta_trajectory_annual_parallax()
            else:
                delta_tau_prlx, delta_beta_prlx = self.__get_delta_trajectory_satellite_parallax()
        else:
            delta_tau_prlx, delta_beta_prlx = 0, 0
        
        # Compute the xallarap shift of circular orbit
        if "xlrp_circ" in self.parameter_set_enabled:
            delta_tau_xlrp, delta_beta_xlrp = self.__get_delta_trajectory_xallarap_circular()
        elif "xlrp_circ_ti" in self.parameter_set_enabled:
            delta_tau_xlrp, delta_beta_xlrp = self.__get_delta_trajectory_xallarap_circular_thiele_innes()
        elif "xlrp_cpb" in self.parameter_set_enabled:
            delta_tau_xlrp, delta_beta_xlrp = self.__get_delta_trajectory_xallarap_campbell()
        elif "xlrp_ti" in self.parameter_set_enabled:
            delta_tau_xlrp, delta_beta_xlrp = self.__get_delta_trajectory_xallarap_thiele_innes()
        else:
            delta_tau_xlrp, delta_beta_xlrp = 0, 0

        # delta_tau_prlx, delta_beta_prlx are the parallax shift for the lens
        # delta_tau_xlrp, delta_beta_xlrp are the xallarap shift for the source
        # The convention is to compute the shift of the lens, with the source fixed,
        # so here is parallax - xallarap
        tau += delta_tau_prlx + delta_tau_xlrp
        beta += delta_beta_prlx + delta_beta_xlrp

        # fmt: on
        self.trajectory = np.array([tau, beta])
        # TODO: refactor this
        if "xlrp_circ_ti_2s" in self.parameter_set_enabled:
            (
                delta_tau_xlrp_1,
                delta_beta_xlrp_1,
                delta_tau_xlrp_2,
                delta_beta_xlrp_2,
            ) = self.__get_delta_trajectory_xallarap_circular_thiele_innes_2s()
            tau_1 = tau + delta_tau_xlrp_1
            beta_1 = beta + delta_beta_xlrp_1
            tau_2 = tau + delta_tau_xlrp_2
            beta_2 = beta + delta_beta_xlrp_2
            self.trajectory = np.array([tau_1, beta_1, tau_2, beta_2])
        elif "bins" in self.parameter_set_enabled:
            tau_1, beta_1, tau_2, beta_2 = self.__get_trajectory_static_2s()
            tau_1 += delta_tau_prlx
            beta_1 += delta_beta_prlx
            tau_2 += delta_tau_prlx
            beta_2 += delta_beta_prlx
            self.trajectory = np.array([tau_1, beta_1, tau_2, beta_2])
        return self.trajectory

    def get_magnification(self):
        """
        Compute the magnification of the source.
        """
        if (
            "xlrp_circ_ti_2s" in self.parameter_set_enabled
            or "bins" in self.parameter_set_enabled
        ):
            tau1, beta1, tau2, beta2 = self.get_trajectory()
            u1 = np.sqrt(tau1**2 + beta1**2)
            u2 = np.sqrt(tau2**2 + beta2**2)
            magnification1 = (u1**2 + 2) / u1 / np.sqrt(u1**2 + 4)
            magnification2 = (u2**2 + 2) / u2 / np.sqrt(u2**2 + 4)
            if self.obname is None:
                qf_xi = self.parameters["qf_xi"]
            else:
                qf_xi = self.parameters[f"qf_{self.obname}"]
            self.magnification = magnification1 + qf_xi * magnification2
        else:
            tau, beta = self.get_trajectory()
            u = np.sqrt(tau**2 + beta**2)
            # Magnification without finite source effect
            self.magnification = (u**2 + 2) / u / np.sqrt(u**2 + 4)

        # TODO: 2s magnification

        return self.magnification

    def get_magnification_norm(self):
        """
        Get normalized magnification.
        """
        magnification_raw = self.get_magnification()
        if (
            "xlrp_circ_ti_2s" in self.parameter_set_enabled
            or "bins" in self.parameter_set_enabled
        ):
            if self.obname is None:
                qf_xi = self.parameters["qf_xi"]
            else:
                qf_xi = self.parameters[f"qf_{self.obname}"]
            magnification_norm = magnification_raw / (1 + qf_xi)
        else:
            magnification_norm = magnification_raw.copy()
        return magnification_norm

    def get_light_curve(self, fs, fb, return_type="mag", zero_point_mag=18.0):
        """
        Compute the light curve.
        """
        self.get_magnification()
        self.model_flux = fs * self.magnification + fb
        self.model_mag = zero_point_mag - 2.5 * np.log10(self.model_flux)
        if return_type == "flux":
            return self.model_flux
        elif return_type == "mag":
            return self.model_mag
        else:
            raise ValueError(
                'Unknown return type. Only "flux" and "mag" are supported.'
            )

    def get_earth_vel_pos(self, return_type, time_format="jd"):
        t_ref, jds = self.t_0_par, self.jds

        east_projected, north_projected = self.__unit_E_N()
        time_ref = Time(t_ref, format=time_format)
        earth_pos_ref, earth_vel_ref = get_body_barycentric_posvel("earth", time_ref)
        sun_pos_ref = -earth_pos_ref.get_xyz().value
        sun_vel_ref = -earth_vel_ref.get_xyz().value

        time_jds = Time(jds, format=time_format)
        # This is faster when only calculate the position
        earth_pos_arr = get_body_barycentric("earth", time_jds)
        delta_sun_pos_N, delta_sun_pos_E = [], []
        delta_sun_vel_N, delta_sun_vel_E = [], []
        for time, earth_pos in zip(jds, earth_pos_arr):
            sun_pos = -earth_pos.get_xyz().value
            delta_sun_pos = sun_pos - sun_pos_ref
            delta_sun_vel = -(time - t_ref) * sun_vel_ref
            delta_sun_pos_N.append(np.dot(delta_sun_pos, north_projected))
            delta_sun_pos_E.append(np.dot(delta_sun_pos, east_projected))
            delta_sun_vel_N.append(np.dot(delta_sun_vel, north_projected))
            delta_sun_vel_E.append(np.dot(delta_sun_vel, east_projected))
        delta_sun_pos = np.array([delta_sun_pos_N, delta_sun_pos_E]).T
        delta_sun_vel = np.array([delta_sun_vel_N, delta_sun_vel_E]).T

        pi_E_N, pi_E_E = self.parameters["pi_E_N"], self.parameters["pi_E_E"]
        pi_E_vec = np.array([pi_E_N, pi_E_E])

        if return_type == "pos":
            delta_sun = delta_sun_pos
        elif return_type == "vel":
            delta_sun = delta_sun_vel
        else:
            raise ValueError("return_type should be 'pos' or 'vel'.")

        delta_tau_prlx = np.dot(pi_E_vec, delta_sun.T)
        delta_beta_prlx = np.cross(pi_E_vec, delta_sun)
        return delta_tau_prlx, delta_beta_prlx

    def get_satellite_pos(self):
        pi_E_N, pi_E_E = self.parameters["pi_E_N"], self.parameters["pi_E_E"]
        pi_E_vec = np.array([pi_E_N, pi_E_E])
        delta_tau_sun_pos, delta_beta_sun_pos = self.get_earth_vel_pos("pos")
        delta_tau_sate = np.dot(pi_E_vec, self.neg_delta_sate.T)
        delta_beta_sate = np.cross(pi_E_vec, self.neg_delta_sate)
        return delta_tau_sun_pos + delta_tau_sate, delta_beta_sun_pos + delta_beta_sate

    def get_source_vel_pos(self, return_type):
        # fmt: off
        p_xi = self.parameters["p_xi"]
        A_xi = self.parameters["A_xi"]
        B_xi = self.parameters["B_xi"]
        F_xi = self.parameters["F_xi"]
        G_xi = self.parameters["G_xi"]

        l_arr = 2 * np.pi * (self.jds - self.t_ref) / p_xi  # mean anomaly, set the reference time as t0
        # Reason for l_ref is 0: here l_ref should be phi_xi,
        # but phi_xi is already included in the Thiele-Innes elements.
        l_ref = 0  # mean anomaly when t=t0.

        delta_xlrp_pos_a = -(np.cos(l_arr) - np.cos(l_ref))
        delta_xlrp_pos_b = -(np.sin(l_arr) - np.sin(l_ref))
        delta_xlrp_vel_a = (self.jds - self.t_ref) * 2*np.pi/p_xi * (-np.sin(l_ref))
        delta_xlrp_vel_b = (self.jds - self.t_ref) * 2*np.pi/p_xi * np.cos(l_ref)
        
        Thiele_Innes_mat = np.array(
            [[A_xi, F_xi],
            [B_xi, G_xi]]
        )

        if return_type == "pos":
            delta_tau_beta_pos = np.dot(Thiele_Innes_mat, np.array([delta_xlrp_pos_a, delta_xlrp_pos_b]))
            return delta_tau_beta_pos[0], delta_tau_beta_pos[1]
        elif return_type == "vel":
            delta_tau_beta_vel = np.dot(Thiele_Innes_mat, np.array([delta_xlrp_vel_a, delta_xlrp_vel_b]))
            return delta_tau_beta_vel[0], delta_tau_beta_vel[1]

    def get_two_source_vel_pos(self, return_type):
        # fmt: off
        p_xi = self.parameters["p_xi"]
        A_xi = self.parameters["A_xi"]
        B_xi = self.parameters["B_xi"]
        F_xi = self.parameters["F_xi"]
        G_xi = self.parameters["G_xi"]
        #if "eta" in self.parameters:
        #    eta = self.parameters["eta"]
        #    if self.obname is None:
        #        qf_xi = self.parameters["qf_xi"]
        #    else:
        #        qf_xi = self.parameters[f"qf_{self.obname}"]
        #    q_xi = (1+(1+eta)*qf_xi)/(eta-1-qf_xi)
        #elif "q_xi" in self.parameters:
        q_xi = self.parameters["q_xi"]

        # Reason for how l is set here please refer to the previous function.
        l1_arr = 2 * np.pi * (self.jds - self.t_ref) / p_xi
        l1_ref = 0
        # Difference 1: is the l1_ref here is 0, while l2_ref should be pi.
        l2_arr = l1_arr.copy() + np.pi
        # Here the velocity is not subtracted, because here we consider the barycenter of the source binary
        delta_xlrp_pos_1_a = -(np.cos(l1_arr) - np.cos(l1_ref)) 
        delta_xlrp_pos_1_b = -(np.sin(l1_arr) - np.sin(l1_ref)) 
        delta_xlrp_vel_1_a = (self.jds - self.t_ref) * 2*np.pi/p_xi * (-np.sin(l1_ref))
        delta_xlrp_vel_1_b = (self.jds - self.t_ref) * 2*np.pi/p_xi * np.cos(l1_ref)
        # Difference 2: here the semi-major axis is divided by the mass ratio
        # The secondary source is less massive than the primary source,
        # and q is always smaller than 1, so here should devide it to get a larger
        # semi-major axis.
        delta_xlrp_arr_2_a = -(1/q_xi*np.cos(l2_arr) - np.cos(l1_ref))
        delta_xlrp_arr_2_b = -(1/q_xi*np.sin(l2_arr) - np.sin(l1_ref))
        Thiele_Innes_mat = np.array(
            [[A_xi, F_xi],
            [B_xi, G_xi]]
        )
        if return_type == "pos":
            delta_tau_beta_1_pos = np.dot(Thiele_Innes_mat, np.array([delta_xlrp_pos_1_a, delta_xlrp_pos_1_b]))
            delta_tau_beta_2_pos = np.dot(Thiele_Innes_mat, np.array([delta_xlrp_arr_2_a, delta_xlrp_arr_2_b]))
            return delta_tau_beta_1_pos[0], delta_tau_beta_1_pos[1], delta_tau_beta_2_pos[0], delta_tau_beta_2_pos[1]
        elif return_type == "vel":
            # Only the vel of the primary source is considered here.
            delta_tau_beta_vel = np.dot(Thiele_Innes_mat, np.array([delta_xlrp_vel_1_a, delta_xlrp_vel_1_b]))
            return delta_tau_beta_vel[0], delta_tau_beta_vel[1]

    def get_std_traj(self):
        return self.__get_trajectory_std()

    def test_xlrp_2s_pos_vel(self):
        s1_test_tau, s1_test_beta, s2_test_tau, s2_test_beta = self.get_trajectory()
        std_tau, std_beta = self.get_std_traj()
        delta_tau_prlx, delta_beta_prlx = self.__get_delta_trajectory_annual_parallax()
        (
            delta_tau_xlrp_1,
            delta_beta_xlrp_1,
            delta_tau_xlrp_2,
            delta_beta_xlrp_2,
        ) = self.__get_delta_trajectory_xallarap_circular_thiele_innes_2s()
        s1_total_tau = std_tau + delta_tau_prlx - delta_tau_xlrp_1
        s1_total_beta = std_beta + delta_beta_prlx - delta_beta_xlrp_1
        s2_total_tau = std_tau + delta_tau_prlx - delta_tau_xlrp_2
        s2_total_beta = std_beta + delta_beta_prlx - delta_beta_xlrp_2
        np.testing.assert_almost_equal(s1_total_tau, s1_test_tau, decimal=10)
        np.testing.assert_almost_equal(s1_total_beta, s1_test_beta, decimal=10)
        np.testing.assert_almost_equal(s2_total_tau, s2_test_tau, decimal=10)
        np.testing.assert_almost_equal(s2_total_beta, s2_test_beta, decimal=10)

        delta_tau_pos_prlx, delta_beta_pos_prlx = self.get_earth_vel_pos(
            return_type="pos"
        )
        delta_tau_vel_prlx, delta_beta_vel_prlx = self.get_earth_vel_pos(
            return_type="vel"
        )
        (
            delta_tau_pos_xlrp_1,
            delta_beta_pos_xlrp_1,
            delta_tau_pos_xlrp_2,
            delta_beta_pos_xlrp_2,
        ) = self.get_two_source_vel_pos(return_type="pos")
        delta_tau_vel_xlrp_1, delta_beta_vel_xlrp_1 = self.get_two_source_vel_pos(
            return_type="vel"
        )
        delta_tau_total_prlx = delta_tau_pos_prlx + delta_tau_vel_prlx
        delta_beta_total_prlx = delta_beta_pos_prlx + delta_beta_vel_prlx
        np.testing.assert_almost_equal(delta_tau_total_prlx, delta_tau_prlx, decimal=5)
        np.testing.assert_almost_equal(
            delta_beta_total_prlx, delta_beta_prlx, decimal=5
        )
        delta_tau_total_xlrp_1 = delta_tau_pos_xlrp_1 + delta_tau_vel_xlrp_1
        delta_beta_total_xlrp_1 = delta_beta_pos_xlrp_1 + delta_beta_vel_xlrp_1
        np.testing.assert_almost_equal(
            delta_tau_total_xlrp_1, delta_tau_xlrp_1, decimal=5
        )
        np.testing.assert_almost_equal(
            delta_beta_total_xlrp_1, delta_beta_xlrp_1, decimal=5
        )
        delta_tau_total_xlrp_2 = delta_tau_pos_xlrp_2 + delta_tau_vel_xlrp_1
        delta_beta_total_xlrp_2 = delta_beta_pos_xlrp_2 + delta_beta_vel_xlrp_1
        np.testing.assert_almost_equal(
            delta_tau_total_xlrp_2, delta_tau_xlrp_2, decimal=5
        )
        np.testing.assert_almost_equal(
            delta_beta_total_xlrp_2, delta_beta_xlrp_2, decimal=5
        )
        s1_total_tau = std_tau + delta_tau_total_prlx - delta_tau_total_xlrp_1
        s1_total_beta = std_beta + delta_beta_total_prlx - delta_beta_total_xlrp_1
        s2_total_tau = std_tau + delta_tau_total_prlx - delta_tau_total_xlrp_2
        s2_total_beta = std_beta + delta_beta_total_prlx - delta_beta_total_xlrp_2
        np.testing.assert_almost_equal(s1_total_tau, s1_test_tau, decimal=5)
        np.testing.assert_almost_equal(s1_total_beta, s1_test_beta, decimal=5)
        np.testing.assert_almost_equal(s2_total_tau, s2_test_tau, decimal=5)
        np.testing.assert_almost_equal(s2_total_beta, s2_test_beta, decimal=5)
        return
