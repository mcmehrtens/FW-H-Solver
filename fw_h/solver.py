"""Implement the FW-H equation solver using Farassat's Formula 1A."""

import logging

import numpy as np

from fw_h.config import ConfigSchema
from fw_h.source import SourceData

logger = logging.getLogger(__name__)


class Solver:
    """Calculate the pressure at an observer using FW-H surface data."""

    def __init__(self, config: ConfigSchema, source: SourceData):
        logger.info("Initializing Solver object...")
        self.config = config
        self.source = source
        self.r_x = np.ndarray(0)
        self.r_y = np.ndarray(0)
        self.r_z = np.ndarray(0)
        self.r = np.ndarray(0)
        self.r_hat_x = np.ndarray(0)
        self.r_hat_y = np.ndarray(0)
        self.r_hat_z = np.ndarray(0)
        self.time_domain = np.ndarray(0)
        self.v_n = np.ndarray(0)
        self.v_n_dot = np.ndarray(0)
        self.p_dot = np.ndarray(0)
        self.cos_theta = np.ndarray(0)
        self.p = np.zeros(0)

    def _compute_r(self):
        """Calculate distance from each FW-H point to the observer."""
        logger.info(
            "Computing the distance from each point on the FW-H "
            "surface to the observer (r)..."
        )
        self.r_x = self.source.observer_surface.x - self.source.fw_h_surface.x
        self.r_y = self.source.observer_surface.y - self.source.fw_h_surface.y
        self.r_z = self.source.observer_surface.z - self.source.fw_h_surface.z
        self.r = np.sqrt(self.r_x**2 + self.r_y**2 + self.r_z**2)
        self.r_hat_x = self.r_x / self.r
        self.r_hat_y = self.r_y / self.r
        self.r_hat_z = self.r_z / self.r

    def _discretize_observer_time(self):
        """Discretize the observer time."""
        logger.info("Discretizing the observer time...")
        if len(self.r) == 0:
            raise RuntimeError(
                "Cannot discretize time until r is calculated. Ensure"
                "compute_r is run before discretize_observer_time()."
            )

        t_start = (
            np.min(self.source.time_domain)
            + np.min(self.r) / self.config.solver.constants.c_0
        )
        t_end = (
            np.max(self.source.time_domain)
            + np.max(self.r) / self.config.solver.constants.c_0
        )
        self.time_domain = np.linspace(
            t_start, t_end, self.config.solver.time_steps
        )
        logger.debug("Discretized Observer Time Outputs:")
        logger.debug("Time Steps: %d", self.config.solver.time_steps)
        logger.debug("t_initial: %f", self.time_domain[0])
        logger.debug("t_final: %f", self.time_domain[-1])

    def _compute_v_n(self):
        """Compute the normal velocity for each FW-H point."""
        logger.info(
            "Computing the normal velocity for each point on the FW-H "
            "surface over each source time step..."
        )
        self.v_n = (
            self.source.fw_h_velocity_x * self.source.fw_h_surface.n_x
            + self.source.fw_h_velocity_y * self.source.fw_h_surface.n_y
            + self.source.fw_h_velocity_z * self.source.fw_h_surface.n_z
        )

    def _compute_v_n_dot(self):
        """Compute the source time derivative of the normal velocity."""
        if len(self.v_n) == 0:
            raise RuntimeError(
                "Cannot compute source time derivative without v_n. Ensure "
                "_compute_v_n() is run before _compute_v_n_dot()."
            )
        logger.info(
            "Computing the source time derivative of the normal "
            "velocity for each point on the FW-H surface over the "
            "source time domain..."
        )
        self.v_n_dot = np.gradient(self.v_n, self.source.time_domain, axis=0)

    def _compute_p_dot(self):
        """Compute the source time derivative of the pressure."""
        logger.info(
            "Computing the source time derivative of the pressure for "
            "each point on the FW-H surface over the source time "
            "domain..."
        )
        self.p_dot = np.gradient(
            self.source.fw_h_pressure, self.source.time_domain, axis=0
        )

    def _compute_cos_theta(self):
        """Compute cos(θ) for each point on the FW-H surface."""
        logger.info(
            "Computing the cos(θ) for each point on the FW-H surface..."
        )
        self.cos_theta = (
            self.r_hat_x * self.source.fw_h_surface.n_x
            + self.r_hat_y * self.source.fw_h_surface.n_y
            + self.r_hat_z * self.source.fw_h_surface.n_z
        )

    def compute(self):
        """Compute the pressure at the observer surface over time."""
        self._compute_r()
        self._discretize_observer_time()
        self._compute_v_n()
        self._compute_v_n_dot()
        self._compute_p_dot()
        self._compute_cos_theta()

        # TODO: this edge length is being pulled from the source stuff.
        logger.info("Computing pressure from Formulation 1A in source time...")
        dA = (2 * self.config.fw_h_surface.r / self.config.fw_h_surface.n) ** 2
        p_t_tau = (
            dA
            / (4 * np.pi)
            * (self.config.solver.constants.rho_0 * self.v_n_dot / self.r)
        )
        p_l_tau = (
            dA
            / (4 * np.pi)
            * (
                (
                    self.p_dot
                    * self.cos_theta
                    / (self.config.solver.constants.c_0 * self.r)
                )
                + (self.source.fw_h_pressure * self.cos_theta / self.r**2)
            )
        )

        p_tau = p_t_tau + p_l_tau

        logger.info("Converting pressure from source time to observer time...")
        logger.debug("Preallocating pressure array...")
        self.p = np.zeros((len(self.time_domain), 1))

        logger.debug("Calculating the retarded times...")
        retarded_time = (
            self.source.time_domain[:, np.newaxis]
            + self.r[np.newaxis, :] / self.config.solver.constants.c_0
        )

        logger.debug("Accumulating pressures at observer...")
        for i in range(len(self.source.time_domain)):
            indices = np.argmin(
                np.abs(self.time_domain[:, np.newaxis] - retarded_time[i, :]),
                axis=0,
            )
            np.add.at(self.p[:, 0], indices, p_tau[i].flatten())
