"""Implement the FW-H equation solver using Farassat's Formula 1A."""

import logging
import math

import numpy as np
from numpy.typing import NDArray

from fw_h.config import ConfigSchema
from fw_h.source import SourceData

logger = logging.getLogger(__name__)


class Solver:
    """Contains solution routines and data for the given source data."""

    def __init__(self, config: ConfigSchema, source: SourceData) -> None:
        logger.info("Initializing Solver object...")
        self.config = config
        self.source = source
        self.r_x = np.ndarray(0, dtype=np.float64)
        self.r_y = np.ndarray(0, dtype=np.float64)
        self.r_z = np.ndarray(0, dtype=np.float64)
        self.r = np.ndarray(0, dtype=np.float64)
        self.r_hat_x = np.ndarray(0, dtype=np.float64)
        self.r_hat_y = np.ndarray(0, dtype=np.float64)
        self.r_hat_z = np.ndarray(0, dtype=np.float64)
        self.time_domain = np.ndarray(0, dtype=np.float64)
        self.v_n = np.ndarray(np.ndarray(0, dtype=np.float64))
        self.v_n_dot = np.ndarray(np.ndarray(0, dtype=np.float64))
        self.p_dot = np.ndarray(np.ndarray(0, dtype=np.float64))
        self.cos_theta = np.ndarray(0, dtype=np.float64)
        self.p = np.ndarray(np.ndarray(0, dtype=np.float64))

    def _compute_r(self) -> None:
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

    def _discretize_observer_time(self) -> None:
        """Discretize the observer time."""
        logger.info("Discretizing the observer time...")
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

    def _compute_v_n(self) -> None:
        """Compute the normal velocity for each FW-H point."""
        logger.info(
            "Computing the normal velocity for each point on the FW-H "
            "surface over each source time step..."
        )
        self.v_n = (
            self.source.fw_h_velocity_x * self.source.fw_h_surface.normals.n_x
            + self.source.fw_h_velocity_y
            * self.source.fw_h_surface.normals.n_y
            + self.source.fw_h_velocity_z
            * self.source.fw_h_surface.normals.n_z
        )

    def _compute_v_n_dot(self) -> None:
        """Compute the source time derivative of the normal velocity."""
        logger.info(
            "Computing the source time derivative of the normal "
            "velocity for each point on the FW-H surface over the "
            "source time domain..."
        )
        self.v_n_dot = np.gradient(self.v_n, self.source.time_domain, axis=0)

    def _compute_p_dot(self) -> None:
        """Compute the source time derivative of the pressure."""
        logger.info(
            "Computing the source time derivative of the pressure for "
            "each point on the FW-H surface over the source time "
            "domain..."
        )
        self.p_dot = np.gradient(
            self.source.fw_h_pressure, self.source.time_domain, axis=0
        )

    def _compute_cos_theta(self) -> None:
        """Compute cos(θ) for each point on the FW-H surface."""
        logger.info(
            "Computing the cos(θ) for each point on the FW-H surface..."
        )
        self.cos_theta = (
            self.r_hat_x * self.source.fw_h_surface.normals.n_x
            + self.r_hat_y * self.source.fw_h_surface.normals.n_y
            + self.r_hat_z * self.source.fw_h_surface.normals.n_z
        )

    def _convert_source_to_observer_pressure(
        self,
        p_tau: NDArray[NDArray[np.float64]],
        retarded_time: NDArray[NDArray[np.float64]],
    ) -> None:
        """Convert pressure from source time to observer time.

        Converts the total pressure contribution in source time to
        observer time by using the retarded time transfer matrix.

        Prints progress updates periodically based on how many time
        steps are being processed.

        Parameters
        ----------
        p_tau
            The accumulated pressure contributions at each position on
            the FW-H surface in source time.
        retarded_time
            Each element of this array corresponds to the "transfer
            time" for each element in the p_tau matrix.
        """
        logger.debug("Accumulating pressures at observer...")
        source_steps = len(self.source.time_domain)
        digits = math.floor(math.log10(source_steps))
        progress_interval = 10**digits // 10
        width = digits + 1
        fmt = f"%0{width}d/%0{width}d"

        for i in range(len(self.source.time_domain)):
            if progress_interval > 0 and (i + 1) % progress_interval == 0:
                logger.info(
                    "Calculating pressure for time step %s...",
                    fmt % (i + 1, source_steps),
                )

            indices = np.argmin(
                np.abs(self.time_domain[:, np.newaxis] - retarded_time[i, :]),
                axis=0,
            )
            np.add.at(self.p[:, 0], indices, p_tau[i].flatten())

    def compute(self) -> None:
        """Compute the pressure at the observer surface over time."""
        self._compute_r()
        self._discretize_observer_time()
        self._compute_v_n()
        self._compute_v_n_dot()
        self._compute_p_dot()
        self._compute_cos_theta()

        logger.info("Computing pressure from Formulation 1A in source time...")
        fw_h_surface_r = np.max(self.source.fw_h_surface.x) - np.min(
            self.source.fw_h_surface.x
        )
        fw_h_surface_n = np.sqrt(len(self.source.fw_h_surface.x) / 6)
        cell_area = (2 * fw_h_surface_r / fw_h_surface_n) ** 2

        logger.debug("Calculating the thickness term...")
        p_t_tau = (
            cell_area
            / (4 * np.pi)
            * (self.config.solver.constants.rho_0 * self.v_n_dot / self.r)
        )

        logger.debug("Calculating the loading term...")
        p_l_tau = (
            cell_area
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

        self._convert_source_to_observer_pressure(p_tau, retarded_time)
        logger.info("Finished converting pressure to observer time.")
