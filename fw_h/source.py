"""Generate the data resulting from a theoretical source."""
from enum import Enum
from typing import (
    Callable,
)

import numpy as np
from numpy.typing import NDArray

from fw_h.config import ConfigSchema
from fw_h.geometry import Surface


class SourceType(Enum):
    MONOPOLE = 1
    DIPOLE = 2
    QUADRUPOLE = 3


class SourceData:
    def __init__(self,
                 config: ConfigSchema,
                 fwh_surface: Surface,
                 observer_surface: Surface,
                 time_domain: NDArray[int],
                 source_type: SourceType,
                 source_shape_function: Callable[
                     [np.ndarray[np.float64]],
                     np.ndarray[np.float64]
                 ]):
        if source_type not in {item.value for item in SourceType}:
            raise ValueError(f"Invalid source type: {source_type}")

        self.config = config
        self.fwh_surface = fwh_surface
        self.observer_surface = observer_surface
        self.time_domain = time_domain
        self.source_type = source_type
        self.source_shape_function = source_shape_function

        self.fwh_surface_pressure = np.empty((0, 0))
        self.fwh_surface_velocity_x1 = np.empty((0, 0))
        self.fwh_surface_velocity_x2 = np.empty((0, 0))
        self.fwh_surface_velocity_x3 = np.empty((0, 0))

        self.observer_surface_pressure = np.empty((0, 0))
        self.observer_surface_velocity_x1 = np.empty((0, 0))
        self.observer_surface_velocity_x2 = np.empty((0, 0))
        self.observer_surface_velocity_x3 = np.empty((0, 0))

        self.generate_source_data()

    def generate_source_data(self):
        """Generate source data for a theoretical sound source.

        Calculates pressure, velocity, and normal vector for a monopole,
        dipole, or quadrupole source.
        """
        self.fwh_surface_pressure = calculate_pressure(
            self.fwh_surface,
            self.time_domain,
            self.source_type,
            self.source_shape_function,
            self.config.source.constants.c_0,
            self.config.source.constants.p_0,
            self.config.source.constants.rho_0
        )
        # TODO: calculate FW-H surface velocity data

        self.observer_surface_pressure = calculate_pressure(
            self.observer_surface,
            self.time_domain,
            self.source_type,
            self.source_shape_function,
            self.config.source.constants.c_0,
            self.config.source.constants.p_0,
            self.config.source.constants.rho_0
        )
        # TODO: calculate observer surface velocity data


def calculate_pressure(surface: Surface,
                       time_domain: NDArray[int],
                       source_type: SourceType,
                       source_shape_function: Callable[
                           [np.ndarray[np.float64]],
                           np.ndarray[np.float64]
                       ],
                       c_0: float,
                       p_0: float,
                       rho_0: float) -> NDArray[NDArray[np.float64]]:
    p = np.zeros((len(time_domain), len(surface.x)), dtype=np.float64)
    match source_type:
        case SourceType.MONOPOLE:
            pass
            # r = np.linalg.norm(X - X_0, axis=1)
            # r = np.where(r == 0, np.nan, r)
            # p = -rho_0 * A * np.cos(t - r / c_0) / (4 * np.pi * r) + p_0
            # p = np.nan_to_num(p, nan=p_0)
        case SourceType.DIPOLE:
            pass
        case SourceType.QUADRUPOLE:
            pass
    return p
