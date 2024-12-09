"""Generate the data resulting from a theoretical source."""
from typing import (
    Callable,
    Tuple,
)

import numpy as np
from numpy import linalg as la
from numpy.typing import NDArray

from fw_h.config import (
    ConfigSchema,
    parse_shape_function,
    SourceType,
)
from fw_h.geometry import (
    Surface,
    generate_fw_h_surface,
)


class SourceData:
    """Generates source data given a source description and surfaces.

    Parameters
    ----------
    config
        Configuration object

    Attributes
    ----------
    fw_h_surface
        Penetrable FW-H surface encapsulating the source
    observer_surface
        Observer points to calculate the theoretical solution for
    time_domain
        Time steps
    source_type
        Analytical description of the source
    source_shape_function
        Shape of pressure perturbations caused by the source
    """

    def __init__(self,
                 config: ConfigSchema):
        self.config = config
        self.fw_h_surface = generate_fw_h_surface(config.fw_h_surface.r,
                                                  config.fw_h_surface.n)
        self.observer_surface = (
            Surface(np.array([config.observer.centroid.x],
                             dtype=np.float64),
                    np.array([config.observer.centroid.y],
                             dtype=np.float64),
                    np.array([config.observer.centroid.z],
                             dtype=np.float64)
                    )
        )
        self.time_domain = np.linspace(config.source.time_domain.start_time,
                                       config.source.time_domain.end_time,
                                       config.source.time_domain.n,
                                       dtype=np.float64)
        self.source_type = config.source.description
        self.source_shape_function = parse_shape_function(config.source.shape)

        self.fw_h_surface_velocity_potential = (
            self.generate_source_data(self.fw_h_surface)
        )
        self.observer_surface_velocity_potential = (
            self.generate_source_data(self.observer_surface)
        )

    def generate_source_data(self,
                             surface: Surface) -> NDArray[NDArray[np.float64]]:
        """Generate source data over a surface.

        Parameters
        ----------
        surface
            Points to calculate velocity potential for

        Returns
        -------
        NDArray[NDArray[np.float64]]
            Matrix of velocity potentials. Each row corresponds to a
            time step. Each column corresponds to the corresponding
            coordinate in the surface object.
        """
        return calculate_velocity_potential(
            surface,
            self.time_domain,
            (
                self.config.source.centroid.x,
                self.config.source.centroid.y,
                self.config.source.centroid.z
            ),
            self.source_type,
            self.source_shape_function,
            self.config.source.amplitude,
            self.config.source.frequency,
            self.config.source.constants.c_0
        )


def calculate_velocity_potential(surface: Surface,
                                 time_domain: NDArray[np.float64],
                                 source_location: Tuple[float, float, float],
                                 source_type: SourceType,
                                 source_shape_function: Callable[
                                     [np.ndarray[np.float64]],
                                     np.ndarray[np.float64]
                                 ],
                                 A: float,
                                 omega: float,
                                 c_0: float) -> NDArray[NDArray[np.float64]]:
    """Calculate velocity potential over a surface.

    Parameters
    ----------
    surface
        Points to calculate the velocity potential over
    time_domain
        Time steps
    source_location
        Source location in cartesian coordinates
    source_type
        Type of analytical sound source
    source_shape_function
        Shape of the source pressure perturbations
    A
        Amplitude of the source pressure perturbations
    omega
        Frequency of source pressure perturbations
    c_0
        Speed of sound

    Returns
    -------
    NDArray[NDArray[np.float64]]
        Matrix of velocity potentials. Each row corresponds to a time
        step. Each column corresponds to the corresponding coordinate in
        the surface object.

    Raises
    ------
    ValueError
        If source_type is invalid
    """
    phi = np.empty(0)
    match source_type:
        case SourceType.MONOPOLE:
            pass
            r = (la.norm(np.stack((surface.x, surface.y, surface.z),
                                  axis=1) - source_location,
                         axis=1))
            phi = (
                    -A
                    * source_shape_function(omega * time_domain[:, np.newaxis]
                                            - r / c_0)
                    / (4 * np.pi * r)
            )
        case SourceType.DIPOLE:
            pass
        case SourceType.QUADRUPOLE:
            pass
        case _:
            raise ValueError(f"Invalid source type: {source_type}")
    return phi
