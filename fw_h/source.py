"""Generate the data resulting from a theoretical source."""
from typing import (
    Tuple,
    Callable,
)

import numpy as np
import sympy as sp
from numpy.typing import NDArray
from sympy.utilities.lambdify import lambdify

from fw_h.config import (
    ConfigSchema,
    parse_shape_function,
    SourceType,
)
from fw_h.geometry import (
    Surface,
    generate_fw_h_surface,
)


def evaluate_monopole_source_functions(
        source_shape_fn: sp.FunctionClass
) -> Tuple[Callable, Callable, Callable, Callable, Callable]:
    """Symbolically evaluate monopole source functions.

    Calculates the velocity potential field function, pressure field
    function, and the velocity field function. These functions are
    evaluated symbolically and then lambdified to use for NumPy.

    Parameters
    ----------
    source_shape_fn
        SymPy function to use for the shape of the source perturbations

    Returns
    -------
    Callable
        Velocity potential field function
    Callable
        Pressure field function
    Callable
        Velocity field (x-component) function
    Callable
        Velocity field (y-component) function
    Callable
        Velocity field (z-component) function
    """
    x, y, z, x_0, y_0, z_0, t, A, omega, c_0, rho_0 = sp.symbols(
        "x y z x_0 y_0 z_0 t A omega c_0 rho_0"
    )

    r = sp.sqrt((x - x_0) ** 2 + (y - y_0) ** 2 + (z - z_0) ** 2)
    phi = -A * source_shape_fn(omega * t - r / c_0) / (4 * sp.pi * r)
    p = -rho_0 * sp.diff(phi, t)
    V_x = sp.diff(phi, x)
    V_y = sp.diff(phi, y)
    V_z = sp.diff(phi, z)

    phi_fn = lambdify(
        (x, y, z, x_0, y_0, z_0, t, A, omega, c_0),
        phi,
        modules="numpy"
    )
    p_fn = lambdify(
        (x, y, z, x_0, y_0, z_0, t, A, omega, c_0, rho_0),
        p,
        modules="numpy"
    )
    V_x_fn = lambdify(
        (x, y, z, x_0, y_0, z_0, t, A, omega, c_0),
        V_x,
        modules="numpy"
    )
    V_y_fn = lambdify(
        (x, y, z, x_0, y_0, z_0, t, A, omega, c_0),
        V_y,
        modules="numpy"
    )
    V_z_fn = lambdify(
        (x, y, z, x_0, y_0, z_0, t, A, omega, c_0),
        V_z,
        modules="numpy"
    )
    return phi_fn, p_fn, V_x_fn, V_y_fn, V_z_fn


class SourceData:
    """Generates all the data for an analytical acoustic source.

    Parameters
    ----------
    config
        Configuration object

    Attributes
    ----------
    config
    fw_h_surface
        Penetrable FW-H surface encapsulating the source
    observer_surface
        Observer points to calculate the theoretical solution for
    time_domain
        Time steps
    source_shape_function
        Shape of pressure perturbations caused by the source
    fw_h_velocity_potential
        Velocity potential on the FW-H surface calculated over time.
        Each row corresponds to a time step. Each column corresponds to
        the same index on the surface.
    observer_velocity_potential
        Same as fw_h_velocity_potential, but for the observer surface
    fw_h_pressure
        Pressure on the FW-H surface calculated over time. Each row
        corresponds to a time step. Each column corresponds to the same
        index on the surface.
    observer_pressure
        Same as fw_h_pressure, but for the observer surface
    fw_h_velocity_x
        x-component of velocity on the FW-H surface calculated over
        time. Each row corresponds to a time step. Each column
        corresponds to the same index on the surface.
    fw_h_velocity_y
        See fw_h_velocity_x
    fw_h_velocity_z
        See fw_h_velocity_x
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
        self.source_shape_function = parse_shape_function(config.source.shape)

        (
            self._velocity_potential_fn,
            self._pressure_fn,
            self._velocity_x_fn,
            self._velocity_y_fn,
            self._velocity_z_fn
        ) = self.generate_source_functions()

        self.fw_h_velocity_potential = self.calculate_velocity_potential(True)
        self.observer_velocity_potential = (
            self.calculate_velocity_potential(False)
        )

        self.fw_h_pressure = self.calculate_pressure(True)
        self.observer_pressure = self.calculate_pressure(False)

        (
            self.fw_h_velocity_x,
            self.fw_h_velocity_y,
            self.fw_h_velocity_z
        ) = self.calculate_velocity()

    def generate_source_functions(
            self
    ) -> Tuple[Callable, Callable, Callable, Callable, Callable]:
        """Call the respective source generation function generator.

        Based on whether the type of the analytical source, call a
        function that will generate analytical, lambdified functions for
        velocity potential, pressure, and velocity. These functions are
        used to calculate the numerical values on the source surfaces.

        Returns
        -------
        Callable
            Velocity potential field function
        Callable
            Pressure field function
        Callable
            Velocity field (x-component) function
        Callable
            Velocity field (y-component) function
        Callable
            Velocity field (z-component) function
        """
        match self.config.source.description:
            case SourceType.MONOPOLE:
                functions = evaluate_monopole_source_functions(
                    self.source_shape_function
                )
            case _:
                raise ValueError("Unknown source type")
        return functions

    def calculate_velocity_potential(
            self,
            fw_h_surface: bool
    ) -> NDArray[NDArray[np.float64]]:
        """Calculate the velocity potential over a surface.

        Parameters
        ----------
        fw_h_surface
            Uses the FW-H surface if true, otherwise uses the observer
            surface

        Returns
        -------
        NDArray[NDArray[np.float64]]
            Matrix of velocity potentials. Each row is a time step. Each
            column corresponds to the respective point on the surface.
        """
        surface = self.fw_h_surface if fw_h_surface else self.observer_surface
        return self._velocity_potential_fn(
            surface.x[:, np.newaxis],
            surface.y[:, np.newaxis],
            surface.z[:, np.newaxis],
            self.config.source.centroid.x,
            self.config.source.centroid.y,
            self.config.source.centroid.z,
            self.time_domain[np.newaxis, :],
            self.config.source.amplitude,
            self.config.source.frequency,
            self.config.source.constants.c_0
        ).T

    def calculate_pressure(
            self,
            fw_h_surface: bool
    ) -> NDArray[NDArray[np.float64]]:
        """Calculate the pressure over a surface.

        Parameters
        ----------
        fw_h_surface
            Uses the FW-H surface if true, otherwise uses the observer
            surface

        Returns
        -------
        NDArray[NDArray[np.float64]]
            Matrix of surface pressure. Each row is a time step. Each
            column corresponds to the respective point on the surface.
        """
        surface = self.fw_h_surface if fw_h_surface else self.observer_surface
        return self._pressure_fn(
            surface.x[:, np.newaxis],
            surface.y[:, np.newaxis],
            surface.z[:, np.newaxis],
            self.config.source.centroid.x,
            self.config.source.centroid.y,
            self.config.source.centroid.z,
            self.time_domain[np.newaxis, :],
            self.config.source.amplitude,
            self.config.source.frequency,
            self.config.source.constants.c_0,
            self.config.source.constants.rho_0
        ).T

    def calculate_velocity(self) -> Tuple[
        NDArray[NDArray[np.float64]],
        NDArray[NDArray[np.float64]],
        NDArray[NDArray[np.float64]]
    ]:
        """Calculate the velocity over the FW-H surface.

        Returns
        -------
        NDArray[NDArray[np.float64]]
            Matrix of surface velocity in the x direction. Each row is a
            time step. Each column corresponds to the respective point
            on the surface.
        NDArray[NDArray[np.float64]]
            Same as above but for velocity in the y direction
        NDArray[NDArray[np.float64]]
            Same as above but for velocity in the z direction
        """

        def calculate_velocity(fn: Callable) -> NDArray[NDArray[np.float64]]:
            return fn(
                self.fw_h_surface.x[:, np.newaxis],
                self.fw_h_surface.y[:, np.newaxis],
                self.fw_h_surface.z[:, np.newaxis],
                self.config.source.centroid.x,
                self.config.source.centroid.y,
                self.config.source.centroid.z,
                self.time_domain[np.newaxis, :],
                self.config.source.amplitude,
                self.config.source.frequency,
                self.config.source.constants.c_0,
            ).T

        return (
            calculate_velocity(self._velocity_x_fn),
            calculate_velocity(self._velocity_y_fn),
            calculate_velocity(self._velocity_z_fn),
        )
