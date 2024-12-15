"""Generate the data resulting from a theoretical source."""

import datetime
import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import sympy as sp
import yaml
from numpy.typing import NDArray
from sympy.utilities.lambdify import lambdify

from fw_h.config import (
    ConfigSchema,
    SourceType,
    parse_shape_function,
)
from fw_h.geometry import (
    Normals,
    Surface,
    generate_fw_h_surface,
)

logger = logging.getLogger(__name__)


def evaluate_monopole_source_functions(
    source_shape_fn: sp.FunctionClass,
) -> tuple[Callable, Callable, Callable, Callable, Callable]:
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
    logger.info("Calculating analytical source functions symbolically...")
    x, y, z, x_0, y_0, z_0, t, amplitude, omega, c_0, rho_0 = sp.symbols(
        "x y z x_0 y_0 z_0 t amplitude omega c_0 rho_0"
    )

    r = sp.sqrt((x - x_0) ** 2 + (y - y_0) ** 2 + (z - z_0) ** 2)
    phi = -amplitude * source_shape_fn(omega * (t - r / c_0)) / (4 * sp.pi * r)
    logger.debug(
        "phi:\n%s\n%s",
        sp.latex(phi),
        sp.pretty(phi, use_unicode=True, wrap_line=False),
    )
    p = -rho_0 * sp.diff(phi, t)
    logger.debug(
        "p:\n%s\n%s",
        sp.latex(p),
        sp.pretty(p, use_unicode=True, wrap_line=False),
    )
    v_x = sp.diff(phi, x)
    logger.debug(
        "v_x:\n%s\n%s",
        sp.latex(v_x),
        sp.pretty(v_x, use_unicode=True, wrap_line=False),
    )
    v_y = sp.diff(phi, y)
    logger.debug(
        "v_y:\n%s\n%s",
        sp.latex(v_y),
        sp.pretty(v_y, use_unicode=True, wrap_line=False),
    )
    v_z = sp.diff(phi, z)
    logger.debug(
        "v_z:\n%s\n%s",
        sp.latex(v_z),
        sp.pretty(v_z, use_unicode=True, wrap_line=False),
    )

    logger.info("Lambdifying symbolic functions...")
    phi_fn = lambdify(
        (x, y, z, x_0, y_0, z_0, t, amplitude, omega, c_0),
        phi,
        modules="numpy",
    )
    p_fn = lambdify(
        (x, y, z, x_0, y_0, z_0, t, amplitude, omega, c_0, rho_0),
        p,
        modules="numpy",
    )
    v_x_fn = lambdify(
        (x, y, z, x_0, y_0, z_0, t, amplitude, omega, c_0),
        v_x,
        modules="numpy",
    )
    v_y_fn = lambdify(
        (x, y, z, x_0, y_0, z_0, t, amplitude, omega, c_0),
        v_y,
        modules="numpy",
    )
    v_z_fn = lambdify(
        (x, y, z, x_0, y_0, z_0, t, amplitude, omega, c_0),
        v_z,
        modules="numpy",
    )
    return phi_fn, p_fn, v_x_fn, v_y_fn, v_z_fn


class SourceData:
    """Generates all the data for an analytical acoustic source.

    To generate an analytical solution, run mesh(),
    compute_source_functions(), and finally compute().

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

    def __init__(self, config: ConfigSchema) -> None:
        logger.info("Initializing SourceData object...")
        self.config = config

        self.fw_h_surface: Surface = Surface(
            np.ndarray(0, dtype=np.float64),
            np.ndarray(0, dtype=np.float64),
            np.ndarray(0, dtype=np.float64),
        )
        self.observer_surface: Surface = Surface(
            np.ndarray(0, dtype=np.float64),
            np.ndarray(0, dtype=np.float64),
            np.ndarray(0, dtype=np.float64),
        )
        self.time_domain = np.ndarray(0, dtype=np.float64)
        self.source_shape_function = None
        self._velocity_potential_fn = None
        self._pressure_fn = None
        self._velocity_x_fn = None
        self._velocity_y_fn = None
        self._velocity_z_fn = None
        self.fw_h_velocity_potential = np.ndarray(0, dtype=np.float64)
        self.observer_velocity_potential = np.ndarray(0, dtype=np.float64)
        self.fw_h_pressure = np.ndarray(0, dtype=np.float64)
        self.observer_pressure = np.ndarray(0, dtype=np.float64)
        self.fw_h_velocity_x = np.ndarray(0, dtype=np.float64)
        self.fw_h_velocity_y = np.ndarray(0, dtype=np.float64)
        self.fw_h_velocity_z = np.ndarray(0, dtype=np.float64)

    def mesh(self) -> None:
        """Mesh the surfaces and discretize the time domain."""
        self.fw_h_surface = generate_fw_h_surface(
            self.config.fw_h_surface.r, self.config.fw_h_surface.n
        )

        logger.info("Meshing observer surface...")
        self.observer_surface = Surface(
            np.array([self.config.observer.centroid.x], dtype=np.float64),
            np.array([self.config.observer.centroid.y], dtype=np.float64),
            np.array([self.config.observer.centroid.z], dtype=np.float64),
        )

        logger.info("Discretizing source time domain")
        self.time_domain = np.linspace(
            self.config.source.time_domain.start_time,
            self.config.source.time_domain.end_time,
            self.config.source.time_domain.n,
            dtype=np.float64,
        )

    def _generate_source_functions(
        self,
    ) -> tuple[Callable, Callable, Callable, Callable, Callable]:
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
        source = self.config.source.description
        match source:
            case SourceType.MONOPOLE:
                functions = evaluate_monopole_source_functions(
                    self.source_shape_function
                )
            case _:
                err = f"Shape function '{source}' is not implemented."
                raise NotImplementedError(err)
        return functions

    def compute_source_functions(self) -> None:
        """Compute the source shape functions and the source functions.

        These source functions include the velocity potential, pressure,
        and velocity.
        """
        self.source_shape_function = parse_shape_function(
            self.config.source.shape
        )
        (
            self._velocity_potential_fn,
            self._pressure_fn,
            self._velocity_x_fn,
            self._velocity_y_fn,
            self._velocity_z_fn,
        ) = self._generate_source_functions()

    def compute(self) -> None:
        """Compute the source data.

        Raises
        ------
        RuntimeError
            If mesh(), compute_source_functions(), or both have not been
            called prior to running compute()

        """
        logger.info("Beginning source computation...")
        self.fw_h_velocity_potential = self.calculate_fw_h_velocity_potential()
        self.observer_velocity_potential = (
            self.calculate_observer_velocity_potential()
        )

        self.fw_h_pressure = self.calculate_fw_h_pressure()
        self.observer_pressure = self.calculate_observer_pressure()

        (self.fw_h_velocity_x, self.fw_h_velocity_y, self.fw_h_velocity_z) = (
            self.calculate_velocity()
        )

    def calculate_fw_h_velocity_potential(
        self,
    ) -> NDArray[NDArray[np.float64]]:
        """Calculate the velocity potential over the FW-H surface.

        Returns
        -------
        NDArray[NDArray[np.float64]]
            Matrix of velocity potentials. Each row is a time step. Each
            column corresponds to the respective point on the surface.

        """
        return self._calculate_velocity_potential(self.fw_h_surface)

    def calculate_observer_velocity_potential(
        self,
    ) -> NDArray[NDArray[np.float64]]:
        """Calculate the velocity potential over the observer surface.

        Returns
        -------
        NDArray[NDArray[np.float64]]
            Matrix of velocity potentials. Each row is a time step. Each
            column corresponds to the respective point on the surface.

        """
        return self._calculate_velocity_potential(self.observer_surface)

    def _calculate_velocity_potential(
        self, surface: Surface
    ) -> NDArray[NDArray[np.float64]]:
        """Calculate the velocity potential over a surface.

        Parameters
        ----------
        surface
            The surface to calculate the velocity potential over.

        Returns
        -------
        NDArray[NDArray[np.float64]]
            Matrix of velocity potentials. Each row is a time step. Each
            column corresponds to the respective point on the surface.

        """
        logger.info("Calculating velocity potential over surface...")
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
            self.config.source.constants.c_0,
        ).T

    def calculate_fw_h_pressure(self) -> NDArray[NDArray[np.float64]]:
        """Calculate the pressure over the FW-H surface.

        Returns
        -------
        NDArray[NDArray[np.float64]]
            Matrix of surface pressure. Each row is a time step. Each
            column corresponds to the respective point on the surface.

        """
        return self._calculate_pressure(self.fw_h_surface)

    def calculate_observer_pressure(self) -> NDArray[NDArray[np.float64]]:
        """Calculate the pressure over the observer surface.

        Returns
        -------
        NDArray[NDArray[np.float64]]
            Matrix of surface pressure. Each row is a time step. Each
            column corresponds to the respective point on the surface.

        """
        return self._calculate_pressure(self.fw_h_surface)

    def _calculate_pressure(
        self, surface: Surface
    ) -> NDArray[NDArray[np.float64]]:
        """Calculate the pressure over a surface.

        Parameters
        ----------
        surface
            The surface to calculate the pressure over.

        Returns
        -------
        NDArray[NDArray[np.float64]]
            Matrix of surface pressure. Each row is a time step. Each
            column corresponds to the respective point on the surface.

        """
        logger.info("Calculating pressure over surface...")
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
            self.config.source.constants.rho_0,
        ).T

    def calculate_velocity(
        self,
    ) -> tuple[
        NDArray[NDArray[np.float64]],
        NDArray[NDArray[np.float64]],
        NDArray[NDArray[np.float64]],
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
        logger.info("Calculating velocity over FW-H surface...")

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

    def write(self) -> None:
        """Write relevant data to binary files."""
        logger.info("Writing analytical source data to file...")
        output_dir = self.config.solver.output.output_dir

        timestamp = datetime.datetime.now(tz=datetime.UTC).strftime(
            self.config.solver.output.output_file_timestamp
        )

        config_path = Path(output_dir) / f"{timestamp}-fw-h-config.yaml"
        with config_path.open("w") as file:
            logger.info("Writing config to %s...", config_path)
            yaml.dump(
                self.config.model_dump(warnings="error"),
                file,
                default_flow_style=False,
            )

        data_path = Path(output_dir) / f"{timestamp}-fw-h.npz"
        logger.info("Writing data to %s...", data_path)
        logger.info("This may take a while...")

        np.savez_compressed(
            data_path,
            fw_h_surface_x=self.fw_h_surface.x,
            fw_h_surface_y=self.fw_h_surface.y,
            fw_h_surface_z=self.fw_h_surface.z,
            fw_h_surface_n_x=self.fw_h_surface.normals.n_x,
            fw_h_surface_n_y=self.fw_h_surface.normals.n_y,
            fw_h_surface_n_z=self.fw_h_surface.normals.n_z,
            observer_surface_x=self.observer_surface.x,
            observer_surface_y=self.observer_surface.y,
            observer_surface_z=self.observer_surface.z,
            time_domain=self.time_domain,
            fw_h_pressure=self.fw_h_pressure,
            observer_pressure=self.observer_pressure,
            fw_h_velocity_x=self.fw_h_velocity_x,
            fw_h_velocity_y=self.fw_h_velocity_y,
            fw_h_velocity_z=self.fw_h_velocity_z,
        )

    def load(self) -> None:
        """Load source data from binary files."""
        input_path = Path(self.config.solver.input.input_file)
        logger.info("Loading source data from %s...", input_path)
        logger.info("This may take a while...")
        source_data = np.load(input_path)
        logger.debug("Loading FW-H surface mesh...")
        self.fw_h_surface = Surface(
            source_data["fw_h_surface_x"],
            source_data["fw_h_surface_y"],
            source_data["fw_h_surface_z"],
            Normals(
                source_data["fw_h_surface_n_x"],
                source_data["fw_h_surface_n_y"],
                source_data["fw_h_surface_n_z"],
            ),
        )
        logger.debug("Loading observer mesh...")
        self.observer_surface = Surface(
            source_data["observer_surface_x"],
            source_data["observer_surface_y"],
            source_data["observer_surface_z"],
        )
        logger.debug("Loading time domain...")
        self.time_domain = source_data["time_domain"]
        logger.debug("Loading FW-H surface pressure...")
        self.fw_h_pressure = source_data["fw_h_pressure"]
        logger.debug("Loading observer pressure...")
        self.observer_pressure = source_data["observer_pressure"]
        logger.debug("Loading FW-H surface velocity...")
        self.fw_h_velocity_x = source_data["fw_h_velocity_x"]
        self.fw_h_velocity_y = source_data["fw_h_velocity_y"]
        self.fw_h_velocity_z = source_data["fw_h_velocity_z"]
