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
        The SymPy function to use for the shape of the source
        perturbations.

    Returns
    -------
    Callable
        The velocity potential field function.
    Callable
        The pressure field function.
    Callable
        The velocity field (x-component) function.
    Callable
        The velocity field (y-component) function.
    Callable
        The velocity field (z-component) function.

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
    """Represents the data relevant to a synthetic acoustic source.

    To generate an analytical solution, run mesh(),
    compute_source_functions(), and finally compute(). Use write() and
    load() to save and load the source data.

    Parameters
    ----------
    config
        The parsed configuration object.

    Attributes
    ----------
    config
    fw_h_surface
        The penetrable FW-H surface encapsulating the sources.
    observer_surface
        The observer points to calculate the theoretical solution for.
    time_domain
        The time steps in the source time domain.
    source_shape_function
        The shape of the pressure perturbations caused by the source.
    fw_h_velocity_potential
        The velocity potential on the FW-H surface calculated over time.
        Each row corresponds to a time step. Each column corresponds to
        the same index on the surface.
    observer_velocity_potential
        Same as fw_h_velocity_potential, but for the observer surface.
    fw_h_pressure
        The pressure on the FW-H surface calculated over time. Each row
        corresponds to a time step. Each column corresponds to the same
        index on the surface.
    observer_pressure
        Same as fw_h_pressure, but for the observer surface.
    fw_h_velocity_x
        The x-component of velocity on the FW-H surface calculated over
        time. Each row corresponds to a time step. Each column
        corresponds to the same index on the surface.
    fw_h_velocity_y
        See fw_h_velocity_x.
    fw_h_velocity_z
        See fw_h_velocity_x.

    """

    def __init__(self, config: ConfigSchema) -> None:
        logger.info("Initializing SourceData object...")
        self.config = config

        self.fw_h_surface = Surface(
            np.ndarray(0, dtype=np.float64),
            np.ndarray(0, dtype=np.float64),
            np.ndarray(0, dtype=np.float64),
        )
        self.observer_surface = Surface(
            np.ndarray(0, dtype=np.float64),
            np.ndarray(0, dtype=np.float64),
            np.ndarray(0, dtype=np.float64),
        )
        self.time_domain = np.ndarray(0, dtype=np.float64)
        self.source_shape_function: sp.FunctionClass | None = None
        self._velocity_potential_fn: Callable | None = None
        self._pressure_fn: Callable | None = None
        self._velocity_x_fn: Callable | None = None
        self._velocity_y_fn: Callable | None = None
        self._velocity_z_fn: Callable | None = None
        self.fw_h_velocity_potential = np.ndarray(
            np.ndarray(0, dtype=np.float64)
        )
        self.observer_velocity_potential = np.ndarray(
            np.ndarray(0, dtype=np.float64)
        )
        self.fw_h_pressure = np.ndarray(np.ndarray(0, dtype=np.float64))
        self.observer_pressure = np.ndarray(np.ndarray(0, dtype=np.float64))
        self.fw_h_velocity_x = np.ndarray(np.ndarray(0, dtype=np.float64))
        self.fw_h_velocity_y = np.ndarray(np.ndarray(0, dtype=np.float64))
        self.fw_h_velocity_z = np.ndarray(np.ndarray(0, dtype=np.float64))

    def mesh(self) -> None:
        """Mesh the surfaces and discretize the time domain.

        Creates a 3D cube as the FW-H surface. This function currently
        only supports a single point as the observer "surface."

        """
        self.fw_h_surface = generate_fw_h_surface(
            self.config.source.fw_h_surface.r,
            self.config.source.fw_h_surface.n,
        )

        logger.info("Meshing observer surface...")
        self.observer_surface = Surface(
            np.array([self.config.source.observer.point.x], dtype=np.float64),
            np.array([self.config.source.observer.point.y], dtype=np.float64),
            np.array([self.config.source.observer.point.z], dtype=np.float64),
        )

        logger.info("Discretizing source time domain")
        self.time_domain = np.linspace(
            self.config.source.time_domain.start_time,
            self.config.source.time_domain.end_time,
            self.config.source.time_domain.time_steps,
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
            The velocity potential field function.
        Callable
            The pressure field function.
        Callable
            The velocity field (x-component) function.
        Callable
            The velocity field (y-component) function.
        Callable
            The velocity field (z-component) function.

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
        and velocity functions.

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
        """Compute the source data."""
        logger.info("Beginning source computation...")
        self.calculate_fw_h_velocity_potential()
        self.calculate_observer_velocity_potential()

        self.calculate_fw_h_pressure()
        self.calculate_observer_pressure()

        (self.fw_h_velocity_x, self.fw_h_velocity_y, self.fw_h_velocity_z) = (
            self.calculate_velocity()
        )

    def calculate_fw_h_velocity_potential(self) -> None:
        """Calculate the velocity potential over the FW-H surface."""
        logger.info("Calculating velocity potential over the FW-H surface...")
        self.fw_h_velocity_potential = self._calculate_velocity_potential(
            self.fw_h_surface
        )

    def calculate_observer_velocity_potential(self) -> None:
        """Calculate velocity potential over the observer surface."""
        logger.info(
            "Calculating velocity potential over the observer surface..."
        )
        self.observer_velocity_potential = self._calculate_velocity_potential(
            self.observer_surface
        )

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
        return self._velocity_potential_fn(
            surface.x[:, np.newaxis],
            surface.y[:, np.newaxis],
            surface.z[:, np.newaxis],
            self.config.source.point.x,
            self.config.source.point.y,
            self.config.source.point.z,
            self.time_domain[np.newaxis, :],
            self.config.source.amplitude,
            self.config.source.frequency,
            self.config.source.constants.c_0,
        ).T

    def calculate_fw_h_pressure(self) -> None:
        """Calculate the pressure over the FW-H surface."""
        logger.info("Calculating pressure over the FW-H surface...")
        self.fw_h_pressure = self._calculate_pressure(self.fw_h_surface)

    def calculate_observer_pressure(self) -> None:
        """Calculate the pressure over the observer surface."""
        logger.info("Calculating pressure over the observer surface...")
        self.observer_pressure = self._calculate_pressure(self.fw_h_surface)

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
        return self._pressure_fn(
            surface.x[:, np.newaxis],
            surface.y[:, np.newaxis],
            surface.z[:, np.newaxis],
            self.config.source.point.x,
            self.config.source.point.y,
            self.config.source.point.z,
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
            Same as above but for velocity in the y direction.
        NDArray[NDArray[np.float64]]
            Same as above but for velocity in the z direction.

        """
        logger.info("Calculating velocity over FW-H surface...")

        def calculate_velocity(fn: Callable) -> NDArray[NDArray[np.float64]]:
            return fn(
                self.fw_h_surface.x[:, np.newaxis],
                self.fw_h_surface.y[:, np.newaxis],
                self.fw_h_surface.z[:, np.newaxis],
                self.config.source.point.x,
                self.config.source.point.y,
                self.config.source.point.z,
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
        """Write relevant data to a NumPy .npz archive."""
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
        """Load source data from NumPy .npz archive."""
        input_path = Path(self.config.solver.input.data_file_path)
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
