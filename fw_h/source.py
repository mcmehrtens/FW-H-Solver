"""Generate the data resulting from a theoretical source."""

import datetime
import logging
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

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


class SymbolicSourceParams(NamedTuple):
    """Encapsulates symbolic source parameters.

    Attributes
    ----------
    x, y, z
        The observer or position of interest in 3D space.
    x_0, y_0, z_0
        The source position in 3D space.
    t
        The source time.
    amplitude
        The amplitude of the source shape function.
    omega
        The frequency of the source shape function.
    c_0
        The speed of sound in the fluid.
    rho_0
        The density of the fluid.
    """

    x: sp.Symbol
    y: sp.Symbol
    z: sp.Symbol
    x_0: sp.Symbol
    y_0: sp.Symbol
    z_0: sp.Symbol
    t: sp.Symbol
    amplitude: sp.Symbol
    omega: sp.Symbol
    c_0: sp.Symbol
    rho_0: sp.Symbol


class NumericalSourceParams(NamedTuple):
    """Encapsulates numerical source parameters.

    Attributes
    ----------
    x, y, z
        The observer or position of interest in 3D space.
    x_0, y_0, z_0
        The source position in 3D space.
    t
        The source time.
    amplitude
        The amplitude of the source shape function.
    omega
        The frequency of the source shape function.
    c_0
        The speed of sound in the fluid.
    rho_0
        The density of the fluid.
    """

    x: NDArray[NDArray[np.float64]]
    y: NDArray[NDArray[np.float64]]
    z: NDArray[NDArray[np.float64]]
    x_0: float
    y_0: float
    z_0: float
    t: NDArray[NDArray[np.float64]]
    amplitude: float
    omega: float
    c_0: float
    rho_0: float


class AnalyticalSource:
    """Abstracts the derivation of different analytical sound sources.

    Represents the common functions between monopoles, dipoles, and
    longitudinal quadrupoles. It's unclear whether this will need to be
    modified to accommodate non-axisymmetric sources.

    Child classes must implement the velocity potential function.

    Parameters
    ----------
    source_shape_fn
        The SymPy function which defines the source perturbation shape.

    Attributes
    ----------
    source_shape_fn
    """

    def __init__(
        self, source_shape_fn: Callable[[sp.Basic], sp.Basic]
    ) -> None:
        self.source_shape_fn = source_shape_fn
        self._params = SymbolicSourceParams(
            x=sp.Symbol("x"),
            y=sp.Symbol("y"),
            z=sp.Symbol("z"),
            x_0=sp.Symbol("x_0"),
            y_0=sp.Symbol("y_0"),
            z_0=sp.Symbol("z_0"),
            t=sp.Symbol("t"),
            amplitude=sp.Symbol("amplitude"),
            omega=sp.Symbol("omega"),
            c_0=sp.Symbol("c_0"),
            rho_0=sp.Symbol("rho_0"),
        )
        self._setup_functions()

    def _velocity_potential_fn(self) -> sp.Basic:
        err = "Subclasses must implement this method."
        raise NotImplementedError(err)

    def _setup_functions(self) -> None:
        """Symbolically calculate the physical source functions.

        Symbolically calculates the velocity potential, pressure, and
        surface velocity functions for a given source. Once calculated,
        it lambdifies these functions for use with NumPy.
        """
        logger.info("Calculating analytical source functions symbolically...")
        phi = self._velocity_potential_fn()
        p = -self._params.rho_0 * sp.diff(phi, self._params.t)
        v_x = sp.diff(phi, self._params.x)
        v_y = sp.diff(phi, self._params.y)
        v_z = sp.diff(phi, self._params.z)

        logger.debug(
            "Symbolic velocity potential (phi):\n%s\n%s",
            sp.pretty(phi, use_unicode=True, wrap_line=False),
            sp.latex(phi),
        )
        logger.debug(
            "Symbolic pressure (p):\n%s\n%s",
            sp.pretty(p, use_unicode=True, wrap_line=False),
            sp.latex(p),
        )
        logger.debug(
            "Symbolic velocity in x (v_x):\n%s\n%s",
            sp.pretty(v_x, use_unicode=True, wrap_line=False),
            sp.latex(v_x),
        )
        logger.debug(
            "Symbolic velocity in y (v_y):\n%s\n%s",
            sp.pretty(v_y, use_unicode=True, wrap_line=False),
            sp.latex(v_y),
        )
        logger.debug(
            "Symbolic velocity in z (v_z):\n%s\n%s",
            sp.pretty(v_z, use_unicode=True, wrap_line=False),
            sp.latex(v_z),
        )

        logger.info("Lambdifying symbolic functions...")
        self._phi_fn = lambdify(self._params, phi, modules="numpy")
        self._p_fn = lambdify(self._params, p, modules="numpy")
        self._v_x_fn = lambdify(self._params, v_x, modules="numpy")
        self._v_y_fn = lambdify(self._params, v_y, modules="numpy")
        self._v_z_fn = lambdify(self._params, v_z, modules="numpy")

    def velocity_potential(
        self, params: NumericalSourceParams
    ) -> NDArray[NDArray[np.float64]]:
        """Calculate the velocity potential.

        Parameters
        ----------
        params
            The arguments of the velocity potential function.

        Returns
        -------
        NDArray[NDArray[np.float64]]
            The velocity potential matrix. Each row corresponds to a
            time step in params.t. Each column i corresponds to a point
            (params.x[i], params.y[i], params.z[i]).
        """
        return self._phi_fn(
            params.x,
            params.y,
            params.z,
            params.x_0,
            params.y_0,
            params.z_0,
            params.t,
            params.amplitude,
            params.omega,
            params.c_0,
            0,
        )

    def pressure(
        self, params: NumericalSourceParams
    ) -> NDArray[NDArray[np.float64]]:
        """Calculate the pressure.

        Parameters
        ----------
        params
            The arguments of the pressure function.

        Returns
        -------
        NDArray[NDArray[np.float64]]
            The pressure matrix. Each row corresponds to a time step in
            params.t. Each column i corresponds to a point
            (params.x[i], params.y[i], params.z[i]).
        """
        return self._p_fn(
            params.x,
            params.y,
            params.z,
            params.x_0,
            params.y_0,
            params.z_0,
            params.t,
            params.amplitude,
            params.omega,
            params.c_0,
            params.rho_0,
        )

    def velocity_x(
        self, params: NumericalSourceParams
    ) -> NDArray[NDArray[np.float64]]:
        """Calculate the velocity in the x-direction.

        Parameters
        ----------
        params
            The arguments of the velocity function in the x-direction.

        Returns
        -------
        NDArray[NDArray[np.float64]]
            The x-component of the velocity matrix. Each row corresponds
            to a time step in params.t. Each column i corresponds to a
            point (params.x[i], params.y[i], params.z[i]).
        """
        return self._v_x_fn(
            params.x,
            params.y,
            params.z,
            params.x_0,
            params.y_0,
            params.z_0,
            params.t,
            params.amplitude,
            params.omega,
            params.c_0,
            0,
        )

    def velocity_y(
        self, params: NumericalSourceParams
    ) -> NDArray[NDArray[np.float64]]:
        """Calculate the velocity in the y-direction.

        Parameters
        ----------
        params
            The arguments of the velocity function in the y-direction.

        Returns
        -------
        NDArray[NDArray[np.float64]]
            The y-component of the velocity matrix. Each row corresponds
            to a time step in params.t. Each column i corresponds to a
            point (params.x[i], params.y[i], params.z[i]).
        """
        return self._v_y_fn(
            params.x,
            params.y,
            params.z,
            params.x_0,
            params.y_0,
            params.z_0,
            params.t,
            params.amplitude,
            params.omega,
            params.c_0,
            0,
        )

    def velocity_z(
        self, params: NumericalSourceParams
    ) -> NDArray[NDArray[np.float64]]:
        """Calculate the velocity in the z-direction.

        Parameters
        ----------
        params
            The arguments of the velocity function in the z-direction.

        Returns
        -------
        NDArray[NDArray[np.float64]]
            The z-component of the velocity matrix. Each row corresponds
            to a time step in params.t. Each column i corresponds to a
            point (params.x[i], params.y[i], params.z[i]).
        """
        return self._v_z_fn(
            params.x,
            params.y,
            params.z,
            params.x_0,
            params.y_0,
            params.z_0,
            params.t,
            params.amplitude,
            params.omega,
            params.c_0,
            0,
        )


class MonopoleSource(AnalyticalSource):
    """Calculate monopole source functions."""

    def _velocity_potential_fn(self) -> sp.Basic:
        """Symbolically calculate the velocity potential.

        Returns
        -------
        sp.Basic
            The velocity potential function as a symbolic expression.
        """
        r = sp.sqrt(
            (self._params.x - self._params.x_0) ** 2
            + (self._params.y - self._params.y_0) ** 2
            + (self._params.z - self._params.z_0) ** 2
        )
        return (
            self._params.amplitude
            * self.source_shape_fn(
                self._params.omega * (self._params.t - r / self._params.c_0)
            )
            / r
        )


class DipoleSource(AnalyticalSource):
    """Calculate dipole source functions."""

    def _velocity_potential_fn(self) -> sp.Basic:
        """Symbolically calculate the velocity potential.

        Returns
        -------
        sp.Basic
            The velocity potential function as a symbolic expression.
        """
        r = sp.symbols("r")
        r_expr = sp.sqrt(
            (self._params.x - self._params.x_0) ** 2
            + (self._params.y - self._params.y_0) ** 2
            + (self._params.z - self._params.z_0) ** 2
        )

        phi = (
            self._params.z
            / r
            * sp.diff(
                self._params.amplitude
                * self.source_shape_fn(
                    self._params.omega
                    * (self._params.t - r / self._params.c_0)
                )
                / r,
                r,
            )
        )

        return phi.subs(r, r_expr)


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
    surface
        The penetrable FW-H surface encapsulating the sources.
    time_domain
        The time steps in the source time domain.
    velocity_potential
        The velocity potential on the FW-H surface calculated over time.
        Each row corresponds to a time step. Each column corresponds to
        the same index on the surface.
    pressure
        The pressure on the FW-H surface calculated over time. Each row
        corresponds to a time step. Each column corresponds to the same
        index on the surface.
    velocity_x
        The x-component of velocity on the FW-H surface calculated over
        time. Each row corresponds to a time step. Each column
        corresponds to the same index on the surface.
    velocity_y
        See fw_h_velocity_x.
    velocity_z
        See fw_h_velocity_x.

    """

    def __init__(self, config: ConfigSchema) -> None:
        self.config = config

        self._mesh()
        self._compute_source_description()

        self.data = NumericalSourceParams(
            x=self.surface.x[:, np.newaxis],
            y=self.surface.y[:, np.newaxis],
            z=self.surface.z[:, np.newaxis],
            x_0=self.config.source.point.x,
            y_0=self.config.source.point.y,
            z_0=self.config.source.point.z,
            t=self.time_domain[np.newaxis, :],
            amplitude=self.config.source.amplitude,
            omega=self.config.source.frequency,
            c_0=self.config.source.constants.c_0,
            rho_0=self.config.source.constants.rho_0,
        )

        self.velocity_potential = np.array([[0]], dtype=np.float64)
        self.pressure = np.array([[0]], dtype=np.float64)
        self.velocity_x = np.array([[0]], dtype=np.float64)
        self.velocity_y = np.array([[0]], dtype=np.float64)
        self.velocity_z = np.array([[0]], dtype=np.float64)

    def _mesh(self) -> None:
        """Mesh the surfaces and discretize the time domain.

        Creates a 3D cube as the surface.

        """
        self.surface = generate_fw_h_surface(
            self.config.source.fw_h_surface.r,
            self.config.source.fw_h_surface.n,
        )

        logger.info("Discretizing source time domain")
        self.time_domain = np.linspace(
            self.config.source.time_domain.start_time,
            self.config.source.time_domain.end_time,
            self.config.source.time_domain.time_steps,
            dtype=np.float64,
        )

    def _compute_source_description(self) -> None:
        """Compute the source shape functions and the source functions.

        These source functions include the velocity potential, pressure,
        and velocity functions.

        """
        self._source_shape_function = parse_shape_function(
            self.config.source.shape
        )

        source = self.config.source.description
        match source:
            case SourceType.MONOPOLE:
                self.analytical_source = MonopoleSource(
                    self._source_shape_function
                )
            case SourceType.DIPOLE:
                self.analytical_source = DipoleSource(
                    self._source_shape_function
                )
            case _:
                err = f"Shape function '{source}' is not implemented."
                raise NotImplementedError(err)

    def calculate_velocity_potential(self) -> None:
        """Calculate the velocity potential."""
        logger.info("Calculating velocity potential...")
        self.velocity_potential = self.analytical_source.velocity_potential(
            self.data
        ).T

    def calculate_pressure(self) -> None:
        """Calculate the pressure."""
        logger.info("Calculating pressure...")
        self.pressure = self.analytical_source.pressure(self.data).T

    def calculate_velocity(self) -> None:
        """Calculate the velocity."""
        logger.info("Calculating velocity...")
        self.velocity_x = self.analytical_source.velocity_x(self.data).T
        self.velocity_y = self.analytical_source.velocity_y(self.data).T
        self.velocity_z = self.analytical_source.velocity_z(self.data).T

    def write(self, *exclude: tuple[str, ...]) -> None:
        """Write relevant data to a NumPy .npz archive.

        Parameters
        ----------
        exclude
            List of arrays to exclude from writing to the file.

        """
        logger.info("Writing analytical source data to file...")
        arrays = {
            "surface_x": self.surface.x,
            "surface_y": self.surface.y,
            "surface_z": self.surface.z,
            "surface_n_x": self.surface.normals.n_x,
            "surface_n_y": self.surface.normals.n_y,
            "surface_n_z": self.surface.normals.n_z,
            "time_domain": self.time_domain,
            "velocity_potential": self.velocity_potential,
            "pressure": self.pressure,
            "velocity_x": self.velocity_x,
            "velocity_y": self.velocity_y,
            "velocity_z": self.velocity_z,
        }
        arrays = {k: v for k, v in arrays.items() if k not in exclude}

        output_dir = self.config.global_config.output.output_dir

        timestamp = datetime.datetime.now(tz=datetime.UTC).strftime(
            self.config.global_config.output.output_file_timestamp
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
        np.savez_compressed(data_path, **arrays)

    def load(self) -> None:
        """Load source data from NumPy .npz archive."""
        input_path = Path(self.config.solver.input.data_file_path)
        logger.info("Loading source data from %s...", input_path)
        logger.info("This may take a while...")
        source_data = np.load(input_path)

        if (
            "surface_x" in source_data
            and "surface_y" in source_data
            and "surface_z" in source_data
        ):
            logger.debug("Loading surface mesh...")
            normals = (
                Normals(
                    source_data["surface_n_x"],
                    source_data["surface_n_y"],
                    source_data["surface_n_z"],
                )
                if "surface_n_x" in source_data
                and "surface_n_y" in source_data
                and "surface_n_z" in source_data
                else None
            )

            self.surface = Surface(
                source_data["surface_x"],
                source_data["surface_y"],
                source_data["surface_z"],
                normals,
            )

        if "time_domain" in source_data:
            logger.debug("Loading time domain...")
            self.time_domain = source_data["time_domain"]

        if "pressure" in source_data:
            logger.debug("Loading surface pressure...")
            self.pressure = source_data["pressure"]

        if (
            "velocity_x" in source_data
            and "velocity_y" in source_data
            and "velocity_z" in source_data
        ):
            logger.debug("Loading surface velocity...")
            self.velocity_x = source_data["velocity_x"]
            self.velocity_y = source_data["velocity_y"]
            self.velocity_z = source_data["velocity_z"]
