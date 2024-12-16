"""Import the FW-H configuration."""

from enum import Enum
from pathlib import Path

import sympy as sp
import yaml
from pydantic import (
    BaseModel,
    Field,
)


class SourceType(Enum):
    """Types of supported theoretical acoustic sources."""

    MONOPOLE = "monopole"
    DIPOLE = "dipole"
    QUADRUPOLE = "quadrupole"


class Logging(BaseModel):
    """Logging configuration."""

    logging_dir: str = Field(description="The directory to store logs in.")
    log_file_timestamp: str = Field(
        description="The format of the timestamps prepended to log files. "
        "Timestamps are created using datetime.datetime.strftime()."
    )


class Output(BaseModel):
    """Output configuration."""

    output_dir: str = Field(
        description="The directory to store output files in."
    )
    output_file_timestamp: str = Field(
        description="The format of the timestamp prepended to output files. "
        "Timestamps are created using datetime.datetime.strftime()."
    )


class GlobalConfig(BaseModel):
    """Global configuration settings.

    These settings apply to the source generation and solving routines.

    """

    logging: Logging
    output: Output


class Input(BaseModel):
    """Input configuration."""

    data_file_path: str = Field(description="The path to the input data file.")


class Constants(BaseModel):
    """Physical constants."""

    c_0: float = Field(
        description="The speed of sound in the fluid [L * T^-1]."
    )
    rho_0: float = Field(description="The density of the fluid [M * L^-3].")


class Point(BaseModel):
    """Defines a point in 3D space."""

    x: float = Field(description="The x-coordinate value [L].")
    y: float = Field(description="The y-coordinate value [L].")
    z: float = Field(description="The z-coordinate value [L].")


class Observer(BaseModel):
    """Observer configuration."""

    point: Point


class Solver(BaseModel):
    """Configuration settings used by the solving routine."""

    input: Input
    constants: Constants
    time_steps: int = Field(description="The number of observer time steps.")
    observer: Observer


class TimeDomain(BaseModel):
    """Time domain configuration."""

    start_time: float = Field(description="The start time [T].")
    end_time: float = Field(description="The end time [T].")
    time_steps: int = Field(description="The number of time steps.")


class FWHSurface(BaseModel):
    """FW-H surface configuration."""

    point: Point
    r: float = Field(
        description="The perpendicular distance from centroid to each face of "
        "the FW-H surface."
    )
    n: int = Field(
        description="The number of points on each edge of the FW-H surface."
    )


class Source(BaseModel):
    """Configuration settings used by the source generation routine."""

    point: Point
    description: SourceType = Field(description="The source type description.")
    shape: str = Field(description="The shape of the source function.")
    amplitude: float = Field(description="The amplitude of the source signal.")
    frequency: float = Field(description="The frequency of the source signal.")
    constants: Constants
    time_domain: TimeDomain
    fw_h_surface: FWHSurface


class ConfigSchema(BaseModel):
    """Top-level configuration schema."""

    global_config: GlobalConfig
    solver: Solver
    source: Source


class Config:
    """Encapsulate all the configuration options for the FW-H solver.

    Parameters
    ----------
    file_path
        The path to the YAML configuration file.

    Attributes
    ----------
    file_path
    data
        The parsed and validated configuration object.

    """

    def __init__(self, file_path: str) -> None:
        self.file_path = Path(file_path)
        self.data = self._load_and_validate()

    def _load_and_validate(self) -> ConfigSchema:
        """Open, parse, and validate the YAML configuration file.

        Returns
        -------
        ConfigSchema
            Parsed and validated configuration object

        """
        with self.file_path.open() as file:
            raw_data = yaml.safe_load(file)
            return ConfigSchema(**raw_data)


def parse_shape_function(shape: str) -> sp.FunctionClass:
    """Parse the shape function field into a callable function.

    As of writing, this function only supports sinusoidal SymPy
    functions.

    Parameters
    ----------
    shape
        A string literal representation of the shape function.

    Returns
    -------
    FunctionClass
        The SymPy function to be used as the shape function.

    Raises
    ------
    ValueError
        Raised if the shape function field is not an implemented
        callable.

    """
    match shape.lower():
        case "sin":
            fn = sp.sin
        case _:
            err = f"Shape function '{shape}' is not implemented."
            raise NotImplementedError(err)
    return fn
