"""Import the FW-H configuration."""

from enum import Enum

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

    logging_dir: str = Field(description="Logging directory")
    log_file_timestamp: str = Field(description="Log file timestamp")


class Input(BaseModel):
    """Input configuration."""

    input_file: str = Field(description="Path to input file")


class Output(BaseModel):
    """Output configuration."""

    output_dir: str = Field(description="Output directory")
    output_file_timestamp: str = Field(description="Output file timestamp")


class Constants(BaseModel):
    """Constants configuration."""

    c_0: float = Field(description="Speed of sound [L * T^-1]")
    rho_0: float = Field(description="Density of fluid [M * L^-3]")


class Solver(BaseModel):
    """Solver configuration."""

    logging: Logging
    input: Input
    output: Output
    constants: Constants
    time_steps: int = Field(description="Number of observer time steps")


class Centroid(BaseModel):
    """Centroid configuration."""

    x: float = Field(description="X-coordinate value [L]")
    y: float = Field(description="Y-coordinate value [L]")
    z: float = Field(description="Z-coordinate value [L]")


class TimeDomain(BaseModel):
    """Time domain configuration."""

    start_time: float = Field(description="Start time [T]")
    end_time: float = Field(description="End time [T]")
    n: int = Field(description="Number of time steps")


class Source(BaseModel):
    """Source configuration."""

    centroid: Centroid
    description: SourceType = Field(description="Source type description")
    shape: str = Field(description="Shape of the source signal")
    amplitude: float = Field(description="Amplitude of the source signal")
    frequency: float = Field(description="Frequency of the source signal")
    constants: Constants
    time_domain: TimeDomain


class FW_H_Surface(BaseModel):
    """FW H surface configuration."""

    centroid: Centroid
    r: float = Field(
        description="Perpendicular distance from centroid to "
        "each face of the FW-H surface"
    )
    n: int = Field(
        description="Number of points on each edge of the FW-H " "surface."
    )


class Observer(BaseModel):
    """Observer configuration."""

    centroid: Centroid


class ConfigSchema(BaseModel):
    """Config schema."""

    solver: Solver
    source: Source
    fw_h_surface: FW_H_Surface
    observer: Observer


class Config:
    """Encapsulate all the configuration options for the FW-H solver.

    Parameters
    ----------
    file_path
        Path to the YAML configuration file

    Attributes
    ----------
    file_path
    data
        Parsed and validated configuration object

    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = self._load_and_validate()

    def _load_and_validate(self) -> ConfigSchema:
        """Open, parse, and validate the YAML configuration file.

        Returns
        -------
        ConfigSchema
            Parsed and validated configuration object

        """
        with open(self.file_path, "r") as file:
            raw_data = yaml.safe_load(file)
            return ConfigSchema(**raw_data)

    def get(self) -> ConfigSchema:
        """Return the configuration object.

        Returns
        -------
        ConfigSchema
            Parsed and validated configuration object

        """
        return self.data


def parse_shape_function(shape: str) -> sp.FunctionClass:
    """Parse the shape function field into a callable function.

    Parameters
    ----------
    shape
        String name of the shape function

    Returns
    -------
    FunctionClass
        SymPy function to be used as the shape function

    Raises
    ------
    ValueError
        If the shape function field is not an implemented callable

    """
    match shape.lower():
        case "sin":
            fn = sp.sin
        case _:
            raise ValueError(f"Invalid shape function: {shape}")
    return fn
