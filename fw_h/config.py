"""Import the FW-H configuration."""
from enum import Enum
from typing import Callable

import numpy as np
import yaml
from pydantic import (
    BaseModel,
    Field,
)


class SourceType(Enum):
    MONOPOLE = "monopole"
    DIPOLE = "dipole"
    QUADRUPOLE = "quadrupole"


class Centroid(BaseModel):
    x: float = Field(description="X-coordinate value [L]")
    y: float = Field(description="Y-coordinate value [L]")
    z: float = Field(description="Z-coordinate value [L]")


class Constants(BaseModel):
    c_0: float = Field(description="Speed of sound [L * T^-1]")
    rho_0: float = Field(description="Density of fluid [M * L^-3]")
    p_0: float = Field(description="Ambient pressure [F * L^-2]")
    T_0: float = Field(description="Ambient temperature [Θ]")


class TimeDomain(BaseModel):
    start_time: float = Field(description="Start time [T]")
    end_time: float = Field(description="End time [T]")
    n: int = Field(description="Number of time steps")


class Source(BaseModel):
    centroid: Centroid
    description: SourceType = Field(description="Source type description")
    shape: str = Field(description="Shape of the source signal")
    amplitude: float = Field(description="Amplitude of the source signal")
    frequency: float = Field(description="Frequency of the source signal")
    constants: Constants
    time_domain: TimeDomain


# noinspection PyPep8Naming
class FW_H_Surface(BaseModel):
    centroid: Centroid
    r: float = Field(description="Perpendicular distance from centroid to "
                                 "each face of the FW-H surface")
    n: int = Field(description="Number of points on each face of the FW-H "
                               "surface. If n is not a perfect square, the "
                               "number of points per face will be rounded to "
                               "the nearest perfect square.")


class Observer(BaseModel):
    centroid: Centroid


class ConfigSchema(BaseModel):
    source: Source
    fw_h_surface: FW_H_Surface
    observer: Observer


class Config:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = self._load_and_validate()

    def _load_and_validate(self) -> ConfigSchema:
        with open(self.file_path, "r") as file:
            raw_data = yaml.safe_load(file)
            return ConfigSchema(**raw_data)

    def get(self) -> ConfigSchema:
        return self.data


def parse_shape_function(shape: str) -> Callable[
    [np.ndarray[np.float64]],
    np.ndarray[np.float64]
]:
    """Parse the shape function field into a callable function.

    Parameters
    ----------
    shape
        String name of the shape function

    Returns
    -------
    Callable[[np.ndarray[np.float64]], np.ndarray[np.float64]]
        Function to be used as the shape function

    Raises
    ------
    ValueError
        If the shape function field is not an implemented callable
    """
    match shape:
        case "sin":
            fn = np.sin
        case "cos":
            fn = np.cos
        case _:
            raise ValueError(f"Invalid shape function: {shape}")
    return fn