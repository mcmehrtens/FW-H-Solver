"""Import the FW-H configuration."""
import yaml
from pydantic import (
    BaseModel,
    Field,
)


class Coordinate(BaseModel):
    x: float = Field(description="X-coordinate value")
    y: float = Field(description="Y-coordinate value")
    z: float = Field(description="Z-coordinate value")


class Constants(BaseModel):
    c_0: float = Field(description="Speed of sound in m/s")
    rho_0: float = Field(description="Density of fluid in kg/m^3")
    p_0: float = Field(description="Ambient pressure in Pa")
    T_0: float = Field(description="Ambient temperature in ºC")


class Source(BaseModel):
    coordinate: Coordinate
    shape: str = Field(description="Shape of the source signal")
    amplitude: float = Field(description="Amplitude of the source signal")
    constants: Constants


class ConfigSchema(BaseModel):
    source: Source


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
