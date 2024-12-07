"""Define general geometric structures."""
import numpy as np
from numpy.typing import NDArray


class Surface:
    def __init__(self,
                 x: NDArray[np.float64],
                 y: NDArray[np.float64],
                 z: NDArray[np.float64]):
        self.x = x
        self.y = y
        self.z = z
