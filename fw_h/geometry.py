"""Define general geometric structures."""
import math

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


def generate_fw_h_surface(r: float,
                          n: int) -> Surface:
    """Generate the FW-H surface as a hollow cube.

    Assumes center of the surface is at (0, 0, 0). Faces of the cube
    will be orthogonal to the coordinate axes.

    Parameters
    ----------
    r
        Perpendicular distance from x_0 to each face
    n
        Number of points per face
    """
    # TODO: implement the ability to specify a centroid from the config
    fw_h_surface = Surface(np.empty(0), np.empty(0), np.empty(0))
    z_faces_x, z_faces_y, z_faces_z = np.meshgrid(
        np.linspace(-r, r, round(math.sqrt(n))),
        np.linspace(-r, r, round(math.sqrt(n))),
        (-r, r)
    )
    y_faces_x, y_faces_y, y_faces_z = np.meshgrid(
        np.linspace(-r, r, round(math.sqrt(n))),
        (-r, r),
        np.linspace(-r, r, round(math.sqrt(n)))
    )
    x_faces_x, x_faces_y, x_faces_z = np.meshgrid(
        (-r, r),
        np.linspace(-r, r, round(math.sqrt(n))),
        np.linspace(-r, r, round(math.sqrt(n)))
    )
    fw_h_surface.x = np.concatenate((z_faces_x.ravel(),
                                     y_faces_x.ravel(),
                                     x_faces_x.ravel()))
    fw_h_surface.y = np.concatenate((z_faces_y.ravel(),
                                     y_faces_y.ravel(),
                                     x_faces_y.ravel()))
    fw_h_surface.z = np.concatenate((z_faces_z.ravel(),
                                     y_faces_z.ravel(),
                                     x_faces_z.ravel()))
    return fw_h_surface
