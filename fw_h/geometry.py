"""Define general geometric structures."""
import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class Surface:
    """Represent a 3-D surface.

    Parameters
    ----------
    x
        Flattened list of x coordinates
    y
        Flattened list of y coordinates
    z
        Flattened list of z coordinates
    n_x : optional
        x-component of the normal vector. See notes below.
    n_y : optional
        y-component of the normal vector. See notes below.
    n_z : optional
        z-component of the normal vector. See notes below.

    Attributes
    ----------
    x
    y
    z
    n_x
    n_y
    n_z

    Notes
    -----
    For each point i on the surface (x[i], y[i], z[i]), the normal
    vector, if defined is [n_x[i], n_y[i], n_z[i]]. If this surface is
    representing an observer surface, it is unlikely to have normal
    vectors defined since they server no purpose for FW-H analysis.
    """

    def __init__(self,
                 x: NDArray[np.float64],
                 y: NDArray[np.float64],
                 z: NDArray[np.float64],
                 n_x: NDArray[np.float64] = None,
                 n_y: NDArray[np.float64] = None,
                 n_z: NDArray[np.float64] = None):
        self.x = x
        self.y = y
        self.z = z
        self.n_x = n_x
        self.n_y = n_y
        self.n_z = n_z


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

    Returns
    -------
    Surface
        Object representing the cuboid FW-H surface
    """
    # TODO: implement the ability to specify a centroid from the config
    logger.info("Meshing FW-H surface...")
    delta = 2 * r / n
    logger.debug("Mesh cell length: %f [L]", delta)
    logger.debug("Number of points: %d", 6 * n ** 2)

    # sp ≔ positive s-face; sn ≔ negative s-face; *_n ≔ normal vector
    zp_x, zp_y, zp_z = np.meshgrid(
        np.linspace(-r + delta / 2, r - delta / 2, n, dtype=np.float64),
        np.linspace(-r + delta / 2, r - delta / 2, n, dtype=np.float64),
        np.float64(r)
    )
    zp_n_x = np.full_like(zp_x, 0, dtype=np.float64)
    zp_n_y = np.full_like(zp_y, 0, dtype=np.float64)
    zp_n_z = np.full_like(zp_z, 1, dtype=np.float64)

    zn_x, zn_y, zn_z = np.meshgrid(
        np.linspace(-r + delta / 2, r - delta / 2, n, dtype=np.float64),
        np.linspace(-r + delta / 2, r - delta / 2, n, dtype=np.float64),
        np.float64(-r)
    )
    zn_n_x = np.full_like(zn_x, 0, dtype=np.float64)
    zn_n_y = np.full_like(zn_y, 0, dtype=np.float64)
    zn_n_z = np.full_like(zn_z, -1, dtype=np.float64)

    yp_x, yp_y, yp_z = np.meshgrid(
        np.linspace(-r + delta / 2, r - delta / 2, n, dtype=np.float64),
        np.float64(r),
        np.linspace(-r + delta / 2, r - delta / 2, n, dtype=np.float64)
    )
    yp_n_x = np.full_like(yp_x, 0, dtype=np.float64)
    yp_n_y = np.full_like(yp_y, 1, dtype=np.float64)
    yp_n_z = np.full_like(yp_z, 0, dtype=np.float64)

    yn_x, yn_y, yn_z = np.meshgrid(
        np.linspace(-r + delta / 2, r - delta / 2, n, dtype=np.float64),
        np.float64(-r),
        np.linspace(-r + delta / 2, r - delta / 2, n, dtype=np.float64)
    )
    yn_n_x = np.full_like(yn_x, 0, dtype=np.float64)
    yn_n_y = np.full_like(yn_y, -1, dtype=np.float64)
    yn_n_z = np.full_like(yn_z, 0, dtype=np.float64)

    xp_x, xp_y, xp_z = np.meshgrid(
        np.float64(r),
        np.linspace(-r + delta / 2, r - delta / 2, n, dtype=np.float64),
        np.linspace(-r + delta / 2, r - delta / 2, n, dtype=np.float64)
    )
    xp_n_x = np.full_like(xp_x, 1, dtype=np.float64)
    xp_n_y = np.full_like(xp_y, 0, dtype=np.float64)
    xp_n_z = np.full_like(xp_z, 0, dtype=np.float64)

    xn_x, xn_y, xn_z = np.meshgrid(
        np.float64(-r),
        np.linspace(-r + delta / 2, r - delta / 2, n, dtype=np.float64),
        np.linspace(-r + delta / 2, r - delta / 2, n, dtype=np.float64)
    )
    xn_n_x = np.full_like(xn_x, -1, dtype=np.float64)
    xn_n_y = np.full_like(xn_y, 0, dtype=np.float64)
    xn_n_z = np.full_like(xn_z, 0, dtype=np.float64)

    logger.debug("Concatenating faces...")
    return Surface(
        np.concatenate((
            zp_x.ravel(), zn_x.ravel(),
            yp_x.ravel(), yn_x.ravel(),
            xp_x.ravel(), xn_x.ravel()
        )),
        np.concatenate((
            zp_y.ravel(), zn_y.ravel(),
            yp_y.ravel(), yn_y.ravel(),
            xp_y.ravel(), xn_y.ravel()
        )),
        np.concatenate((
            zp_z.ravel(), zn_z.ravel(),
            yp_z.ravel(), yn_z.ravel(),
            xp_z.ravel(), xn_z.ravel()
        )),
        np.concatenate((
            zp_n_x.ravel(), zn_n_x.ravel(),
            yp_n_x.ravel(), yn_n_x.ravel(),
            xp_n_x.ravel(), xn_n_x.ravel()
        )),
        np.concatenate((
            zp_n_y.ravel(), zn_n_y.ravel(),
            yp_n_y.ravel(), yn_n_y.ravel(),
            xp_n_y.ravel(), xn_n_y.ravel()
        )),
        np.concatenate((
            zp_n_z.ravel(), zn_n_z.ravel(),
            yp_n_z.ravel(), yn_n_z.ravel(),
            xp_n_z.ravel(), xn_n_z.ravel()
        ))
    )
