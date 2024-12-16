"""Visualize source or solution data."""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PathCollection
from numpy.typing import NDArray

from fw_h.geometry import Surface


def animate_surface_pressure(
    surface: Surface,
    time: NDArray[np.float64],
    pressure: NDArray[NDArray[np.float64]],
) -> None:
    """Plot surface pressure over time in 3D space.

    Parameters
    ----------
    surface
        The surface to plot.
    time
        The time domain to plot over.
    pressure
        The pressure values for each point on the plot. Each row[i] of
        this matrix should correspond to a time step time[i]. Each
        column[j] of this matrix should correspond to a point
        (surface.x[j], surface.y[j], surface.z[j]).
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    norm = plt.Normalize(np.min(pressure), np.max(pressure))
    cmap = plt.get_cmap("viridis")

    def init() -> tuple[PathCollection]:
        """Initialize the scatter plot with the first time step.

        Returns
        -------
        PathCollection
            The initial pressure scatter plot.
        """
        ax.clear()
        ax.set_title("FW-H Surface Pressure")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        scatter = ax.scatter(
            surface.x,
            surface.y,
            surface.z,
            c=pressure[0],
            cmap=cmap,
            norm=norm,
        )
        fig.colorbar(scatter, ax=ax, orientation="vertical", label="Pressure")
        return (scatter,)

    def update(frame: int) -> tuple[PathCollection]:
        """Update the scatter plot for each time step.

        Parameters
        ----------
        frame
            The timestep to plot.

        Returns
        -------
        PathCollection
            The scatter plot at timestep t[frame].
        """
        ax.clear()
        ax.set_title(f"FW-H Surface Pressure at t={time[frame]:.4f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        scatter = ax.scatter(
            surface.x,
            surface.y,
            surface.z,
            c=pressure[frame],
            cmap=cmap,
            norm=norm,
        )
        return (scatter,)

    # necessary to prevent animation from being garbage collected
    ani = FuncAnimation(  # noqa: F841
        fig, update, frames=len(time), init_func=init, blit=False
    )

    plt.show()


def plot_observer_pressure(
    time: NDArray[np.float64],
    pressure: NDArray[np.float64],
    validation_pressure: NDArray[np.float64] = None,
) -> None:
    """Plot the calculated observer pressure against the analytical.

    Parameters
    ----------
    time
        The time domain.
    pressure
        The calculated observer pressure.
    validation_pressure
        The analytical observer pressure.
    """
    plt.figure()
    plt.plot(time, pressure, label="Observer Pressure")
    if validation_pressure is not None:
        plt.plot(
            time,
            validation_pressure,
            label="Analytical Observer Pressure",
            linestyle="--",
            color="red",
        )
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Pressure [Pa]")
    plt.title("Observer Pressure vs. Time")
    plt.legend()
    plt.show()
