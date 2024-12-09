"""Entrypoint to the FW-H solver."""
import argparse

from matplotlib import pyplot as plt

from fw_h.config import Config
from fw_h.source import SourceData


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns
    -------
    Namespace
        The parsed arguments
    """
    parser = argparse.ArgumentParser(
        description=(
            "A solver for the Ffowcs Williams and Hawkings (FW-H) "
            "equation using Farassat's Formulation 1A written in Python."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("config",
                        help="relative path to the config file")

    return parser.parse_args()


def main():
    """Start the FW-H solver."""
    args = parse_arguments()
    config = Config(args.config).get()

    source = SourceData(config)

    plt.figure()
    plt.plot(source.fw_h_surface.x)
    plt.show()


if __name__ == "__main__":
    main()
