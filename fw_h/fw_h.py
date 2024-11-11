"""Entrypoint to the FW-H solver."""
import argparse


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "A solver for the Ffowcs Williams and Hawkings (FW-H) "
            "equation written in Python using Farassat's Formulation "
            "1A."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("-c", "--config-dir",
                        default="../config",
                        help="relative path to the config directory")

    return parser.parse_args()


def main():
    """Start the FW-H solver."""
    args = parse_arguments()


if __name__ == "__main__":
    main()
