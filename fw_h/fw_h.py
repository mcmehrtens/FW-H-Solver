"""Entrypoint to the FW-H solver."""

import argparse
import datetime
import json
import logging
from pathlib import Path

import yaml

from fw_h.config import (
    Config,
    ConfigSchema,
)
from fw_h.solver import Solver
from fw_h.source import SourceData
from fw_h.visualize import animate_surface_pressure, plot_observer_pressure


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns
    -------
    Namespace
        The parsed arguments.

    """
    parser = argparse.ArgumentParser(
        description=(
            "A solver for the Ffowcs Williams and Hawkings (FW-H) equation "
            "using Farassat's Formulation 1A written in Python."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("config", help="relative path to the config file")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="increase logging verbosity",
    )
    parser.add_argument(
        "-s", "--source", action="store_true", help="generate source data"
    )
    parser.add_argument(
        "-S",
        "--write-source",
        action="store_true",
        help="write source data to file",
    )

    return parser.parse_args()


def configure_logging(
    logging_dir: str,
    log_file_timestamp: str,
    timestamp: datetime,
    *,
    verbose: bool,
) -> logging.Logger:
    """Configure the logging module.

    Parameters
    ----------
    logging_dir
        The directory where logs should be written.
    log_file_timestamp
        The timestamp format to prepend to log files.
    timestamp
        The approximate time at which the program began executing.
    verbose
        If true, sets the logging level to DEBUG, otherwise, the logging
        level will be set to INFO.

    Returns
    -------
    Logger
        This modules logger.

    """
    verbosity = logging.DEBUG if verbose else logging.INFO

    console_handler = logging.StreamHandler()
    console_handler.setLevel(verbosity)

    start_time = timestamp.strftime(log_file_timestamp)
    file_handler = logging.FileHandler(
        Path(logging_dir) / f"{start_time}-fw-h.log"
    )
    file_handler.setLevel(verbosity)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    formatter.default_msec_format = "%s.%03d"

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(verbosity)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    return logging.getLogger(__name__)


def source_routine(
    logger: logging.Logger, config: ConfigSchema, *, write: bool
) -> None:
    """Generate and optionally write the source data to a file.

    Parameters
    ----------
    logger
        This module's logging object.
    config
        The configuration object parsed from the configuration file.
    write
        Whether the source data should be written to a file.

    """
    logger.info("Beginning source generation routine...")
    source = SourceData(config)
    source.calculate_pressure()
    source.calculate_velocity()
    if write:
        logger.info("Beginning source writing routine...")
        exclude = ("velocity_potential",)
        source.write(*exclude)

    animate_surface_pressure(
        source.surface, source.time_domain, source.pressure
    )


def solver_routine(logger: logging.Logger, config: ConfigSchema) -> None:
    """Solve for observer pressure given the input data.

    Parameters
    ----------
    logger
        This module's logging object.
    config
        The configuration object parsed from the configuration file.

    """
    logger.info("Beginning solver routine...")
    source = SourceData(config)
    source.load()
    solver = Solver(config, source)
    solver.compute()
    solver.validate()
    solver.write()
    plot_observer_pressure(
        solver.time_domain, solver.p, solver.p_analytical[0]
    )


def main() -> None:
    """Start the FW-H solver."""
    program_start = datetime.datetime.now(tz=datetime.UTC)
    args = parse_arguments()

    config = Config(args.config).data

    logger = configure_logging(
        config.global_config.logging.logging_dir,
        config.global_config.logging.log_file_timestamp,
        program_start,
        verbose=args.verbose,
    )

    logger.info("Starting FW-H solver...")
    logger.debug(
        "Command-Line Arguments:\n%s", json.dumps(vars(args), indent=4)
    )
    logger.debug(
        "Configuration file:\n%s",
        yaml.dump(
            config.model_dump(warnings="error"), default_flow_style=False
        ),
    )

    if args.source or args.write_source:
        source_routine(logger, config, write=args.write_source)
    else:
        solver_routine(logger, config)

    logger.info("Exiting script...")


if __name__ == "__main__":
    main()
