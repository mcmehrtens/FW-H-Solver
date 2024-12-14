"""Entrypoint to the FW-H solver."""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from fw_h.config import (
    Config,
    ConfigSchema,
)
from fw_h.solver import Solver
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
    verbose: bool,
    logging_dir: str,
    log_file_timestamp: str,
    timestamp: datetime,
) -> logging.Logger:
    """Configure the logging module.

    Parameters
    ----------
    verbose
        If true, sets the logging level to DEBUG, otherwise, the logging
        level will be set to INFO
    logging_dir
        Directory where logs should be written
    log_file_timestamp
        Timestamp format for logging files
    timestamp
        Approximate time at which the program started

    Returns
    -------
    Logger
        This modules logger
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
        "%(asctime)s " "- %(name)s " "- %(levelname)s " "- %(message)s"
    )
    formatter.default_msec_format = "%s.%03d"

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(verbosity)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    return logging.getLogger(__name__)


def source_routine(logger: logging.Logger, config: ConfigSchema, write: bool):
    """Generate and optionally write the source data to a file.

    Parameters
    ----------
    logger
        This modules logging object
    config
        Configuration object parsed from the YAML configuration file
    write
        Whether the source data should be written to a file
    """
    logger.info("Beginning source generation routine...")
    source = SourceData(config)
    source.mesh()
    source.compute_source_functions()
    source.compute()
    if write:
        logger.info("Beginning source writing routine...")
        source.write()


def solver_routine(logger: logging.Logger, config: ConfigSchema):
    """Solve for observer pressure given the input data.

    Parameters
    ----------
    logger
        This modules logging object
    config
        Configuration object parsed from the YAML configuration file
    """
    logger.info("Beginning solver routine...")
    source = SourceData(config)
    source.load()
    solver = Solver(config, source)
    solver.compute()

    logger.info("Writing solution...")
    np.savez_compressed(
        Path(config.solver.output.output_dir) / "solution.npz",
        source_time=solver.source.time_domain,
        analytical_observer_pressure=solver.source.observer_pressure,
        observer_time=solver.time_domain,
        observer_pressure=solver.p,
    )


def main():
    """Start the FW-H solver."""
    program_start = datetime.now()
    args = parse_arguments()

    config = Config(args.config).get()

    logger = configure_logging(
        args.verbose,
        config.solver.logging.logging_dir,
        config.solver.logging.log_file_timestamp,
        program_start,
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
        source_routine(logger, config, args.write_source)
    else:
        solver_routine(logger, config)

    logger.info("Exiting script...")


if __name__ == "__main__":
    main()
