import argparse
import logging
from sys import exit as sys_exit

from ogs_riverger.efas.cli import add_efas_subcommands
from ogs_riverger.efas.cli import efas_cli
from ogs_riverger.efas.cli import EFAS_COMMAND
from ogs_riverger.pontelagoscuro.cli import add_pontelagoscuro_subcommands
from ogs_riverger.pontelagoscuro.cli import PONTELAGOSCURO_COMMAND
from ogs_riverger.settings import Settings


if __name__ == "__main__" or __name__ == "ogs_riverger.__main__":
    LOGGER = logging.getLogger()
else:
    LOGGER = logging.getLogger(__name__)


def configure_logger():
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    LOGGER.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)

    LOGGER.addHandler(handler)


def argument(sys_argv=None):
    """
    Generate a subparser for each type of file that can be produced by this
    script and delegate the task of configuring the subparser to the
    corresponding implementation
    """
    parser = argparse.ArgumentParser(
        description="A versatile tool to download and manage rivers' data "
        "from different sources"
    )
    subparsers = parser.add_subparsers(
        title="source",
        dest="cmd",
        required=True,
    )

    add_efas_subcommands(subparsers)
    add_pontelagoscuro_subcommands(subparsers)

    if sys_argv is not None:
        return parser.parse_args(sys_argv)
    else:
        return parser.parse_args()


def main():
    configure_logger()
    args = argument()

    cmd_map = {EFAS_COMMAND: efas_cli, PONTELAGOSCURO_COMMAND: None}
    settings = Settings()

    if args.cmd not in cmd_map:
        raise ValueError(f"Unknown command {args.cmd}")

    return cmd_map[args.cmd](args, settings)


if __name__ == "__main__":
    sys_exit(main())
