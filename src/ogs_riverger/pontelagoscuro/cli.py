PONTELAGOSCURO_COMMAND = "pontelagoscuro"


def add_pontelagoscuro_subcommands(subparsers):
    """Adds the section related to the sensor located at Pontelagoscuro to
    the main CLI.

    Args:
        subparsers: The subparsers obtained by calling the method
            `add_subparsers` of the main parser
    """

    # pontelagoscuro_parser = subparsers.add_parser(
    subparsers.add_parser(
        PONTELAGOSCURO_COMMAND,
        help="Retrieve data collected by a sensor that measure the discharge "
        "of the Po River near the Pontelagoscuro station (Ferrara).",
    )
    # TODO: Implement a sensible CLI for pontelagoscuro
