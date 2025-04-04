import argparse
from datetime import datetime
from pathlib import Path


def generic_path(arg_path: str) -> Path:
    return Path(arg_path)


def existing_dir_path(arg_path: str) -> Path:
    arg_path = generic_path(arg_path)

    if not arg_path.exists():
        raise argparse.ArgumentTypeError(
            'Path "{}" does not exist.'.format(arg_path)
        )

    if not arg_path.is_dir():
        raise argparse.ArgumentTypeError(
            'Path "{}" is not a directory.'.format(arg_path)
        )

    return arg_path


def existing_file_path(arg_path: str) -> Path:
    arg_path = generic_path(arg_path)

    if not arg_path.exists():
        raise argparse.ArgumentTypeError(
            'Path "{}" does not exist.'.format(arg_path)
        )

    if not arg_path.is_file():
        raise argparse.ArgumentTypeError(
            'Path "{}" does not point to a file.'.format(arg_path)
        )

    return arg_path


def dir_to_be_created_if_not_exists(arg_path: str) -> Path:
    """
    This function should be used as a type in argparse when the argument being
    parsed might be an existing directory or a directory that has yet to be
    created.
    If the directory does not already exist, this function will create it
    before returning a Path object that refers to it.

    Parameters:
        arg_path (str): The input path to be validated or created as a
        directory if it does not exist.

    Returns:
        Path: A Path object that refers to the validated or created directory.

    Raises:
        argparse.ArgumentTypeError: If the argument refers to a directory that
        cannot be created due to non-existent parent directory.

        argparse.ArgumentTypeError: If the argument refers to a directory that
        cannot be created because its proposed parent is not a directory.

        argparse.ArgumentTypeError: If the argument refers to a location that
        exists but is not a directory.

    """

    arg_path = generic_path(arg_path)
    if not arg_path.parent.exists():
        raise argparse.ArgumentTypeError(
            'Path "{}" can not be created because "{}" does not exist.'.format(
                arg_path, arg_path.parent
            )
        )
    if not arg_path.parent.is_dir():
        raise argparse.ArgumentTypeError(
            'Path "{}" can not be created because "{}" is not a '
            "directory.".format(arg_path, arg_path.parent)
        )
    if not arg_path.exists():
        arg_path.mkdir()
    if not arg_path.is_dir():
        raise argparse.ArgumentTypeError(
            'Path "{}" is not a directory.'.format(arg_path)
        )
    return arg_path


def path_inside_an_existing_dir(arg_path: str) -> Path:
    arg_path = generic_path(arg_path)

    if not arg_path.exists():
        if not arg_path.parent.exists():
            raise argparse.ArgumentTypeError(
                f"Neither path {arg_path} nor its parent directory do exist"
            )

    return arg_path


def date_from_str(arg_path: str) -> datetime:
    formats = ("%Y%m%d", "%Y-%m-%d")

    output_date = None
    for date_format in formats:
        try:
            output_date = datetime.strptime(arg_path, date_format)
        except ValueError:
            continue
        break

    if output_date is None:
        raise argparse.ArgumentTypeError(
            'Argument "{}" can not be interpreted as a date; valid formats '
            "are: {}".format(
                arg_path, ",".join(('"{}"'.format(f) for f in formats))
            )
        )

    return output_date
