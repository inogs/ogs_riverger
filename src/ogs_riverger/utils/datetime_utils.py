from datetime import datetime
from datetime import timezone
from typing import Literal
from warnings import warn

import numpy as np


class MixedTimezoneAwareness(ValueError):
    """
    Represents an error raised when there is a mismatch or inconsistency in
    timezone awareness between two or more datetime objects.

    This error is typically used to identify and handle cases where operations
    on datetime objects require consistent timezone-awareness, but the
    provided datetime objects have mixed timezone awareness.

    For instance, this might occur when attempting to compare or perform an
    operation between a timezone-aware datetime object and a naive datetime
    object.

    This exception inherits from the built-in ValueError to signify that it is
    raised due to invalid or inconsistent values related to timezone awareness.
    """

    pass


def is_timezone_aware(dt: datetime) -> bool:
    """
    Determine whether a given datetime object is timezone-aware or not.

    This function checks the `tzinfo` attribute and the `utcoffset` method of
    the provided datetime object to determine whether it is time-aware or not.
    A timezone-aware datetime object contains timezone information that allows
    it to handle different time zones accurately.

    Args:
        dt: A datetime object to be checked for timezone awareness.
            If the `tzinfo` attribute is `None` or the result of `utcoffset`
            is `None`, the datetime is considered naive (not time-aware).
            Otherwise, it is considered time-aware.

    Returns:
        A boolean indicating whether the provided datetime object is
        timezone-aware (`True`) or timezone-naive (`False`).
    """
    naive = dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None
    return not naive


def check_all_timezone_awareness(*datetime_objects: datetime) -> bool:
    """
    Determines if all provided datetime objects have consistent timezone
    awareness (either all timezone-aware or all timezone-naive). If no
    datetime objects are provided, it assumes consistency and returns True.

    Args:
        *datetime_objects (datetime): Variable-length argument list of
            datetime objects to be checked for timezone awareness
            consistency.

    Returns:
        bool: True if all datetime objects have consistent timezone
        awareness or no datetime objects are provided, False otherwise.

    Raises:
        MixedTimezoneAwareness: If the datetime objects have mixed
        timezone awareness (some are timezone-aware and some are
        timezone-naive).
    """
    if len(datetime_objects) == 0:
        return True

    timezone_aware = is_timezone_aware(datetime_objects[0])

    for dt in datetime_objects:
        current_timezone_aware = is_timezone_aware(dt)
        if current_timezone_aware != timezone_aware:
            raise MixedTimezoneAwareness(
                "The provided datetime objects have mixed timezone "
                "awareness. All datetime objects must either be "
                "time-aware or time-naive."
            )

    return timezone_aware


def datetime_to_datetime64(
    dt: datetime,
    precision: Literal["s", "ms", "us", "ns"] = "us",
) -> np.datetime64:
    """Convert datetime to datetime64.

    If the datetime object is timezone-aware, it will be converted to
    UTC before conversion to datetime64.
    This function silences a warning that happens every time a time-aware
    object is converted into a datetime64 object and triggers a warning when
    the code uses non-UTC timezones or naive datetime objects.

    Args:
        dt: The datetime object to convert.
        precision: The precision of the datetime64 object.

    Returns:
        The datetime64 object.
    """
    warn_message = (
        "A datetime object that was expected to be timezone-aware and to "
        "have its timezone be set to UTC "
    )

    if is_timezone_aware(dt):
        if dt.tzname() != "UTC":
            warn_problem = f" has its timezone set to {dt.tzname()}."
            warn(warn_message + warn_problem)
            dt = dt.astimezone(timezone.utc)
    else:
        warn_problem = " was instead a naive datetime object."
        warn(warn_message + warn_problem)

    # Remove the timezone, so the conversion to datetime64 does not trigger a
    # warning
    naive_dt = dt.replace(tzinfo=None)
    return np.datetime64(naive_dt, precision)
