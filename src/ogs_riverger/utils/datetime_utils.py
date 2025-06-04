from datetime import datetime


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
