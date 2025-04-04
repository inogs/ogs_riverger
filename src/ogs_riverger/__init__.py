from typing import Any

from pydantic import BaseModel


class RiverRawData(BaseModel):
    """
    Represents raw data for a single river.

    This class stores information about a river, including its unique ID, name,
    and a dictionary containing its attributes. The raw data is designed to be
    passed to another class for further processing and interpretation of the
    attributes.
    """

    id: int
    name: str
    raw_data: dict[str, Any]
