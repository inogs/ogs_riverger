from typing import Literal

from pydantic import BaseModel


class EFASConfigElement(BaseModel):
    type: Literal["EFAS"]
    longitude: float
    latitude: float
    longitude_index: int
    latitude_index: int
