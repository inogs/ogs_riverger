from pydantic import BaseModel
from pydantic import model_serializer


class AreaSelection(BaseModel):
    """Defines the geographic area for data download.

    This class is can be used by several different modules. At the time
    being, however, it is only used by the `EFAS` module (CMES CDSAPI).

    Attributes:
        north (float): Northern boundary latitude of the area.
        west (float): Western boundary longitude of the area.
        south (float): Southern boundary latitude of the area.
        east (float): Eastern boundary longitude of the area.
    """

    north: float
    west: float
    south: float
    east: float

    @model_serializer
    def ser_model(self) -> tuple[float, float, float, float]:
        """Serializes the bounding box coordinates to a tuple.

        Returns:
            tuple[float, float, float, float]: A tuple containing the
            north, west, south, and east coordinates in this order.
        """
        return self.north, self.west, self.south, self.east
