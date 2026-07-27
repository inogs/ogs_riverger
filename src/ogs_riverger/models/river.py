import inspect
import json
from abc import ABC
from abc import abstractmethod
from collections import deque
from collections.abc import Sequence
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from enum import Enum
from itertools import product as cart_prod
from numbers import Real
from typing import Annotated
from typing import Callable
from typing import cast
from typing import List
from typing import Tuple

import numpy as np
from bitsea.commons.utils import search_closest_sorted
from bitsea.utilities.bc import Side
from pydantic import BaseModel
from pydantic import BeforeValidator
from pydantic import ConfigDict
from pydantic import field_serializer
from pydantic import field_validator

from ogs_riverger.utils.datetime_utils import datetime_to_datetime64


Coordinate2DList = List[Tuple[int, int]]

NpDType = Annotated[str, BeforeValidator(lambda x: np.dtype(x).name)]


class VelocityDirection(Enum):
    """Direction of water fluxes.

    "zonal" for East-West movement (or vice versa), "meridional" for
    South-North movement (or vice versa).
    """

    ZONAL = "u"
    MERIDIONAL = "v"


class Flux:
    """A numpy array with associated `VelocityDirection`.

    The values inside the numpy array can be positive or negative: for
    zonal velocity, a positive flux indicates water flowing from West
    to East (and vice versa for a negative value).
    For meridional velocity, a positive flux means water is flowing
    from South to North.
    """

    def __init__(self, values: np.ndarray, direction: VelocityDirection):
        self._values = np.asarray(values)
        self._direction = direction

    @property
    def direction(self) -> VelocityDirection:
        """Direction of flow."""
        return self._direction

    def __array__(self):
        return self._values

    @property
    def shape(self):
        """Shape of river values."""
        return self._values.shape

    @property
    def dtype(self):
        """Dtype of river values."""
        return self._values.dtype

    def __getitem__(self, item):
        output_data = self._values.__getitem__(item)
        if output_data.ndim == 0:
            return output_data

        return self.__class__(output_data, self._direction)

    def __setitem__(self, key, value):
        return self._values.__setitem__(key, value)

    def __repr__(self):
        return f"Flux({self._values}, {self._direction})"

    def __eq__(self, other):
        if not hasattr(other, "direction"):
            return False
        if self._direction != other.direction:
            return False

        if not hasattr(other, "__getitem__"):
            return NotImplemented
        return bool(np.all(self._values == np.asarray(other)))

    @classmethod
    def from_side(cls, values: np.ndarray, side: Side) -> "Flux":
        """Generates a flux from positive values and a `Side` object.

        This function is typically used to create boundary fluxes based
        on boundary conditions. It associates each `Side` with the
        appropriate `VelocityDirection`, while also adjusting the flux
        by multiplying by -1 for values on the East or North sides.
        This adjustment is necessary because flux entering from these
        sides moves in the opposite direction to the one typically
        considered positive in the model.

        Args:
            values: Array of positive values, usually velocities
                or fluxes.
            side: Side object.

        Returns:
            A Flux object with the same values, multiplied by -1 if needed
        """
        match side:
            case Side.NORTH:
                return cls(values * -1, VelocityDirection.MERIDIONAL)
            case Side.WEST:
                return cls(values, VelocityDirection.ZONAL)
            case Side.SOUTH:
                return cls(values, VelocityDirection.MERIDIONAL)
            case Side.EAST:
                return cls(values * -1, VelocityDirection.ZONAL)


class RiverGeometry(BaseModel):
    """Information about a river's mouth.

    This object contains all the information that we store about the mouth
    of a river.

    Attributes:
        mouth_lat: The latitude of the mouth of the river.
        mouth_lon: The longitude of the mouth of the river.
        lat_indices: The latitude indices of the cells where the mouth is
        lon_indices: The longitude indices of the cells where the mouth is
        side (Side): The `Side` of the river as a boundary condition
    """

    mouth_lat: float
    mouth_lon: float
    lat_indices: list[int]
    lon_indices: list[int]
    side: Side

    model_config = ConfigDict(use_enum_values=False)

    @field_serializer("side", when_used="always")
    def serialize_side(self, side: Side, _info):
        """Serialize the side enum using just its name (N, S, W, or E)."""
        return side.value

    def concerned_cells(self) -> Coordinate2DList:
        """Returns the cells from where the flux starts.

        Returns a list of tuples (i,j) of all the cells from where the flux
        of the river starts.
        The first index is the longitude and the second is the latitude.
        """
        return list(cart_prod(self.lon_indices, self.lat_indices))


class RiverComponent(ABC):
    """A specific component of a River.

    This object represents the value of a single variable for a river, such
    as temperature, salinity, or any other biogeochemical variable.

    The only constraint is that the computation of this variable must be
    independent of other variables (i.e., this class cannot be used for
    the salinity if the computation requires the knowledge of the
    temperature).
    """

    @abstractmethod
    def __call__(self, time_list: Sequence[datetime]) -> np.ndarray:
        """Returns the values of this component.

        Returns the values of this component computed on the assigned
        time-steps

        Args:
            time_list: a sequence of timezone-aware datetime objects

        Returns:
            A numpy array with the same shape as `time_list`.
        """
        raise NotImplementedError

    def serialize(self) -> dict[str, str | Real | list | dict]:
        """Serialize the component.

        Transform the RiverComponent into a dictionary that can be saved as
        a JSON file. The dictionary contains the class name and the data
        associated with the Component. Each component has the responsibility
        to implement the serialization of its data.

        Returns:
            A dictionary with the name of the class and the data associated
            with the instance.

        """
        return {
            "class": self.__class__.__name__,
            "data": self.serialize_data(),
        }

    @abstractmethod
    def serialize_data(self) -> dict[str, str | Real | list | dict]:
        """Serialize the data of this component."""
        raise NotImplementedError

    @staticmethod
    def deserialize(
        data: dict[str, str | Real | list | dict],
    ) -> "RiverComponent":
        """Reconstruct a RiverComponent from serialized data.

        This method is the inverse operation of serialize(). It takes a
        dictionary containing the class name and data of a previously
        serialized RiverComponent and reconstructs the original object.

        Args:
            data: A dictionary containing:
                - "class": The name of the RiverComponent subclass
                - "data": The serialized data specific to that component type

        Returns:
            A new instance of the appropriate RiverComponent subclass

        Raises:
            ValueError: If the class name in the data is not a known
                RiverComponent subclass
        """
        # Get all subclasses of RiverComponent
        subclasses = {}
        to_check = deque([RiverComponent])
        while len(to_check) > 0:
            current_class = to_check.popleft()
            # Only add non-abstract classes to subclasses dictionary
            if not inspect.isabstract(current_class):
                subclasses[current_class.__name__] = current_class
            for subclass in current_class.__subclasses__():
                to_check.append(subclass)

        # Get the component class name
        class_name = data["class"]

        if class_name not in subclasses:
            raise ValueError(f"Unknown river component class: {class_name}")

        # Get the appropriate class and deserialize
        component_class = subclasses[class_name]
        return component_class.deserialize_data(data["data"])

    @classmethod
    @abstractmethod
    def deserialize_data(
        cls, data: dict[str, str | Real | list | dict]
    ) -> "RiverComponent":
        """Inverse operation of serialize_data().

        This abstract method must be implemented by subclasses to reconstruct
        their specific RiverComponent type from serialized data. It is used
        internally by the deserialize() method to handle the component-specific
        deserialization logic.

        Args:
            data: A dictionary containing the component-specific serialized
                data that was previously created by serialize_data()

        Returns:
            A new instance of the specific RiverComponent subclass
        """
        raise NotImplementedError

    def __neg__(self):
        return OppositeRiverComponent(self)

    def __add__(self, other):
        if isinstance(other, RiverComponent):
            return SummedRiverComponent((self, other))
        if not isinstance(other, Real):
            return NotImplemented
        return AdditiveComponent(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, RiverComponent):
            return SummedRiverComponent((self, -other))
        if not isinstance(other, Real):
            return NotImplemented
        return AdditiveComponent(self, -other)

    def __rsub__(self, other):
        if isinstance(other, Real):
            return OppositeRiverComponent(self) + other
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, RiverComponent):
            return MultipliedRiverComponent((self, other))
        if not isinstance(other, Real):
            return NotImplemented
        return MultiplicativeComponent(self, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, RiverComponent):
            return RatioRiverComponent(self, other)
        if not isinstance(other, Real):
            return NotImplemented
        return self * (1.0 / other)

    def __rtruediv__(self, other):
        if isinstance(other, Real):
            return ReciprocalRiverComponent(self) * other
        return NotImplemented


class FixedRiverComponent(RiverComponent, BaseModel):
    """A RiverComponent that always returns the same value.

    A RiverComponent that consistently returns the same value, regardless of
    the time.
    """

    value: float
    dtype: NpDType = "float64"

    def serialize_data(self) -> dict[str, str | Real | list | dict]:
        """Serialize the data of this component."""
        return self.model_dump()

    @classmethod
    def deserialize_data(
        cls, data: dict[str, str | Real | list | dict]
    ) -> "FixedRiverComponent":
        """Deserialize the data of this component."""
        return cls(**data)

    def __call__(self, time_list: Sequence[datetime]) -> np.ndarray:
        """Returns the values of this component."""
        n_values = len(time_list)
        return np.full((n_values,), self.value, dtype=self.dtype)


class ZeroRiverComponent(FixedRiverComponent):
    """A RiverComponent that always returns zero, regardless of the time."""

    def __init__(self, dtype: NpDType = np.float64):
        super().__init__(value=0, dtype=dtype)

    def __repr__(self):
        return f"{self.__class__.__name__}(dtype = {self.dtype})"

    def serialize_data(self) -> dict[str, str | Real | list | dict]:
        """Serialize the data of this component."""
        output_data = self.model_dump()
        if "value" in output_data:
            del output_data["value"]
        return output_data


class SinusoidalRiverComponent(RiverComponent, BaseModel):
    """A RiverComponent that follows a sinusoidal pattern throughout the year.

    Several parameters can be submitted that define the shape of the
    sinusoidal function throughout the year.
    The `average` represents the average value of the function over
    the course of the year. Analytically, the function is defined as
    the average plus a sinusoidal component with an amplitude of
    `average * modulation`. This means the maximum and minimum values
    returned by this object are `average(1 + modulation)` and
    `average(1 - modulation)`, respectively.
    The maximum value is reached when `peak` days have passed since
    the first of January.
    If `n_peaks` is 1, there is only one maximum (occurring `peak`
    days after the first of January) and a minimum six months later.
    If `n_peaks` is greater than 1, the period of the sinusoidal is
    divided by `n_peaks`, resulting in multiple peaks: specifically,
    `n_peaks` peaks throughout the year.
    """

    average: float
    modulation: float
    peak: int
    n_peaks: int = 1
    dtype: NpDType = "float64"

    @field_validator("n_peaks", mode="after")
    @classmethod
    def n_peaks_positive(cls, value: int) -> int:
        """Validate that n_peaks is positive.

        This validator ensures that the n_peaks value is greater than or equal
        to 1. The n_peaks parameter determines how many peaks occur throughout
        the year in the sinusoidal pattern.

        Args:
            value: The n_peaks value to validate

        Returns:
            The validated n_peaks value if it's valid

        Raises:
            ValueError: If n_peaks is less than 1
        """
        if value < 1:
            raise ValueError(
                "n_peaks must be greater or equal to 1: received {}".format(
                    value
                )
            )
        return value

    def serialize_data(self) -> dict[str, str | Real | list | dict]:
        """Serialize the data of this component."""
        return self.model_dump()

    @classmethod
    def deserialize_data(
        cls, data: dict[str, str | Real | list | dict]
    ) -> "SinusoidalRiverComponent":
        """Deserialize the data of this component."""
        return cls(**data)

    def __call__(self, time_list: Sequence[datetime]) -> np.ndarray:
        """Returns the values of this component."""
        # The days since the first day of the year
        julian_day = np.array(
            [int(t.strftime("%j")) for t in time_list], dtype=np.int32
        )

        alpha = self.average
        beta = self.modulation
        phi = 2 * self.n_peaks * np.pi / 365 * (julian_day - self.peak - 1)
        return np.asarray(alpha * (1.0 + beta * np.cos(phi)), dtype=self.dtype)


class _FromDatasetRiverComponentData(BaseModel):
    time_list: list[datetime]
    values: list[float]
    max_error: timedelta | None
    upper_limit: float | None


class FromDatasetRiverComponent(RiverComponent):
    """A `RiverComponent` generated from a dataset.

    This RiverComponent associates a sequence of times with a sequence
    of values. Then, the class performs nearest-neighbor interpolation and,
    for a given time, returns the value closest in time.

    If the returned value corresponds to a time that deviates from the
    requested time by more than `max_error` (and `max_error` is not
    `None`), an error is raised.

    The temporal resolution of this object is 1 millisecond. If two
    different datetime values are provided with a difference of less than
    1 millisecond, they will be treated as the same datetime, causing the
    code to raise an error.

    If `upper_limit` is specified, the component ensures that no value
    exceeds this limit. When a value exceeds the limit, it is capped
    and the excess is redistributed to the next time interval, scaled by
    the ratio of their time interval areas. This preserves the total flow
    while respecting the upper limit constraint. The last value is simply
    capped if it exceeds the limit (no redistribution).
    """

    def __init__(
        self,
        time_list: Sequence[datetime],
        values: np.ndarray,
        max_error: timedelta | None = timedelta(hours=1),
        upper_limit: float | None = None,
    ):
        self._values = np.asarray(values)
        self._max_error = max_error
        self._upper_limit = upper_limit
        if self._upper_limit is not None and self._upper_limit <= 0:
            raise ValueError("upper_limit must be positive")

        self._time_array = np.array(time_list, dtype="datetime64[ms]")
        if np.any(self._time_array[1:] - self._time_array[:-1] <= 0):
            raise ValueError(
                "time_list must be a strictly increasing list of dates and "
                "the minimum allowed difference between two dates is 1 "
                f"millisecond. Received: {time_list}"
            )

        if len(self._values.shape) != 1:
            raise ValueError(
                "values must be a 1D Numpy array: received {}".format(values)
            )

        if self._values.shape[0] != self._time_array.shape[0]:
            raise ValueError(
                f"values and time_list must have the same number of elements; "
                f"values has shape {self._values.shape} and time_list has "
                f"{self._time_array.shape[0]} elements"
            )

        # Apply upper limit capping logic if upper_limit is specified
        if self._upper_limit is not None:
            self._apply_upper_limit_capping()

    def _apply_upper_limit_capping(self) -> None:
        """Apply upper limit capping to values with overflow redistribution.

        This method ensures that no value exceeds the specified `upper_limit`.
        When a value exceeds the limit, it is capped and the excess is
        redistributed to the next time interval, scaled by the ratio of
        the time interval areas (lengths).

        The algorithm works as follows:
        1. For each time interval, calculate its length:
           - First interval: 2 * (t[1] - t[0])
           - Middle intervals: (t[i+1] - t[i-1]) / 2
           - Last interval: 2 * (t[n-1] - t[n-2])

        2. If a value exceeds `upper_limit`:
           - Cap it to `upper_limit`
           - Calculate the excess: (original_value - upper_limit)
           - Redistribute to next interval: excess * area[i] / area[i+1]

        3. For the last value, simply cap it if it exceeds the limit.

        This approach conserves the total "flow" while respecting the
        upper limit constraint at each time point.
        """
        n = len(self._values)
        if n == 0:
            return

        if self._upper_limit is None:
            raise Exception(
                "This method should not be called if upper_limit is None."
            )
        upper_limit = float(self._upper_limit)

        if np.any(np.isnan(self._values)) or np.any(np.isinf(self._values)):
            raise ValueError(
                "values must not contain NaN or infinite values when "
                "upper_limit capping is applied"
            )

        if n == 1:
            self._values = np.minimum(self._values, upper_limit)
            return

        time_diffs = (self._time_array[1:] - self._time_array[:-1]).astype(
            np.float64
        )

        # Calculate the area (length) of each time interval
        areas = np.zeros(n, dtype=np.float64)

        if n == 2:
            # Two points: both use the full interval
            areas[0] = 2.0 * time_diffs[0]
            areas[1] = 2.0 * time_diffs[0]
        else:
            # First interval: 2 * (t[1] - t[0])
            areas[0] = 2.0 * time_diffs[0]

            areas[1:-1] = (time_diffs[1:] + time_diffs[:-1]) / 2.0

            # Last interval: 2 * (t[n-1] - t[n-2])
            areas[-1] = 2.0 * time_diffs[-1]

        # Apply capping with overflow redistribution
        for i in range(n - 1):
            if self._values[i] > upper_limit:
                excess = self._values[i] - upper_limit
                self._values[i] = upper_limit

                # Redistribute excess to the next interval, scaled by area
                # ratio
                redistribution = excess * areas[i] / areas[i + 1]
                self._values[i + 1] += redistribution

        # Cap the last value if needed (no redistribution possible)
        if self._values[-1] > upper_limit:
            self._values[-1] = upper_limit

    def serialize_data(self) -> dict[str, str | Real | list | dict]:
        """Serialize the data of this component."""
        dates = [
            d.replace(tzinfo=timezone.utc) for d in self._time_array.tolist()
        ]
        data = _FromDatasetRiverComponentData(
            time_list=dates,
            values=self._values.tolist(),
            max_error=self._max_error,
            upper_limit=self._upper_limit,
        )
        return data.model_dump(mode="json")

    @classmethod
    def deserialize_data(
        cls, data: dict[str, str | Real | list | dict]
    ) -> "FromDatasetRiverComponent":
        """Deserialize the data of this component."""
        data = _FromDatasetRiverComponentData(**data)
        return FromDatasetRiverComponent(
            time_list=data.time_list,
            values=np.asarray(data.values),
            max_error=data.max_error,
            upper_limit=data.upper_limit,
        )

    def __call__(self, time_list: Sequence[datetime]) -> np.ndarray:
        """Returns the values of this component."""
        requested_times = np.array(
            [datetime_to_datetime64(t, "ms") for t in time_list],
            dtype="datetime64[ms]",
        )
        values_indices = search_closest_sorted(
            self._time_array, requested_times
        )
        output = self._values[values_indices]

        if self._max_error is not None:
            time_diff = np.abs(
                self._time_array[values_indices] - requested_times
            )
            max_error = np.timedelta64(int(self._max_error.total_seconds() * 1000), "ms")
            if np.any(time_diff > max_error):
                t_index = np.where(time_diff > max_error)[0][0]
                t = time_list[t_index]
                t0 = np.datetime64(
                    self._time_array[values_indices][t_index], "s"
                ).astype(datetime)
                raise ValueError(
                    f"The nearest value for time {t} refers to {t0} and "
                    f"it is greater than the maximum allowed error: "
                    f"{self._max_error}"
                )

        return output


class _ClimatologicalRiverComponentData(BaseModel):
    values: list[float]
    resolution: timedelta


class ClimatologicalRiverComponent(RiverComponent):
    """A RiverComponent that repeats its values along the year.

    A climatological river component divides the year into a fixed number of
    equally spaced intervals (e.g., 1 day).

    It assigns a value to each date based on the corresponding day of
    the year.
    """

    def __init__(
        self,
        values: Sequence[float],
        resolution: timedelta = timedelta(days=1),
    ):
        self._resolution = int(resolution.total_seconds())

        # Here there is a 366 because we consider also leap years
        seconds_per_year = 366 * 24 * 60 * 60

        if seconds_per_year % self._resolution != 0:
            raise ValueError(
                "Invalid resolution: it must divide evenly a single year"
            )

        n_of_elements = seconds_per_year // self._resolution

        if len(values) != n_of_elements:
            raise ValueError(
                f'There are {len(values)} elements inside "values", while to '
                f"fill an annual climatology with the requested resolution "
                f"({resolution}), {n_of_elements} values are needed"
            )
        self._values = np.asarray(values)

    def __call__(self, time_list: Sequence[datetime]) -> np.ndarray:
        """Returns the values of this component."""
        output_values = np.empty((len(time_list),), dtype=self._values.dtype)
        # We move al the datetime in the reference year, that it is a leap year
        reference_year = datetime(year=2020, month=1, day=1)
        for i, t in enumerate(time_list):
            reference_t = datetime(
                year=reference_year.year,
                month=t.month,
                day=t.day,
                hour=t.hour,
                minute=t.minute,
                second=t.second,
            )
            seconds_of_the_year = (
                reference_t - reference_year
            ).total_seconds()

            value_index = int(seconds_of_the_year / self._resolution)
            output_values[i] = self._values[value_index]

        return output_values

    def serialize_data(self) -> dict[str, str | Real | list | dict]:
        """Serialize the data of this component."""
        data = _ClimatologicalRiverComponentData(
            values=self._values.tolist(),
            resolution=timedelta(seconds=self._resolution),
        )
        return json.loads(data.model_dump_json())

    @classmethod
    def deserialize_data(
        cls, data: dict[str, str | Real | list | dict]
    ) -> "ClimatologicalRiverComponent":
        """Deserialize the data of this component."""
        data = _ClimatologicalRiverComponentData.model_validate_json(
            json.dumps(data)
        )
        return ClimatologicalRiverComponent(data.values, data.resolution)


class AggregatedRiverComponent(RiverComponent, ABC):
    """A RiverComponent that combines multiple components.

    This abstract class provides functionality to create river components that
    combine the values of multiple other components using a specific
    aggregation function (e.g., sum, product). The aggregation function is
    defined by implementing the `get_aggregator` method.

    Args:
        components: A sequence of RiverComponent objects whose values will be
            combined.

    Examples:
        The SummedRiverComponent is a concrete implementation that sums values:
            >>> component1 = FixedRiverComponent(value=5.0)
            >>> component2 = FixedRiverComponent(value=3.0)
            >>> summed = SummedRiverComponent([component1, component2])
            >>> times = [datetime(2025, 1, 1)]
            >>> summed(times)
            array([8.0])
    """

    def __init__(self, components: Sequence[RiverComponent]):
        self._components = components
        self._aggregator = self.get_aggregator()

    @classmethod
    @abstractmethod
    def get_aggregator(cls) -> Callable[[Sequence[np.ndarray]], np.ndarray]:
        """Returns the aggregation function to combine component values.

        This abstract method must be implemented by subclasses to define how
        the values from multiple river components should be combined.
        The returned function should take multiple numpy arrays as input
        (one for each component) and return a single numpy array containing
        the aggregated values.

        Returns:
            A callable that takes multiple numpy arrays as input and returns a
            single numpy array containing the aggregated values. The output
            array should have the same shape as each input array.
        """
        raise NotImplementedError

    def serialize_data(self) -> dict[str, str | Real | list | dict]:
        """Serialize the data of this component."""
        components = []
        for component in self._components:
            components.append(component.serialize())
        return {"components": components}

    @classmethod
    def deserialize_data(
        cls, data: dict[str, str | Real | list | dict]
    ) -> "RiverComponent":
        """Deserialize the data of this component."""
        components = []
        for component in cast(list[dict], data["components"]):
            components.append(RiverComponent.deserialize(component))
        return cls(components)

    def __call__(self, time_list: Sequence[datetime]) -> np.ndarray:
        """Return the values of a river for the selected times."""
        if len(time_list) == 0:
            return np.empty((0,), dtype=np.float32)
        values = tuple(c(time_list) for c in self._components)
        return self._aggregator(values)


class PairwiseAggregatedRiverComponent(AggregatedRiverComponent):
    """A RiverComponent that combines components using pairwise operations.

    This abstract class provides functionality to create river components that
    combine values from multiple components using a pairwise operation (e.g.,
    addition, multiplication). The pairwise operation is defined by
    implementing the `get_pairwise_aggregator` method.

    The aggregation is performed by applying the pairwise operation
    sequentially to pairs of arrays, starting with the first two arrays and
    then combining the result with each subsequent array.

    Examples:
        SummedRiverComponent uses pairwise addition to sum values:
            >>> component1 = FixedRiverComponent(value=5.0)
            >>> component2 = FixedRiverComponent(value=3.0)
            >>> component3 = FixedRiverComponent(value=2.0)
            >>> summed = SummedRiverComponent(
            ...     [component1, component2, component3]
            ... )
            >>> times = [datetime(2025, 1, 1)]
            >>> summed(times)  # Computes ((5 + 3) + 2)
            array([10.0])
    """

    @classmethod
    @abstractmethod
    def get_pairwise_aggregator(
        cls,
    ) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        """Returns the pairwise operation function for combining arrays.

        This abstract method must be implemented by subclasses to define how
        pairs of arrays should be combined. The returned function should take
        two numpy arrays as input and return a single numpy array containing
        the result of applying the pairwise operation.

        Returns:
            A callable that takes two numpy arrays as input and returns a
            single numpy array containing the result of the pairwise operation.
            The output array should have the same shape as each input array.
        """
        raise NotImplementedError

    @classmethod
    def get_aggregator(cls) -> Callable[[Sequence[np.ndarray]], np.ndarray]:
        """Returns an aggregation function that uses the pairwise operation.

        This method creates an aggregation function that combines multiple
        arrays by repeatedly applying the pairwise operation defined by
        get_pairwise_aggregator(). The aggregation is performed sequentially:
        first combining arrays[0] and arrays[1], then combining the result
        with arrays[2], and so on.

        Returns:
            A callable that takes a sequence of numpy arrays as input and
            returns a single numpy array containing the result of applying
            the pairwise operation sequentially to all input arrays.

        Raises:
            ValueError: If an empty sequence of arrays is provided.
        """
        pairwise_f = cls.get_pairwise_aggregator()

        def aggregation(arrays: Sequence[np.ndarray]):
            if len(arrays) == 0:
                raise ValueError("No arrays provided")
            if len(arrays) == 1:
                return arrays[0]
            v_temp = arrays[0]
            for v in arrays[1:]:
                v_temp = pairwise_f(v_temp, v)
            return v_temp

        return aggregation


class SummedRiverComponent(PairwiseAggregatedRiverComponent):
    """Sum of a sequence of river components.

    This class represents a river component that combines multiple components
    by summing their values element-wise. It implements pairwise aggregation
    using addition, so components are combined by adding their values two at
    a time.

    Examples:
        Sum three components sequentially:
            >>> component1 = FixedRiverComponent(value=5.0)
            >>> component2 = FixedRiverComponent(value=3.0)
            >>> component3 = FixedRiverComponent(value=2.0)
            >>> summed = SummedRiverComponent(
            ...     [component1, component2, component3]
            ... )
            >>> times = [datetime(2025, 1, 1)]
            >>> summed(times)  # Computes ((5 + 3) + 2)
            array([10.0])
    """

    @classmethod
    def get_pairwise_aggregator(
        cls,
    ) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        """Returns the NumPy add function for combining component values.

        This method returns the `np.add` function, which combines multiple
        arrays by summing them element-wise.

        Returns:
            The NumPy add function (`np.add`) which takes multiple arrays as
            input and returns their element-wise sum.
        """
        return np.add


class MultipliedRiverComponent(PairwiseAggregatedRiverComponent):
    """Product of a sequence of river components.

    This class represents a river component that combines multiple components
    by multiplying their values element-wise. It implements pairwise
    aggregation using multiplication, so components are combined by multiplying
    their values two at a time.

    Examples:
        Multiply three components sequentially:
            >>> component1 = FixedRiverComponent(value=5.0)
            >>> component2 = FixedRiverComponent(value=3.0)
            >>> component3 = FixedRiverComponent(value=2.0)
            >>> multiplied = MultipliedRiverComponent(
            ...     [component1, component2, component3]
            ... )
            >>> times = [datetime(2025, 1, 1)]
            >>> multiplied(times)  # Computes ((5 * 3) * 2)
            array([30.0])
    """

    @classmethod
    def get_pairwise_aggregator(
        cls,
    ) -> Callable[[Sequence[np.ndarray]], np.ndarray]:
        """Retrieves the NumPy multiply function.

        This method returns the `np.multiply` function, which computes the
        element-wise multiplication of input arrays.

        Returns:
            The NumPy multiply function (`np.multiply`) which takes multiple
            arrays as input and returns their element-wise product.
        """
        return np.multiply


class RatioRiverComponent(PairwiseAggregatedRiverComponent):
    """Division of one river component by another.

    This class represents a river component that computes the ratio between two
    components by performing element-wise division of their values. The first
    component serves as the numerator and the second as the denominator.

    Args:
        numerator: The RiverComponent whose values will be used as dividends
        denominator: The RiverComponent whose values will be used as divisors

    Examples:
        Divide two fixed components:
            >>> component1 = FixedRiverComponent(value=6.0)
            >>> component2 = FixedRiverComponent(value=2.0)
            >>> ratio = RatioRiverComponent(component1, component2)
            >>> times = [datetime(2025, 1, 1)]
            >>> ratio(times)
            array([3.0])
    """

    def __init__(self, numerator: RiverComponent, denominator: RiverComponent):
        super().__init__([numerator, denominator])

    @classmethod
    def get_pairwise_aggregator(
        cls,
    ) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        """Retrieves the NumPy divide function.

        This method returns the `np.divide` function, which computes the
        element-wise division between two arrays. When called with two arrays
        as arguments, it produces an array where each element is the ratio of
        the corresponding elements in the input arrays.

        Returns:
            The NumPy divide function (`np.divide`) which takes two arrays as
            input and returns their element-wise quotient.
        """
        return np.divide


class DerivedRiverComponent(RiverComponent, ABC):
    """A RiverComponent that applies a transformation to another component.

    This abstract base class provides functionality to create river components
    that derive their values by applying a mathematical function to the values
    of another river component. The transformation is defined by implementing
    the `get_function` method which returns a callable that takes a Numpy
    array and returns a Numpy array.

    Args:
        component: The source RiverComponent whose values will be transformed.

    Examples:
        The OppositeRiverComponent is a concrete implementation that negates
        values:
            >>> component = FixedRiverComponent(value=5.0)
            >>> opposite = OppositeRiverComponent(component)
            >>> times = [datetime(2025, 1, 1)]
            >>> opposite(times)
            array([-5.0])
    """

    def __init__(self, component: RiverComponent):
        self._component = component
        self._func = self.get_function()

    @classmethod
    @abstractmethod
    def get_function(cls) -> Callable[[np.ndarray], np.ndarray]:
        """Returns the function to transform river component values.

        This abstract method must be implemented by subclasses to define the
        specific mathematical transformation that will be applied to the values
        of the source river component.

        Returns:
            A callable that takes a numpy array as input and returns a Numpy
            array containing the transformed values. The output array should
            have the same shape as the input array.
        """
        raise NotImplementedError

    def serialize_data(self):
        """Serialize the data of this component."""
        return {"original_component": self._component.serialize()}

    @classmethod
    def deserialize_data(cls, data):
        """Deserialize the data of this component."""
        original_component = RiverComponent.deserialize(
            data["original_component"]
        )
        return cls(original_component)

    def __call__(self, time_list: Sequence[datetime]) -> np.ndarray:
        """Return the values of a river for the selected times."""
        if len(time_list) == 0:
            return np.empty((0,), dtype=np.float32)
        return self._func(self._component(time_list))


class OppositeRiverComponent(DerivedRiverComponent):
    """Opposite of a river component."""

    @classmethod
    def get_function(cls) -> Callable[[np.ndarray], np.ndarray]:
        """Retrieves the NumPy negative function.

        This method returns the `np.negative` function, which computes the
        element-wise numerical negation of the given input (given `x`, it
        returns `-x`).

        Returns:
            The NumPy negative function (`np.negative`).
        """
        return np.negative


class ReciprocalRiverComponent(DerivedRiverComponent):
    """Reciprocal of a river component."""

    @classmethod
    def get_function(cls) -> Callable[[np.ndarray], np.ndarray]:
        """Retrieves the NumPy reciprocal function.

        This method returns the `np.reciprocal` function, which computes the
        element-wise reciprocal of the given input (given `x`, it returns
        `1/x`).

        Returns:
            The NumPy reciprocal function (`np.reciprocal`).
        """
        return np.reciprocal


class AdditiveComponent(RiverComponent):
    """River data plus a constant.

    A `RiverComponent` that returns the sum of the values of another
    component with a constant.
    """

    def __init__(self, original_component: RiverComponent, add_factor: Real):
        self._component = original_component
        self._add_factor = add_factor

    def serialize_data(self) -> dict[str, str | Real | list | dict]:
        """Serialize the data of this component."""
        return {
            "original_component": self._component.serialize(),
            "add_factor": self._add_factor,
        }

    @classmethod
    def deserialize_data(
        cls, data: dict[str, str | Real | list | dict]
    ) -> "AdditiveComponent":
        """Deserialize the data of this component."""
        original_component = RiverComponent.deserialize(
            data["original_component"]
        )
        add_factor = data["add_factor"]
        return cls(
            original_component=original_component, add_factor=add_factor
        )

    def __call__(self, time_list: Sequence[datetime]) -> np.ndarray:
        """Return the values of a river for the selected times."""
        return self._component.__call__(time_list) + self._add_factor


class MultiplicativeComponent(RiverComponent):
    """River data times a constant.

    A `RiverComponent` that returns the same values of another component
    multiplied by a constant.
    """

    def __init__(self, original_component: RiverComponent, mul_factor: Real):
        self._component = original_component
        self._mul_factor = mul_factor

    def serialize_data(self) -> dict[str, str | Real | list | dict]:
        """Serialize the data of this component."""
        return {
            "original_component": self._component.serialize(),
            "mul_factor": self._mul_factor,
        }

    @classmethod
    def deserialize_data(
        cls, data: dict[str, str | Real | list | dict]
    ) -> "MultiplicativeComponent":
        """Deserialize the data of this component."""
        original_component = RiverComponent.deserialize(
            data["original_component"]
        )
        mul_factor = data["mul_factor"]
        return cls(
            original_component=original_component, mul_factor=mul_factor
        )

    def __call__(self, time_list: Sequence[datetime]) -> np.ndarray:
        """Return the values of a river for the selected times."""
        return self._component.__call__(time_list) * self._mul_factor


class RiverPhysicalModel(ABC):
    """Physical model of a river.

    A `RiverPhysicalModel` contains all the information required by the
    physical model for a specific river.
    """

    @classmethod
    @abstractmethod
    def get_discharge_type(cls) -> type[Flux] | type[np.ndarray]:
        """Returns the type of the object returned by get_discharge.

        The discharge of a river can be represented in two different ways:
        - it can be a flux, if the river is modeled as a stream
        - it can be a sequence of values, if the river is modeled as a "source
          of water".

        In the first case, we also have a direction for the discharge, while in
        the second case we only have the amount of water that the river
        discharges.

        This method returns the type of the discharge that this model provides.

        Returns:
            The type of the discharge that this model provides.
        """
        raise NotImplementedError

    @abstractmethod
    def get_discharge(
        self, time_list: Sequence[datetime]
    ) -> Flux | np.ndarray:
        """Discharge of the river."""
        raise NotImplementedError

    @abstractmethod
    def get_temperature(self, time_list: Sequence[datetime]) -> np.ndarray:
        """Temperature of the river."""
        raise NotImplementedError

    @abstractmethod
    def get_salinity(self, time_list: Sequence[datetime]) -> np.ndarray:
        """Salinity of the river."""
        raise NotImplementedError

    @abstractmethod
    def serialize(self) -> dict:
        """Convert this RiverPhysicalModel instance to a dictionary format.

        This method should convert the model instance into a dictionary that
        can be easily serialized to JSON for storage or transmission.
        The dictionary must include a 'class' key with the name of the
        concrete implementation class, allowing proper deserialization later.

        Returns:
            A dictionary containing the serialized data of this model instance.
        """
        raise NotImplementedError

    @staticmethod
    def deserialize(data: dict) -> "RiverPhysicalModel":
        """Create a RiverPhysicalModel instance from serialized data.

        This method acts as a factory, determining the correct subclass to use
        based on the 'class' field in the data dictionary, then delegating the
        actual deserialization to that subclass's from_dict() method.

        Args:
            data: A dictionary containing the serialized model data, including
                a 'class' key identifying the specific implementation class.

        Returns:
            A new instance of the appropriate RiverPhysicalModel subclass.
        """
        cls_name = data["class"]
        subclasses = {
            cls.__name__: cls for cls in RiverPhysicalModel.__subclasses__()
        }
        cls = subclasses[cls_name]
        return cls.from_dict(data)

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict) -> "RiverPhysicalModel":
        """Create an instance of this specific model class from a dictionary.

        This method is called by deserialize() after determining the correct
        subclass to use. Each subclass must implement this method to handle
        its specific deserialization logic.

        Args:
            data: A dictionary containing the serialized model data specific
                to this implementation class.

        Returns:
            A new instance of this RiverPhysicalModel subclass.
        """
        raise NotImplementedError


class IndependentDischargeRiverPhysicalModel(RiverPhysicalModel):
    """Independent discharge river physical model.

    In an IndependentDischargeRiverPhysicalModel, the discharge, the
    temperature, and the salinity are independent of each other. Therefore,
    each one of them can be computed by a different RiverComponent.

    The discharge, which is a `Flux`, is obtained by combining the amount of
    water produced by the component of the discharge with the assigned side.
    """

    def __init__(
        self,
        side: Side,
        discharge: RiverComponent,
        temperature: RiverComponent,
        salinity: RiverComponent,
    ):
        self.side: Side = side
        self.discharge: RiverComponent = discharge
        self.temperature: RiverComponent = temperature
        self.salinity: RiverComponent = salinity

    @classmethod
    def get_discharge_type(cls) -> type[Flux] | type[np.ndarray]:
        """Returns the type of the object returned by get_discharge."""
        return Flux

    def get_discharge(self, time_list: Sequence[datetime]) -> Flux:
        """Discharge of the river."""
        return Flux.from_side(side=self.side, values=self.discharge(time_list))

    def get_temperature(self, time_list: Sequence[datetime]) -> np.ndarray:
        """Temperature of the river."""
        return self.temperature(time_list)

    def get_salinity(self, time_list: Sequence[datetime]) -> np.ndarray:
        """Salinity of the river."""
        return self.salinity(time_list)

    def serialize(self):
        """Convert this RiverPhysicalModel instance to a dictionary format."""
        return {
            "class": self.__class__.__name__,
            "side": self.side.value,
            "discharge": self.discharge.serialize(),
            "temperature": self.temperature.serialize(),
            "salinity": self.salinity.serialize(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RiverPhysicalModel":
        """Create a RiverPhysicalModel instance from serialized data."""
        side = Side(data["side"])
        discharge = RiverComponent.deserialize(data["discharge"])
        temperature = RiverComponent.deserialize(data["temperature"])
        salinity = RiverComponent.deserialize(data["salinity"])
        return cls(
            side=side,
            discharge=discharge,
            temperature=temperature,
            salinity=salinity,
        )


class SeparateComponentsRiverPhysicalModel(RiverPhysicalModel):
    """Separate component physical model.

    A RiverPhysicalModel where each variable is independent and, therefore,
    can be computed by a different RiverComponent.

    This class is really similar to IndependentDischargeRiverPhysicalModel; the
    main difference is that here the discharge is not a `Flux`, but just
    a numpy array (and, therefore, it is not necessary to provide a side).
    """

    def __init__(
        self,
        discharge: RiverComponent,
        temperature: RiverComponent,
        salinity: RiverComponent,
    ):
        self.discharge: RiverComponent = discharge
        self.temperature: RiverComponent = temperature
        self.salinity: RiverComponent = salinity

    @classmethod
    def get_discharge_type(cls) -> type[Flux] | type[np.ndarray]:
        """Returns the type of the object returned by get_discharge."""
        return np.ndarray

    def get_discharge(self, time_list: Sequence[datetime]) -> np.ndarray:
        """Discharge of the river."""
        return self.discharge(time_list)

    def get_temperature(self, time_list: Sequence[datetime]) -> np.ndarray:
        """Temperature of the river."""
        return self.temperature(time_list)

    def get_salinity(self, time_list: Sequence[datetime]) -> np.ndarray:
        """Salinity of the river."""
        return self.salinity(time_list)

    def serialize(self):
        """Convert this RiverPhysicalModel instance to a dictionary format."""
        return {
            "class": self.__class__.__name__,
            "discharge": self.discharge.serialize(),
            "temperature": self.temperature.serialize(),
            "salinity": self.salinity.serialize(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RiverPhysicalModel":
        """Create a RiverPhysicalModel instance from serialized data."""
        discharge = RiverComponent.deserialize(data["discharge"])
        temperature = RiverComponent.deserialize(data["temperature"])
        salinity = RiverComponent.deserialize(data["salinity"])
        return cls(
            discharge=discharge,
            temperature=temperature,
            salinity=salinity,
        )


class ClimatologicalBioChemicalModel(BaseModel):
    """Biochemical model of the river.

    A `RiverBioChemicalModel` stores all the concentrations for the
    biogeochemical variables of the model.
    """

    values: dict[str, float]

    def get_variables(self) -> tuple[str, ...]:
        """Names of all the variables available for this river.

        Return a tuple containing the names of all the variables stored by
        this model.
        """
        return tuple(sorted(self.values.keys()))

    def get_concentration(
        self, var_name: str, time_list: Sequence[datetime]
    ) -> np.ndarray:
        """Concentration of a specific variable at given times."""
        return np.full((len(time_list),), self.values[var_name])


class River:
    def __init__(
        self,
        river_name: str,
        river_geometry: RiverGeometry,
        physical_model: RiverPhysicalModel,
        biochemical_model: ClimatologicalBioChemicalModel,
    ):
        self._name = river_name
        self._geometry = river_geometry
        self._physical_model = physical_model
        self._biochemical_model = biochemical_model

    @property
    def name(self) -> str:
        """Name of the river."""
        return self._name

    @property
    def geometry(self) -> RiverGeometry:
        """Geometry of the river."""
        return self._geometry

    @property
    def side(self) -> Side:
        """Side of the river."""
        return self.geometry.side

    def get_physical_model(self):
        return self._physical_model

    def get_biochemical_model(self):
        return self._biochemical_model

    def get_discharge_type(self) -> type[Flux] | type[np.ndarray]:
        """Returns the type of the object returned by get_discharge."""
        return self._physical_model.get_discharge_type()

    def get_discharge(
        self, time_list: Sequence[datetime]
    ) -> Flux | np.ndarray:
        """Get the discharge of the river for selected times."""
        return self._physical_model.get_discharge(time_list=time_list)

    def get_variable(self, var_name, time_list) -> np.ndarray:
        """Get river data for a given variable at given times."""
        if var_name.lower() == "temperature" or var_name == "T":
            return self._physical_model.get_temperature(time_list=time_list)
        if var_name.lower() == "salinity" or var_name == "S":
            return self._physical_model.get_salinity(time_list=time_list)
        return self._biochemical_model.get_concentration(
            var_name=var_name, time_list=time_list
        )

    def serialize(self) -> dict:
        """Serialize the River object to a dictionary format.

        This method converts the River object and all its components (name,
        geometry, physical model, and biochemical model) into a dictionary
        that can be easily converted to JSON for storage or transmission.

        Returns:
            dict: A dictionary containing the serialized data of the River
                object with the following keys:
                - name: The name of the river
                - geometry: Serialized RiverGeometry data
                - physical_model: Serialized physical model data
                - biochemical_model: Serialized biochemical model data
        """
        return {
            "name": self.name,
            "geometry": self.geometry.model_dump(mode="json"),
            "physical_model": self._physical_model.serialize(),
            "biochemical_model": self._biochemical_model.model_dump(),
        }

    @classmethod
    def deserialize(cls, data: dict) -> "River":
        """Create a River object from serialized data.

        This class method reconstructs a River object from a dictionary
        containing serialized data, typically created by the serialize()
        method. It validates and converts each component of the data back
        into the appropriate object types.

        Args:
            data: A dictionary containing the serialized River data with
                the following required keys:
                - name: The name of the river
                - geometry: Serialized RiverGeometry data
                - physical_model: Serialized physical model data
                - biochemical_model: Serialized biochemical model data

        Returns:
            River: A new River instance constructed from the deserialized
                data.
        """
        river_name = data["name"]
        river_geometry = RiverGeometry.model_validate(data["geometry"])
        physical_model = RiverPhysicalModel.deserialize(data["physical_model"])
        biochemical_model = ClimatologicalBioChemicalModel.model_validate(
            data["biochemical_model"]
        )
        return cls(
            river_name=river_name,
            river_geometry=river_geometry,
            physical_model=physical_model,
            biochemical_model=biochemical_model,
        )
