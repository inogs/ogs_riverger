import json
import logging
from collections import OrderedDict
from collections.abc import Callable
from collections.abc import Collection
from collections.abc import Iterable
from datetime import datetime
from typing import Any

import numpy as np
import xarray as xr
from bitsea.utilities.bc import Side

from ogs_riverger.models.river import ClimatologicalBioChemicalModel
from ogs_riverger.models.river import ClimatologicalRiverComponent
from ogs_riverger.models.river import FixedRiverComponent
from ogs_riverger.models.river import FromDatasetRiverComponent
from ogs_riverger.models.river import IndependentDischargeRiverPhysicalModel
from ogs_riverger.models.river import River
from ogs_riverger.models.river import RiverComponent
from ogs_riverger.models.river import RiverGeometry
from ogs_riverger.models.river import SeparateComponentsRiverPhysicalModel
from ogs_riverger.models.river import SinusoidalRiverComponent
from ogs_riverger.read_config import RiverConfig

logger = logging.getLogger(__name__)


class RiverCollection(Collection[River]):
    """Collection of rivers.

    It behaves like a list, maintaining insertion order and supporting
    iteration, but it also allows retrieval of a river by name using the
    square bracket operator.

    This object also ensures uniqueness: two rivers with the same name cannot
    be stored within it
    """

    def __init__(self, rivers: Iterable[River] | None = None):
        if rivers is None:
            rivers = []

        rivers = tuple(rivers)
        names = tuple(r.name for r in rivers)
        # Check if there are different rivers with the same name and raise
        # a ValueError if this is the case
        if len(set(names)) != len(names):
            for name in names:
                if names.count(name) != 1:
                    raise ValueError(
                        f'There are multiple rivers with the name "{name}"'
                    )

        self._data = OrderedDict(tuple((r.name, r) for r in rivers))

    def add(self, river):
        """Add a new river to the collection."""
        if river.name in self._data:
            raise ValueError(
                'A river named "{}" already exists in this collection'.format(
                    river.name
                )
            )
        self._data[river.name] = river

    def __getitem__(self, river_name: str) -> River:
        return self._data[river_name]

    def __iter__(self):
        return iter(self._data.values())

    def __len__(self):
        return len(self._data)

    def __eq__(self, other):
        if not isinstance(other, RiverCollection):
            return NotImplemented
        return self._data == other._data

    def __contains__(self, river_name: Any) -> bool:
        if isinstance(river_name, River):
            raise ValueError(
                "To check if a river is in the collection, use its name"
            )
        return river_name in self._data

    def on_side(self, side: Side | str):
        """RiverCollection containing only the rivers on one side."""
        if isinstance(side, str):
            side = Side(side)

        return RiverCollection(r for r in self if r.side == side)

    def as_dict(self) -> dict[str, River]:
        """Dictionary representation of the River Collection.

        Return the content of this object as a dictionary, where each
        river's name is associated with its corresponding object.
        """
        return {name: river for name, river in self._data.items()}

    def as_tuple(self) -> tuple[River, ...]:
        """Return a tuple with all the rivers stored inside this object."""
        return tuple(river for river in self._data.values())

    def serialize(self) -> tuple[dict[str, Any], ...]:
        """Serialize RiverCollection into a tuple of dictionaries.

        Creates a serialized representation of the RiverCollection by
        converting each River object into its dictionary representation.
        The resulting tuple can be used for persistence or data transfer.

        Returns:
            A tuple containing dictionaries, where each dictionary represents
            the serialized form of a River object in the collection.
        """
        return tuple(river.serialize() for river in self)

    def to_json(self, **kwargs) -> str:
        """Convert RiverCollection to JSON string representation.

        Serializes the entire RiverCollection into a JSON string format,
        making it suitable for storage or transmission. Each river in the
        collection is first serialized to a dictionary and then converted
        to JSON format.

        Returns:
            A JSON string containing the serialized representation of all
            rivers in the collection.
        """
        return json.dumps(self.serialize(), **kwargs)

    @classmethod
    def deserialize(
        cls, data: tuple[dict[str, Any], ...]
    ) -> "RiverCollection":
        """Recreate RiverCollection from serialized data.

        This is the inverse operation of serialize(). It takes a tuple of
        dictionaries (each representing a serialized River object) and
        reconstructs a RiverCollection instance containing all the rivers.

        Args:
            data: A tuple of dictionaries, where each dictionary contains
                the serialized representation of a River object as produced
                by River.serialize().

        Returns:
            A new RiverCollection instance containing all the deserialized
            River objects.
        """
        return cls(River.deserialize(river_data) for river_data in data)

    @classmethod
    def from_json(cls, json_data: str) -> "RiverCollection":
        """Create RiverCollection from JSON string.

        This is the inverse operation of to_json(). It takes a JSON string
        (typically created by to_json()) and reconstructs a RiverCollection
        instance. The method first parses the JSON string into Python data
        structures and then uses deserialize() to create the actual
        RiverCollection.

        Args:
            json_data: A JSON string containing the serialized representation
                of a RiverCollection, typically created by to_json().

        Returns:
            A new RiverCollection instance containing all the rivers
            described in the JSON data.
        """
        return cls.deserialize(json.loads(json_data))

    def __repr__(self):
        return "RiverCollection{}".format(tuple(k for k in self._data.keys()))

    @staticmethod
    def _build_mer_river_collection(
        river_config: RiverConfig,
        get_discharge: Callable[[int], RiverComponent],
        river_positions: dict,
    ) -> "RiverCollection":
        """RiverCollection with configured river models.

        This internal method serves as a factory function for creating
        RiverCollection instances. It provides a flexible interface that allows
        different discharge calculation strategies while keeping the rest of
        the river configuration process consistent.

        Each river in the configuration is processed to create a complete river
        model that includes physical properties (discharge, temperature,
        salinity) and biochemical characteristics.

        Args:
            river_config (RiverConfig): Configuration object containing
                comprehensive river specifications including names, IDs, model
                types, geometries, and biophysical parameters.
            get_discharge: Function that accepts a river ID (integer) and
                returns a RiverComponent representing the river's discharge.
                This allows for different discharge calculation strategies
                (e.g., from real data or climatological averages).

        Returns:
            A RiverCollection of rivers with defined configurations based
            on the input parameters.

        """
        river_collection = RiverCollection()
        for river in river_config:
            logger.debug("Reading river %s (id %s)", river.name, river.id)
            river_active = True
            if hasattr(river, "active"):
                river_active = river.active
            if not river_active:
                logger.debug(
                    "River %s (id %s) is inactive and will not be added to "
                    "the RiverCollection",
                    river.name,
                    river.id,
                )
                continue

            logger.debug("River model is %s", river.model)

            river_side = river.geometry.side
            latitude = river.geometry.mouth_latitude
            longitude = river.geometry.mouth_longitude
            logger.debug(
                "Mouth of %s (id %s) has coordinates (%s, %s) and it is on "
                "side %s",
                river.name,
                river.id,
                latitude,
                longitude,
                river_side,
            )

            for river_pos in river_positions:
                if river_pos["name"] == river.name:
                    lat_indices = river_pos["latitude_indices"]
                    lon_indices = river_pos["longitude_indices"]
                    river_side = Side(river_pos["side"])
                    break
            else:
                raise ValueError(
                    f"No river named {river.name} in river_positions "
                    f"dictionary"
                )

            logger.debug(
                "%s (id %s) is located on coordinates lat = %s, lon = %s",
                river.name,
                river.id,
                lat_indices,
                lon_indices,
            )

            river_geometry = RiverGeometry(
                mouth_lat=latitude,
                mouth_lon=longitude,
                lat_indices=lat_indices,
                lon_indices=lon_indices,
                side=river_side,
            )

            discharge = get_discharge(river.id)
            if river.runoff_factor is not None:
                normalized_discharge = discharge * river.runoff_factor
            else:
                normalized_discharge = discharge

            # Prepare two components for the temperature and the salinity
            logger.debug("Reading temperature of %s", river.name)
            temperature_variation = river.physical.temperature.variation
            temperature_average = river.physical.temperature.average
            temperature_modulation = (
                temperature_variation / temperature_average
            )
            temperature = SinusoidalRiverComponent(
                average=temperature_average,
                modulation=temperature_modulation,
                peak=212,  # Max temperature on 31 July
            )
            logger.debug("Reading salinity of %s", river.name)
            salinity = FixedRiverComponent(value=river.physical.salinity.value)

            if river.model == "rain_like":
                physical_model = SeparateComponentsRiverPhysicalModel(
                    discharge=normalized_discharge,
                    temperature=temperature,
                    salinity=salinity,
                )
            else:
                physical_model = IndependentDischargeRiverPhysicalModel(
                    side=river_side,
                    discharge=normalized_discharge,
                    temperature=temperature,
                    salinity=salinity,
                )
            biochemical_model = ClimatologicalBioChemicalModel(
                values=river.biogeochemical
            )

            logger.debug(
                "Building a model for river %s (id %s)", river.name, river.id
            )
            river = River(
                river_name=river.name,
                river_geometry=river_geometry,
                physical_model=physical_model,
                biochemical_model=biochemical_model,
            )

            river_collection.add(river)
        return river_collection

    @staticmethod
    def build_mer_river_collection(
        river_config: RiverConfig,
        discharge_data: xr.Dataset,
        river_positions: dict,
    ) -> "RiverCollection":
        """Build a river collection based on river configuration and discharge.

        This function processes river configuration data and discharge
        timeseries provided as input. It reads and interprets biogeochemical
        variable names, source coordinates, discharge data, and other river
        attributes to create a collection of river models with defined
        physical and biochemical properties. Each river model is constructed
        based on its specific attributes, including its geometry, physical
        discharge, temperature, salinity, and biochemical variables.

        Args:
            river_config: River configuration, including data and
                biogeochemical variable definitions, used to define the
                properties and sources of the rivers.
            discharge_data: An xarray dataset containing discharge
                timeseries data for the rivers, indexed by time and river ID.
                The dataset must contain a variable named "discharge" with 2
                dimensions: "time" and "id" of the river. The dimension
                "time" must have an associated coordinate. We expect to have
                daily values and the time values to refer to the first
                instant of the day.

        Returns:
            A RiverCollection of rivers with defined configurations based
                on the input parameters.

        Raises:
            ValueError: If the source coordinate format is invalid or cannot
                be parsed.
        """

        def get_discharge(river_id: int) -> RiverComponent:
            # Compute discharge and store it in RiverComponent.
            discharge_values = discharge_data.sel(id=river_id)
            # Convert the time values into seconds (numpy is able to convert
            # datetime64 to datetime only if we start from datetime64[s])
            times_start = discharge_values.time.values.astype("datetime64[s]")
            # The datetimes that we have refers to the beginning of the
            # temporal interval. We add 12 hours to move them to the center
            # (daily data)
            times_middle = times_start + np.timedelta64(12 * 3600, "s")
            discharge = FromDatasetRiverComponent(
                time_list=[t for t in times_middle.astype(datetime)],
                values=discharge_values.discharge.values,
            )
            return discharge

        return RiverCollection._build_mer_river_collection(
            river_config, get_discharge, river_positions=river_positions
        )

    @staticmethod
    def build_climatological_mer_river_collection(
        river_config: RiverConfig,
        efas_climatological_data: xr.Dataset,
        river_positions: dict,
    ) -> "RiverCollection":
        """River collection from configuration and climatological data.

        This method functions similarly to `build_mer_river_collection` but
        specializes in handling climatological discharge data. Usually, this
        function is used when the data about the real discharge of the rivers
        for the time window of the simulation is not available.

        Args:
            river_config: A configuration object containing comprehensive river
                data, including biogeochemical variable definitions and
                specifications for defining river properties and sources.
            efas_climatological_data: An xarray Dataset containing daily
                climatological measurements for each river. The dataset must
                include a 'discharge' variable with two dimensions: 'time' and
                'id'. The time dimension must contain precisely 366 values,
                representing each day of the year (including February 29th).

        Returns:
            A configured RiverCollection instance containing river objects
            initialized with the specified parameters and climatological
            discharge data.

        Raises:
            ValueError: If the provided source coordinates are invalid or
                cannot be properly parsed.
        """

        def get_discharge(river_id: int) -> RiverComponent:
            discharge_values = efas_climatological_data.sel(
                id=river_id
            ).discharge.values
            return ClimatologicalRiverComponent(discharge_values)

        return RiverCollection._build_mer_river_collection(
            river_config, get_discharge, river_positions=river_positions
        )
