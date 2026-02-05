import json
from collections import OrderedDict
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Annotated
from typing import Literal
from typing import TypeAlias
from typing import Union

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator
from pydantic import RootModel

from ogs_riverger.efas.efas_config import EFASConfigElement
from ogs_riverger.physical_models import PhysicalModelConfig


BGCProfile: TypeAlias = dict[str, float]

# Right now we have only one RiverDataSource
RiverDataSource = Annotated[
    Union[EFASConfigElement], Field(discriminator="type")
]


class RiverGeometryConfig(BaseModel):
    """
    Representation of a river's geometric configuration.

    This class allows encapsulating data needed to describe the geometry of a
    river, including its location, dimensions, and structural properties.
    It also provides validation logic to ensure the configuration is
    consistent.

    Attributes:
        mouth_latitude (float): The latitude of the river's mouth.
        mouth_longitude (float): The longitude of the river's mouth.
        width (float): The width of the river's mouth.
        depth (float): The depth of the river.
        side (Literal["N", "S", "E", "W"] | None): The orientation of the
            river's side, if specified.
        stem_length (float | None): The length of the river's stem,
            if provided.
        stem (list[str] | None): The list of string identifiers that represent
            the stem elements, if provided. This list is expected to be a
            sequence of strings like "+5z" or "-2m" which is "move by 5 cells
            following zonal direction" or "move by 2 cells following meridional
            direction (but backwards, because of the minus)"
    """

    mouth_latitude: float
    mouth_longitude: float
    width: float
    depth: float
    side: Literal["N", "S", "E", "W"] | None = None
    stem_length: float | None = None
    stem: list[str] | None = None

    @model_validator(mode="after")
    def check_stem_information(self):
        if self.stem is None and self.stem_length is None:
            raise ValueError(
                "At least one of stem or stem_length must be set."
            )
        if self.stem is not None and self.stem_length is not None:
            raise ValueError(
                "You can not configure at the same time the stem and the "
                "stem_length."
            )
        return self

    @staticmethod
    def merge_dicts(d1: dict, d2: dict) -> dict:
        """
        Merges the information from the domain-specific file.

        This function merges the information from the domain-specific file of
        a river into the dictionary that describes the river's geometry. If
        `d2` contains information about the stem, this overrides the
        information in `d1` even if the information is stored into two
        different fields (like `stem` and `stem_length`).

        Args:
            d1 (dict): The first dictionary to merge.
            d2 (dict): The second dictionary to merge, which may contain keys
                "stem" or "stem_length" leading to their removal in the copy
                of the first dictionary.

        Returns:
            A merged dictionary containing all key-value pairs from the second
            dictionary and the remaining key-value pairs from the original
            first dictionary.
        """
        d1 = d1.copy()
        if "stem" in d2 or "stem_length" in d2:
            if "stem" in d1:
                del d1["stem"]
            if "stem_length" in d1:
                del d1["stem_length"]
        d1.update(d2)
        return d1


class RiverConfigElement(BaseModel):
    id: int
    name: str
    model: Literal["rain_like", "stem_flux"]
    geometry: RiverGeometryConfig
    data_source: RiverDataSource
    physical: PhysicalModelConfig
    biogeochemical: BGCProfile = Field(default_factory=dict)
    concentrations: list[str] = Field(default_factory=list)
    biogeochemical_profile: str | None = None


class RiverConfig(RootModel, Iterable):
    root: OrderedDict[int, RiverConfigElement]

    @staticmethod
    def _load_domain_rivers(
        domain_file_path: PathLike | None,
    ) -> dict[int, dict]:
        """
        Loads the information stored in a domain-specific river file.

        A common pattern is that the general information about rivers is stored
        into a `main` file and then the information specific to a particular
        domain is stored into a `domain` file.

        This method reads a JSON domain file, parses its contents, and extracts
        the 'rivers' section. Each river's data is stored in a dictionary,
        with the river ID as the key and the corresponding river information as
        the value.

        Args:
            domain_file_path: The file path to the domain JSON file containing
                river data. If None, an empty dictionary is returned.

        Returns:
            A dictionary containing river data. The keys are river IDs, and
            the values are dictionaries representing their respective details.

        Raises:
            ValueError: If the domain file does not contain a "rivers" section.
        """
        if domain_file_path is None:
            return {}

        domain_file_path = Path(domain_file_path)
        domain_file_content = json.loads(domain_file_path.read_text())

        if "rivers" not in domain_file_content:
            raise ValueError(
                f'The domain file "{domain_file_path}" does not contain '
                f'a "rivers" section'
            )
        return {r["id"]: r for r in domain_file_content["rivers"]}

    @staticmethod
    def _apply_physical_defaults(raw_river: dict, physical_defaults: dict):
        """
        Applies default physical properties to a raw river object.

        This static method updates the physical properties of a given raw
        river object  by combining the provided physical defaults with any
        existing physical properties in the object. If physical defaults are
        not provided, no action is taken.

        Args:
            raw_river (dict): The raw river object where the physical defaults
                should be applied. The object should have a 'physical' key
                representing existing physical properties if applicable.
            physical_defaults (dict): A dictionary of default physical
                properties to apply to the raw river object. If None, no
                updates are performed.
        """
        if physical_defaults is None:
            return

        physical = physical_defaults.copy()
        if "physical" in raw_river:
            physical.update(raw_river["physical"])
        raw_river["physical"] = physical

    @staticmethod
    def _merge_domain_river(raw_river: dict, raw_domain_river: dict):
        """
        Merges the information read from a domain-specific river file into the
        main river configuration.

        This method updates the `raw_river` dictionary by merging the contents
        of `raw_domain_river` into it.

        Args:
            raw_river: The dictionary representing the main configuration of
                the river. This object will be updated with the contents of
                `raw_domain_river`.
            raw_domain_river: The dictionary containing domain-specific
                overrides for the river.

        Raises:
            ValueError: If the `name` field in `raw_domain_river` differs from
                the `name` field in raw_river`. This ensures that the river
                names across configuration sources are consistent.
        """
        current_name = raw_river["name"]
        domain_name = raw_domain_river.get("name", None)
        if domain_name is not None and domain_name != current_name:
            raise ValueError(
                f"River with id {raw_river['id']} has a different "
                "name in the domain file compared to the river "
                f"main configuration file (from {current_name} to "
                f"{domain_name})"
            )

        # We can safely update the values that are not dictionaries.
        # Instead, if they are dictionaries, we need to carefully merge them.
        without_maps = {
            d: r
            for d, r in raw_domain_river.items()
            if not isinstance(r, Mapping)
        }
        raw_river.update(without_maps)

        if "biogeochemical" in raw_domain_river:
            if "biogeochemical" not in raw_river:
                raw_river["biogeochemical"] = {}
            raw_river["biogeochemical"].update(
                raw_domain_river["biogeochemical"]
            )

        if "geometry" in raw_domain_river:
            if "geometry" not in raw_river:
                raw_river["geometry"] = raw_domain_river["geometry"].copy()
            else:
                raw_river["geometry"] = RiverGeometryConfig.merge_dicts(
                    raw_river["geometry"],
                    raw_domain_river["geometry"],
                )

    @staticmethod
    def _apply_biogeochemical_profile(
        raw_river: dict, profiles: dict[str, BGCProfile]
    ):
        biogeochemical = {}
        if "biogeochemical_profile" in raw_river:
            biogeochemical = profiles[
                raw_river["biogeochemical_profile"]
            ].copy()

        if "biogeochemical" in raw_river:
            biogeochemical.update(raw_river["biogeochemical"])

        raw_river["biogeochemical"] = biogeochemical

    @staticmethod
    def _convert_legacy_physical(raw_river: dict):
        """
        Converts legacy physical data within a river's raw data.

        If the "physical" key does not exist in the `raw_river` dictionary, or
        if the "physical" key already contains a "type" key, no action is
        taken. Otherwise, the function processes the given raw data by assuming
        that they contain two values for the temperature (that will be
        interpreted as a SinusoidalConfig) and one for the salinity (that will
        be interpreted as a ConstantConfig).

        Args:
            raw_river (dict): The dictionary containing raw river data that
                potentially includes physical attributes such as average
                temperature, temperature variation, and average salinity.
        """
        if "physical" not in raw_river:
            return
        if "type" in raw_river["physical"]:
            return

        new_physical = {
            "temperature": dict(
                type="sinusoidal",
                average=raw_river["physical"]["average_temperature"],
                variation=raw_river["physical"]["temperature_variation"],
            ),
            "salinity": dict(
                type="constant",
                value=raw_river["physical"]["average_salinity"],
            ),
        }
        raw_river["physical"] = new_physical

    @classmethod
    def from_json(
        cls, file_path: PathLike, domain_file_path: PathLike | None = None
    ) -> "RiverConfig":
        """
        Parses a JSON file containing river configuration and optionally
        combines it with domain-specific file to create a `RiverConfig`
        instance.

        Args:
            file_path (PathLike): Path to the JSON file containing river
                configuration.
            domain_file_path (PathLike | None): Optional path to the
                domain-specific river configuration file.

        Returns:
            RiverConfig: An instance of the `RiverConfig` class populated
                with the parsed configuration data.

        Raises:
            ValueError: If rivers in the configuration contain duplicate IDs.
        """
        file_path = Path(file_path)
        json_content = json.loads(file_path.read_text())
        rivers = json_content["rivers"]

        profiles = json_content.get("bgc_profiles", {})
        physical_defaults = json_content.get("physical_defaults", {})
        geometry_defaults = json_content.get("geometry_defaults", {})
        default_model = json_content.get("default_model", None)

        domain_rivers = cls._load_domain_rivers(domain_file_path)

        config_root: OrderedDict[int, RiverConfigElement] = OrderedDict()
        for raw_river in rivers:
            cls._apply_physical_defaults(raw_river, physical_defaults)

            if raw_river["id"] in domain_rivers:
                cls._merge_domain_river(
                    raw_river, domain_rivers[raw_river["id"]]
                )

            cls._apply_biogeochemical_profile(raw_river, profiles)

            if default_model is not None and "model" not in raw_river:
                raw_river["model"] = default_model

            cls._convert_legacy_physical(raw_river)

            raw_river["geometry"] = RiverGeometryConfig.merge_dicts(
                geometry_defaults, raw_river["geometry"]
            )

            river = RiverConfigElement.model_validate(raw_river)
            if river.id in config_root:
                raise ValueError(
                    f'River "{config_root[river.id].name}" and river '
                    f'"{river.name}" share the same id ({river.id}).'
                )
            config_root[river.id] = river

        return cls(root=config_root)

    def __iter__(self) -> Iterator[RiverConfigElement]:
        return iter(self.root.values())

    def __len__(self):
        return len(self.root)
