import json
from collections import OrderedDict
from collections.abc import Iterable
from collections.abc import Iterator
from os import PathLike
from pathlib import Path
from typing import Annotated
from typing import Union

from pydantic import BaseModel
from pydantic import Field
from pydantic import RootModel

from ogs_riverger.efas.efas_config import EFASConfigElement


# Right now we have only one RiverDataSource
RiverDataSource = Annotated[
    Union[EFASConfigElement], Field(discriminator="type")
]


class RiverConfigElement(BaseModel):
    id: int
    name: str
    data_source: RiverDataSource


class RiverConfig(RootModel, Iterable):
    root: OrderedDict[int, RiverConfigElement]

    @classmethod
    def from_json(cls, file_path: PathLike) -> "RiverConfig":
        file_path = Path(file_path)

        json_content = json.loads(file_path.read_text())
        rivers = json_content["rivers"]

        config_root: OrderedDict[int, RiverConfigElement] = OrderedDict()
        for raw_river in rivers:
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
