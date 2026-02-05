from typing import Annotated
from typing import Literal
from typing import Union

from pydantic import BaseModel
from pydantic import Field


class SinusoidalConfig(BaseModel):
    average: float
    variation: float
    type: Literal["sinusoidal"] = "sinusoidal"


class ConstantValueConfig(BaseModel):
    value: float
    type: Literal["constant"] = "constant"


PhysicalComponentConfig = Annotated[
    Union[SinusoidalConfig, ConstantValueConfig],
    "PhysicalComponent",
    Field(discriminator="type"),
]
