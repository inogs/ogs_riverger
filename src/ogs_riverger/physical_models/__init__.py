from pydantic import BaseModel

from ogs_riverger.physical_models.components import PhysicalComponentConfig


class PhysicalModelConfig(BaseModel):
    temperature: PhysicalComponentConfig
    salinity: PhysicalComponentConfig
