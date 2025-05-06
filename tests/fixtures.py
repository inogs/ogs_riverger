from pathlib import Path

import pytest

from ogs_riverger.read_config import RiverConfig
from ogs_riverger.settings import Settings


@pytest.fixture
def test_data_dir():
    return Path(__file__).resolve().parent / "data"


@pytest.fixture(scope="session")
def settings():
    return Settings()


@pytest.fixture
def config_file_example(tmp_path):
    content_example = """
    {
      "variables": {
        "B1c": {
          "unit": "mgC/m^3"
        },
        "B1n": {
          "unit": "mmolN/m^3"
        },
        "B1p": {
          "unit": "mmolP/m^3"
        }
      },
      "geometry_defaults": {
        "width": 500,
        "depth": 3,
        "stem_length": 10
      },
      "physical_defaults": {
        "average_temperature": 10,
        "temperature_variation": 5,
        "average_salinity": 5
      },
      "default_model": "stem_flux",
      "rivers": [
        {
          "id": 0,
          "name": "Roia",
          "geometry": {
            "mouth_longitude": 7.60607812492043,
            "mouth_latitude": 43.7889609932052,
            "side": "N"
          },
          "data_source": {
            "type": "EFAS",
            "longitude": 7.60833333,
            "latitude": 43.79166667,
            "longitude_index": 1971,
            "latitude_index": 1707
          },
          "biogeochemical_profile": "tyr"
        },
        {
          "id": 1,
          "name": "Centa",
          "geometry": {
            "mouth_longitude": 8.22429947600262,
            "mouth_latitude": 44.045,
            "side": "N",
            "stem": [
              "-6z",
              "+4m"
            ]
          },
          "data_source": {
            "type": "EFAS",
            "longitude": 8.225,
            "latitude": 44.04166667,
            "longitude_index": 2008,
            "latitude_index": 1692
          },
          "biogeochemical_profile": "tyr"
        },
        {
          "id": 2,
          "name": "Entella",
          "geometry": {
            "mouth_longitude": 9.33052903402932,
            "mouth_latitude": 44.3100179047012,
            "side": "N"
          },
          "data_source": {
            "type": "EFAS",
            "longitude": 9.325,
            "latitude": 44.30833333,
            "longitude_index": 2074,
            "latitude_index": 1676
          },
          "biogeochemical_profile": "tyr"
        },
        {
          "id": 3,
          "name": "Magra Ligure",
          "geometry": {
            "mouth_longitude": 9.98676752631333,
            "mouth_latitude": 44.0482528898037,
            "side": "N"
          },
          "data_source": {
            "type": "EFAS",
            "longitude": 9.99166667,
            "latitude": 44.04166667,
            "longitude_index": 2114,
            "latitude_index": 1692
          },
          "biogeochemical_profile": "tyr"
        },
        {
          "id": 4,
          "name": "Serchio",
          "geometry": {
            "mouth_longitude": 10.2689845511678,
            "mouth_latitude": 43.7812689069969,
            "side": "E"
          },
          "data_source": {
            "type": "EFAS",
            "longitude": 10.275,
            "latitude": 43.775,
            "longitude_index": 2131,
            "latitude_index": 1708
          },
          "biogeochemical_profile": "tyr"
        },
        {
          "id": 5,
          "name": "Arno",
          "geometry": {
            "mouth_longitude": 10.2736526600165,
            "mouth_latitude": 43.6807026831873,
            "side": "E"
          },
          "data_source": {
            "type": "EFAS",
            "longitude": 10.29166667,
            "latitude": 43.675,
            "longitude_index": 2132,
            "latitude_index": 1714
          },
          "biogeochemical_profile": "tyr"
        }
      ]
    }
    """

    json_file = tmp_path / "main_config.json"
    json_file.write_text(content_example)

    return json_file


@pytest.fixture
def config_example(config_file_example):
    return RiverConfig.from_json(config_file_example)
