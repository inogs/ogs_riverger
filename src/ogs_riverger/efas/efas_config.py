import inspect
import json
import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from ogs_riverger import RiverRawData


def read_efas_config(rivers_data: Iterable[RiverRawData]) -> pd.DataFrame:
    """Processes EFAS river configuration data and generates a DataFrame
    containing the relevant river metadata.

    This function iterates over a collection of EFAS river data,
    extracts specific metadata including IDs, names, and geographical
    coordinates, and structures the data into a pandas DataFrame.
    The resulting DataFrame contains columns for river
    identifiers, names, mouth latitude, longitude, and their respective
    indices.

    Args:
        rivers_data : An iterable of RiverRawData objects. Each RiverRawData
            shall have a dictionary with the following keys:
            - "latitude" (float): Latitude of the river's mouth.
            - "longitude" (float): Longitude of the river's mouth.
            - "latitude_index" (int): Index of the mouth's latitude.
            - "longitude_index" (int): Index of the mouth's longitude.

    Returns:
        pandas.DataFrame: A DataFrame with the columns:
            - "id" (int): River IDs.
            - "name" (str): River names.
            - "mouth_latitude" (float): Latitudes of the river mouths.
            - "mouth_longitude" (float): Longitudes of the river mouths.
            - "mouth_latitude_index" (int): Latitude indices of the river
              mouths.
            - "mouth_longitude_index" (int): Longitude indices of the river
              mouths.

    Raises:
        KeyError: If the provided dictionary does not contain one of the
          required keys:
          ("latitude", "longitude", "latitude_index", "longitude_index").
    """
    ids = []
    names = []
    mouth_latitude = []
    mouth_longitude = []
    mouth_lat_index = []
    mouth_lon_index = []
    for raw_river in rivers_data:
        ids.append(raw_river.id)
        names.append(raw_river.name)
        efas_values = raw_river.raw_data
        mouth_latitude.append(np.float32(efas_values["latitude"]))
        mouth_longitude.append(np.float32(efas_values["latitude_index"]))
        mouth_lat_index.append(int(efas_values["latitude_index"]))
        mouth_lon_index.append(int(efas_values["longitude_index"]))
    overall_data = pd.DataFrame(
        {
            "id": ids,
            "name": names,
            "mouth_latitude": mouth_latitude,
            "mouth_longitude": mouth_longitude,
            "mouth_latitude_index": mouth_lat_index,
            "mouth_longitude_index": mouth_lon_index,
        }
    )
    overall_data.set_index(["id", "name"])
    return overall_data


def read_efas_config_file(config_file: Path) -> pd.DataFrame:
    """Reads and processes the EFAS (European Flood Awareness System)
    configuration file to extract river details and their associated data
    sources.

    This function loads a JSON configuration file containing information about
    rivers, processes the information, and returns the EFAS configuration
    data in the form of a Pandas DataFrame.

    Args:
        config_file (Path): The path to the configuration file in JSON format.
            The file must have an ASCII encoding and include a "rivers" key
            with the details for each river.

    Returns:
        pd.DataFrame: A DataFrame containing processed river data, including
        river IDs, names, and their associated EFAS data sources.
    """
    logger = logging.getLogger(f"{__name__}.{inspect.stack()[0][3]}")
    logger.debug("Reading configuration file %s", config_file)
    with config_file.open("r", encoding="ascii") as file:
        rivers_data = json.load(file)["rivers"]

    config_content = []
    for river in rivers_data:
        river_id = int(river["id"])
        name = river["name"]
        efas_config = river["data_source"]
        config_content.append(
            RiverRawData(id=river_id, name=name, raw_data=efas_config)
        )

    logger.debug(
        "%s rivers have been found inside the config file", len(config_content)
    )
    return read_efas_config(config_content)
