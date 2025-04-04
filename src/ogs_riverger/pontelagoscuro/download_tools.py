# fmt: off
"""Functions to download data on the river discharge of the Po River near the
Pontelagoscuro station (Ferrara).

Data is retrieved from the ERDDAP server hosted at larissa.ogs.it.
"""
# fmt: on
from datetime import datetime
from io import StringIO
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
from httpx import AsyncClient
from httpx import URL

from ogs_riverger.utils.retrievers import http_retrieve_on_file


SOURCE_URL = "https://larissa.ogs.it/erddap/tabledap/Pontelagoscuro_TS.csv"
URL_DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
TIMEOUT_RIVERS = 15


async def save_discharges_to_file(
    output_file: Path,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
):
    """
    Downloads Po River discharge data from the Pontelagoscuro station in
    Ferrara and save them into a CSV file with two columns: reference time and
    river discharge (m^3 s^-1).

    Args:
        output_file (PathLike): The path to save the CSV file to.
        start_time (datetime | None): The beginning of the data range. If
            `None`, the generated URL will link to a CSV file with all
            available data up to `end_time`.
        end_time (datetime | None): The end of the data range. If `None`, the
            generated URL will link to a CSV file with all available data
            starting from `start_time`.
    """

    query = "time,river_discharge"
    start_query = (
        ""
        if not start_time
        else f"&time>={start_time.strftime(URL_DATE_FORMAT)}"
    )
    end_query = (
        ""
        if not start_time
        else f"&time<={end_time.strftime(URL_DATE_FORMAT)}"
    )
    url = URL(
        SOURCE_URL,
        query=quote_plus(f"{query}{start_query}{end_query}").encode("utf-8"),
    )
    async with AsyncClient(timeout=TIMEOUT_RIVERS) as client:
        await http_retrieve_on_file(client=client, url=url, dest=output_file)


def read_discharge_file(discharge_file: Path) -> pd.DataFrame:
    """Reads the content of a CSV file downloaded using the
    `save_discharges_to_file` function and returns it as a `Pandas DataFrame`.

    Args:
        discharge_file (PathLike): The path to the CSV file.

    Returns:
        DataFrame: A DataFrame containing the content of the file.
    """
    with open(discharge_file, "r") as f:
        dframe = pd.read_csv(
            StringIO(f.read()),
            skiprows=[1],
            parse_dates=["time"],
            dtype={"river_discharge": np.float64},
        )
    return dframe
