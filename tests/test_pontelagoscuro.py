from datetime import datetime
from datetime import timezone

import pytest

from ogs_riverger.pontelagoscuro.download_tools import read_discharge_file
from ogs_riverger.pontelagoscuro.download_tools import save_discharges_to_file


@pytest.mark.external_resources
async def test_download_data_into_dir(tmp_path):
    """
    GIVEN a tuple with the start and end date,
    WHEN the function `save_discharges_to_file` is executed,
    THEN a valid CSV file that can be read by the `read_discharge_file`
      function is generated.
    """
    start_time = datetime(2023, 11, 1, tzinfo=timezone.utc)
    end_time = datetime(2024, 11, 1, tzinfo=timezone.utc)

    download_path = tmp_path / "po_river.csv"

    await save_discharges_to_file(download_path, start_time, end_time)

    assert download_path.exists()

    downloaded_data = read_discharge_file(download_path)

    # number of columns
    assert downloaded_data.shape[1] == 2
    assert downloaded_data.shape[0] > 0
