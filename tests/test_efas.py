from datetime import datetime
from datetime import timedelta
from itertools import product as cart_prod
from logging import getLogger
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import pytest
import xarray as xr
from pydantic import SecretStr

from ogs_riverger.efas.download_tools import AreaSelection
from ogs_riverger.efas.download_tools import download_from_cdsapi
from ogs_riverger.efas.download_tools import download_yearly_data_from_cdsapi
from ogs_riverger.efas.download_tools import EfasCEMSDataFormat
from ogs_riverger.efas.download_tools import EfasCEMSDownloadFormat
from ogs_riverger.efas.download_tools import EfasDataSource
from ogs_riverger.efas.download_tools import EfasOperationalDownloader
from ogs_riverger.efas.download_tools import get_cdsapi_client
from ogs_riverger.efas.download_tools import HistoricalCDSApiRequest
from ogs_riverger.efas.efas_manager import generate_efas_climatology
from ogs_riverger.efas.efas_manager import generate_efas_domain_file
from ogs_riverger.efas.efas_manager import read_efas_data_files


class CDSClientMock:
    """
    A mock that simulates the interface of `cdsapi.Client`, storing the
    requests that would have been made to the API.
    """

    def __init__(self):
        self._requests = []

    def retrieve(self, name: str, request, output_file) -> None:
        self._requests.append(request)

    def get_requests(self) -> tuple[dict, ...]:
        return tuple(self._requests)


@pytest.fixture
def efas_domain_file(test_data_dir):
    """
    Return a path to a file containing the EFAS domain.
    """
    return test_data_dir / "efas_domain.nc"


class EfasLikeFileGenerator:
    """
    Writes a file with the same format of the ones downloaded from
    the EFAS archive.
    """

    def __init__(
        self,
        tmp_path: Path,
        start_time: datetime,
        end_time: datetime,
        efas_domain: xr.Dataset,
        file_type=EfasDataSource.FORECAST,
    ):
        self._tmp_path = tmp_path
        self._tmp_path.mkdir(exist_ok=True)

        self._start_time = start_time
        self._end_time = end_time

        self._efas_domain = efas_domain

        if file_type not in (
            EfasDataSource.FORECAST,
            EfasDataSource.HISTORICAL,
        ):
            raise ValueError(f"Invalid file type: {file_type}")

        self._file_type = file_type

    def create(self, output_file: Path):
        lats = self._efas_domain.latitude.values[1500:2500]
        lons = self._efas_domain.longitude.values[1600:2500]

        time_dim = int(
            (self._end_time - self._start_time).total_seconds() / (3600 * 6)
        )

        start_time64 = np.datetime64(
            self._start_time.strftime("%Y-%m-%dT%H:%M:%S"), "s"
        )
        times = start_time64 + np.arange(
            0, time_dim * 3600 * 6, 3600 * 6, dtype="timedelta64[s]"
        )

        data_raw = np.full(
            (1,), fill_value=self._start_time.year, dtype=np.float32
        )
        data_raw = np.broadcast_to(
            data_raw, (time_dim, 1, lats.shape[0], lons.shape[0])
        )

        data = xr.DataArray(
            data=data_raw,
            dims=("time", "step", "latitude", "longitude"),
            coords={
                "time": times.astype("datetime64[ns]"),
                "latitude": lats,
                "longitude": lons,
                "valid_time": (
                    ("time", "step"),
                    times.astype("datetime64[ns]")[:, np.newaxis],
                ),
            },
        )

        dataset = xr.Dataset({"dis06": data})
        nc_path = self._tmp_path / "data_operational-version-5.nc"
        dataset.to_netcdf(nc_path)

        with ZipFile(output_file, "w") as zip_file:
            zip_file.write(nc_path, arcname=nc_path.name)


@pytest.mark.external_resources
def test_efas_domain_file_download(efas_domain_file, tmp_path):
    """
    GIVEN the efas_domain_file available on the static repository,
    WHEN the function generate_efas_domain_file is called,
    THEN it returns a file with the same dimensions and values as the one
        available on the static repository.
    """
    tmp_path = Path(tmp_path)

    reference_domain_file = xr.load_dataset(efas_domain_file)
    generated_domain_file = xr.load_dataset(
        generate_efas_domain_file(tmp_path / "efas_domain.nc")
    )

    assert np.allclose(
        reference_domain_file.latitude, generated_domain_file.latitude
    )
    assert np.allclose(
        reference_domain_file.longitude, generated_domain_file.longitude
    )


def test_historical_cdsapi_request_dump():
    """
    GIVEN a year for which the historical data is requested,
    WHEN the request is dumped,
    THEN we obtain a dictionary with the correct keys and values.
    """
    request = HistoricalCDSApiRequest(year=(2018,), area=None)
    dumped = request.dump()

    assert dumped["hyear"] == ("2018",)
    assert dumped["hmonth"] == tuple(f"{d:02}" for d in range(1, 13))
    assert "year" not in dumped
    assert "month" not in dumped
    assert dumped["download_format"] == "zip"


@pytest.mark.parametrize(
    "year,month,n_days",
    [(2015, 1, 31), (2015, 2, 28), (2016, 2, 29), (2015, 4, 30)],
)
def test_download_forecast_data_single_month(year, month, n_days, tmp_path):
    """
    GIVEN a range of dates to download the EFAS forecast,
    WHEN the date range falls within the same month,
    THEN the function downloads all the data into a single file.
    """
    start_date = datetime(year, month, 1)

    if month == 12:
        end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        end_date = datetime(year, month + 1, 1) - timedelta(days=1)

    client = CDSClientMock()
    area = AreaSelection(north=1, west=2, south=3, east=4)

    # noinspection PyTypeChecker
    download_from_cdsapi(
        EfasDataSource.FORECAST,
        start_date,
        end_date,
        tmp_path,
        area,
        cdsapi_client=client,
    )

    requests = client.get_requests()

    assert len(requests) == 1
    assert tuple(int(i) for i in requests[0]["day"]) == tuple(
        range(1, n_days + 1)
    )
    assert requests[0]["month"] == (f"{month:0>2}",)
    assert requests[0]["year"] == (f"{year}",)
    assert requests[0]["area"] == area.model_dump()

    assert requests[0]["data_format"] == "netcdf"
    assert requests[0]["download_format"] == "zip"
    assert requests[0]["originating_centre"] == "ecmwf"
    assert requests[0]["variable"] == "river_discharge_in_the_last_6_hours"
    assert requests[0]["model_levels"] == "surface_level"


def test_download_forecast_data_two_months(tmp_path):
    """
    GIVEN a range of dates to download the EFAS forecast,
    WHEN the date range is within the same year but spans two different
         months,
    THEN the function downloads the data using one file per month.
    """
    start_date = datetime(2024, 3, 12)
    end_date = datetime(2024, 4, 18)

    client = CDSClientMock()
    area = AreaSelection(north=1, west=2, south=3, east=4)

    # noinspection PyTypeChecker
    download_from_cdsapi(
        EfasDataSource.FORECAST,
        start_date,
        end_date,
        tmp_path,
        area,
        cdsapi_client=client,
    )

    requests = client.get_requests()

    assert len(requests) == 2

    days_first_request = tuple(int(i) for i in requests[0]["day"])
    days_second_request = tuple(int(i) for i in requests[1]["day"])

    assert days_first_request == tuple(range(start_date.day, 31 + 1))
    assert days_second_request == tuple(range(1, end_date.day + 1))


@pytest.mark.parametrize("year", [2012, 2014, 2015])
def test_download_yearly_data(year, tmp_path):
    """
    GIVEN a year to download the yearly EFAS data for,
    WHEN the function is executed,
    THEN the function downloads the data using the correct cdsapi request.
    """
    client = CDSClientMock()
    area = AreaSelection(north=1, west=2, south=3, east=4)

    # noinspection PyTypeChecker
    download_yearly_data_from_cdsapi(
        EfasDataSource.HISTORICAL,
        year=year,
        output_file=tmp_path / f"historical_{year}.netcdf.zip",
        area=area,
        cdsapi_client=client,
    )

    requests = client.get_requests()

    assert len(requests) == 1

    assert requests[0]["data_format"] == "netcdf"
    assert requests[0]["download_format"] == "zip"
    assert requests[0]["originating_centre"] == "ecmwf"
    assert requests[0]["variable"] == "river_discharge_in_the_last_6_hours"
    assert requests[0]["model_levels"] == "surface_level"
    assert requests[0]["system_version"] == "version_5_0"
    assert requests[0]["hyear"] == (f"{year}",)

    assert requests[0]["hmonth"] == tuple(f"{i:02}" for i in range(1, 13))
    assert requests[0]["hday"] == tuple(
        "{:0>2}".format(d) for d in range(1, 32)
    )
    assert requests[0]["time"] == ("00:00", "06:00", "12:00", "18:00")


def test_read_efas_zipped_data(config_example, efas_domain_file, tmp_path):
    """
    GIVEN a set of EFAS files,
    WHEN the function read_efas_data_files is called,
    THEN it returns a dataset with the correct dimensions and values.
    """
    tmp_path = tmp_path / "test_read_efas_zipped_data"
    tmp_path.mkdir(exist_ok=True)

    efas_domain = xr.load_dataset(efas_domain_file)

    start_date = datetime(2024, 9, 3)
    change_month = datetime(2024, 10, 1)
    end_date = datetime(2024, 10, 22)
    file_generator1 = EfasLikeFileGenerator(
        tmp_path / "f1", start_date, change_month, efas_domain=efas_domain
    )
    file_generator2 = EfasLikeFileGenerator(
        tmp_path / "f2", change_month, end_date, efas_domain=efas_domain
    )

    test_file1 = tmp_path / "test_file01.zip"
    test_file2 = tmp_path / "test_file02.zip"
    file_generator1.create(test_file1)
    file_generator2.create(test_file2)

    test_dataset = read_efas_data_files(
        (test_file1, test_file2),
        config_example,
        efas_domain_file=efas_domain_file,
    )
    output_dims = test_dataset.dis06.dims

    assert output_dims == ("time", "id")

    n_days = (end_date - start_date).days
    assert test_dataset.dis06.shape[0] == n_days * 4
    assert test_dataset.dis06.shape[1] == len(config_example)


def test_generate_efas_climatology(config_example, efas_domain_file, tmp_path):
    """
    GIVEN a set of yearly EFAS files,
    WHEN the function generate_efas_climatology is called,
    THEN it generates a climatology dataset with the correct dimensions and
        values
    """
    tmp_path = tmp_path / "test_generate_efas_climatology"
    tmp_path.mkdir(exist_ok=True)

    efas_domain = xr.load_dataset(efas_domain_file)

    years = (2020, 2021, 2022, 2023)
    clima_files = {}
    for year in years:
        start_date = datetime(year, 1, 1)
        end_date = datetime(year + 1, 1, 1)
        file_generator = EfasLikeFileGenerator(
            tmp_path / f"year_{year}",
            start_date,
            end_date,
            efas_domain=efas_domain,
        )

        test_file = tmp_path / f"test_year_{year}.zip"
        file_generator.create(test_file)
        clima_files[year] = test_file

    clima_file = tmp_path / "clima.nc"
    generate_efas_climatology(
        clima_files,
        config_example,
        clima_file,
        efas_domain_file=efas_domain_file,
    )

    with xr.open_dataset(clima_file) as ds:
        clima_dataset = ds
    assert sorted(set(clima_dataset.month.values)) == list(range(1, 13))
    assert clima_dataset.discharge.shape[0] == 366

    leap_days = np.logical_and(
        clima_dataset.month.values == 2, clima_dataset.day.values == 29
    )

    # Test that February, 29th coincides with the value of the only
    # leap year (2020). All the other days are the average of the four years
    assert np.all(clima_dataset.discharge.values[leap_days] == 2020.0)
    assert np.all(clima_dataset.discharge.values[~leap_days] == 2021.5)


async def test_efas_operational_downloader_generate_caches(tmp_path):
    """
    GIVEN a directory containing operational EFAS files,
    WHEN an `EfasOperationalDownloader` is instantiated with that directory
      as its data source,
    THEN it generates a cache storing the paths and sizes of the EFAS files.
    """
    efas_cache = tmp_path / "efas_cache_test"
    efas_cache.mkdir()

    cached_files = (
        efas_cache / "eud.fc.dis_2024111312.grb",
        efas_cache / "dwd.fc.dis_2024111312.grb",
        efas_cache / "eud.fc.dis_2024111400.grb",
    )
    for f in cached_files:
        f.touch()

    downloader = await EfasOperationalDownloader.create(
        tmp_path / "efas_cache_test",
        efas_user="user",
        efas_password=SecretStr("password"),
    )

    assert len(downloader._cache) == 3


@pytest.mark.parametrize("one_with_diff_size", [True, False])
async def test_efas_operational_downloader_avoid_download_cached(
    tmp_path, one_with_diff_size
):
    """
    GIVEN an `EfasOperationalDownloader` with some files from the remote FTP
      server already cached,
    WHEN the `download_efas_operational_data` method is called,
    THEN the `EfasOperationalDownloader` skips downloading files that are
      already in the cache and match the size of the files on the remote
      server.
    """

    class EODTest(EfasOperationalDownloader):
        pass

    # The size of the fake files used as cache
    f_size = 42

    efas_cache = tmp_path / "efas_cache"
    efas_cache.mkdir()

    test_files = (
        "eud.fc.dis_2024111312.grb",
        "dwd.fc.dis_2024111312.grb",
        "eud.fc.dis_2024111400.grb",
        "eud.fc.dis_2024111412.grb",
        "eud.fc.dis_2024111500.grb",
        "eud.fc.dis_2024111512.grb",
        "eud.fc.dis_2024111600.grb",
    )

    # How many of the previous files are cached
    n_cached_files = 5

    cached_files = tuple(efas_cache / t for t in test_files[:n_cached_files])

    if one_with_diff_size:
        cached_of_the_right_size = cached_files[:-1]
    else:
        cached_of_the_right_size = cached_files

    for cached_file in cached_of_the_right_size:
        # Create a file with size `f_size`
        with open(cached_file, "b+w") as f:
            f.seek(f_size - 1)
            f.write(b"\0")
    if one_with_diff_size:
        cached_files[-1].touch()

    EODTest._get_remote_server_available_files = AsyncMock(
        return_value={t: (Path("server_dir") / t, f_size) for t in test_files}
    )
    EODTest._single_download = AsyncMock(
        side_effect=lambda remote_path: efas_cache / remote_path.name
    )

    downloader = await EODTest.create(
        efas_cache,
        efas_user="user",
        efas_password=SecretStr("password"),
    )

    returned_files = await downloader.download_efas_operational_data(
        start_time=datetime(2024, 11, 13), end_time=datetime(2024, 11, 18)
    )

    # We expect to the dwd file to have been ignored and to return all the
    # other files.
    n_expected_returned_files = len(test_files) - 1
    assert len(returned_files) == n_expected_returned_files

    # Add "+ 1" because we ignore the dwd file (not the right version)
    expected_n_of_downloads = n_expected_returned_files - len(cached_files) + 1
    if one_with_diff_size:
        expected_n_of_downloads += 1

    assert EODTest._single_download.call_count == expected_n_of_downloads


async def test_efas_operational_downloader_ok_if_in_cache(tmp_path):
    """
    GIVEN an `EfasOperationalDownloader` with some files cached that are not
      available on the FTP server, which instead has fallback versions of
      those files,
    WHEN the `download_efas_operational_data` method is called,
    THEN the cached files are used, and the fallback versions on the FTP server
      are not downloaded.
    """

    class EODTest(EfasOperationalDownloader):
        pass

    efas_cache = tmp_path / "efas_cache"
    efas_cache.mkdir()

    test_files = (
        "eud.fc.dis_2024111312.grb",
        "dwd.fc.dis_2024111312.grb",
        "eud.fc.dis_2024111400.grb",
        "dwd.fc.dis_2024111400.grb",
        "eud.fc.dis_2024111412.grb",
        "dwd.fc.dis_2024111412.grb",
        "eud.fc.dis_2024111500.grb",
        "dwd.fc.dis_2024111500.grb",
        "eud.fc.dis_2024111512.grb",
        "dwd.fc.dis_2024111512.grb",
    )

    cached_files = tuple(efas_cache / t for t in test_files)

    for cached_file in cached_files:
        cached_file.touch()

    EODTest._get_remote_server_available_files = AsyncMock(
        return_value={
            f.name: (f, 0) for f in cached_files if f.name.startswith("dwd")
        }
    )
    EODTest._single_download = AsyncMock(
        side_effect=lambda remote_path: efas_cache / remote_path.name
    )

    downloader = await EODTest.create(
        efas_cache,
        efas_user="user",
        efas_password=SecretStr("password"),
        versions=("eud",),
        fallback_versions={"eud": "dwd"},
    )

    await downloader.download_efas_operational_data(
        start_time=datetime(2024, 11, 13), end_time=datetime(2024, 11, 18)
    )

    assert EODTest._single_download.call_count == 0


async def test_efas_operational_downloader_rounds_start_time(tmp_path):
    """
    GIVEN an `EfasOperationalDownloader`,
    WHEN the `download_efas_operational_data` method is called with a
      `start_time` that is not aligned with midnight or midday,
    THEN the `start_time` is rounded to the next midnight or midday after the
      specified `start_time`.
    """

    class EODTest(EfasOperationalDownloader):
        pass

    test_files = []
    for day, hour, version in cart_prod(range(13, 16), (0, 12), ("eud",)):
        test_files.append(f"{version}.fc.dis_202411{day:0>2}{hour:0>2}.grb")

    EODTest._get_remote_server_available_files = AsyncMock(
        return_value={f: (Path("remote_dir") / f, 0) for f in test_files}
    )
    EODTest._single_download = AsyncMock(
        side_effect=lambda remote_path: tmp_path / remote_path.name
    )

    downloader = await EODTest.create(
        tmp_path,
        efas_user="user",
        efas_password=SecretStr("password"),
        versions=("eud",),
    )

    call_day = await downloader.download_efas_operational_data(
        start_time=datetime(2024, 11, 13), end_time=datetime(2024, 11, 18)
    )

    call_middle_day = await downloader.download_efas_operational_data(
        start_time=datetime(2024, 11, 13, hour=3),
        end_time=datetime(2024, 11, 18),
    )

    call_next_day = await downloader.download_efas_operational_data(
        start_time=datetime(2024, 11, 13, hour=19),
        end_time=datetime(2024, 11, 18),
    )

    assert len(call_day) == len(test_files)
    assert call_middle_day == call_day[1:]
    assert call_next_day == call_day[2:]


async def test_efas_operational_downloader_check_uses_fallback_mechanism(
    tmp_path,
):
    """
    GIVEN an `EfasOperationalDownloader` configured to point to a server where
      some files are missing, but their fallback versions are available,
    WHEN the `download_efas_operational_data` method is called,
    THEN the fallback version files are downloaded.
    """

    class EODTest(EfasOperationalDownloader):
        pass

    test_files = []
    for day, hour, version in cart_prod(range(13, 16), (0, 12), ("eud",)):
        test_files.append(f"{version}.fc.dis_202411{day:0>2}{hour:0>2}.grb")

    test_files[2] = "dwd" + test_files[2][3:]
    test_files[3] = "dwd" + test_files[3][3:]

    EODTest._get_remote_server_available_files = AsyncMock(
        return_value={f: (Path("remote_dir") / f, 0) for f in test_files}
    )
    EODTest._single_download = AsyncMock(
        side_effect=lambda remote_path: tmp_path / remote_path.name
    )

    downloader = await EODTest.create(
        tmp_path,
        efas_user="user",
        efas_password=SecretStr("password"),
        versions=("eud",),
        fallback_versions={"eud": "dwd"},
    )

    downloaded_files = await downloader.download_efas_operational_data(
        start_time=datetime(2024, 11, 13), end_time=datetime(2024, 11, 18)
    )
    right_version = [f for f in downloaded_files if f.name.startswith("eud")]
    fallback_version = [
        f for f in downloaded_files if f.name.startswith("dwd")
    ]

    assert len(right_version) == len(test_files) - 2
    assert len(fallback_version) == 2


@patch("logging.Logger.exception")
async def test_efas_operational_downloader_handles_exceptions(
    log_exception, tmp_path
):
    """
    GIVEN an `EfasOperationalDownloader`,
    WHEN a download executed by the `download_efas_operational_data` method
      fails and raises an exception,
    THEN the method logs the exception, continues execution, and returns only
      the successfully downloaded files.
    """

    class EODTest(EfasOperationalDownloader):
        pass

    test_files = []
    for day, hour, version in cart_prod(range(13, 16), (0, 12), ("eud",)):
        test_files.append(f"{version}.fc.dis_202411{day:0>2}{hour:0>2}.grb")

    n_test_files = len(test_files)
    already_downloaded = 0

    def failing_download(remote_path):
        nonlocal already_downloaded
        already_downloaded += 1
        if already_downloaded == n_test_files - 2:
            raise Exception
        return Path("local_dir") / remote_path.name

    EODTest._get_remote_server_available_files = AsyncMock(
        return_value={f: (Path("remote_dir") / f, 0) for f in test_files}
    )
    EODTest._single_download = AsyncMock(side_effect=failing_download)

    downloader = await EODTest.create(
        tmp_path,
        efas_user="user",
        efas_password=SecretStr("password"),
        versions=("eud",),
        fallback_versions={"eud": "dwd"},
    )

    downloaded_files = await downloader.download_efas_operational_data(
        start_time=datetime(2024, 11, 13), end_time=datetime(2024, 11, 18)
    )

    assert len(downloaded_files) == len(test_files) - 1
    assert log_exception.call_count == 1


@pytest.mark.external_resources
async def test_efas_operational_download_and_read(
    settings, config_example, efas_domain_file, tmp_path
):
    """
    GIVEN a set of options for downloading some operational EFAS files,
    WHEN the files are downloaded using the EfasOperationalDownloader class,
    THEN we receive a dataset with the expected dimensions and variables.
    """
    today = datetime.today()
    start_date = datetime(today.year, today.month, today.day) - timedelta(
        days=20
    )
    end_date = start_date + timedelta(days=3)

    downloader = await EfasOperationalDownloader.create(
        Path(tmp_path),
        efas_user=settings.EFAS_FTP_USER,
        efas_password=settings.EFAS_FTP_PASSWORD,
        versions=("eud",),
        fallback_versions={"eud": "dwd"},
    )

    downloaded_files = await downloader.download_efas_operational_data(
        start_date, end_date
    )
    efas_dataset = read_efas_data_files(
        downloaded_files, config_example, efas_domain_file=efas_domain_file
    )

    assert "time" in efas_dataset.dims
    assert "id" in efas_dataset.dims

    assert "time" in efas_dataset.coords
    assert "dis06" in efas_dataset.variables


@pytest.mark.external_resources
@pytest.mark.parametrize(
    "data_source, download_zipped, data_format",
    cart_prod(
        (EfasDataSource.FORECAST, EfasDataSource.HISTORICAL),
        (True, False),
        EfasCEMSDataFormat,
    ),
)
def test_efas_from_cdsapi_download_and_read(
    settings,
    config_example,
    tmp_path,
    data_source: EfasDataSource,
    download_zipped: bool,
    data_format: EfasCEMSDataFormat,
    efas_domain_file,
):
    """
    GIVEN a set of options for downloading some EFAS files from the CDS API,
    WHEN the `efas_from_cdsapi` method is called with those options,
    THEN we download some files that can be read using the
        `read_efas_data_files` method
    """
    if download_zipped:
        download_format = EfasCEMSDownloadFormat.ZIP
    else:
        download_format = EfasCEMSDownloadFormat.UNARCHIVED

    if data_source is EfasDataSource.HISTORICAL:
        start_date = datetime(year=2016, month=2, day=24)
        end_date = datetime(year=2016, month=3, day=2)
    elif data_source is EfasDataSource.FORECAST:
        now = datetime.now()
        today = datetime(now.year, now.month, now.day)
        start_date = today - timedelta(days=45)
        end_date = start_date + timedelta(days=4)
    else:
        raise ValueError(f"Invalid source: {data_source}")

    cdsapi_client = get_cdsapi_client(
        url=settings.CDS_API_URL,
        key=settings.CDS_API_KEY,
        client_logger=getLogger("tests.cdsapi_client"),
    )
    downloaded_files = download_from_cdsapi(
        service=data_source,
        start_date=start_date,
        end_date=end_date,
        output_dir=tmp_path,
        area=AreaSelection(west=7.0, south=43.0, east=11.0, north=45.5),
        cdsapi_client=cdsapi_client,
        download_format=download_format,
        file_data_format=data_format,
    )

    efas_dataset = read_efas_data_files(
        downloaded_files, config_example, efas_domain_file=efas_domain_file
    )
    assert "time" in efas_dataset.dims
    assert "id" in efas_dataset.dims
    assert "step" not in efas_dataset.dims

    assert "time" in efas_dataset.coords
    assert "dis06" in efas_dataset.variables
