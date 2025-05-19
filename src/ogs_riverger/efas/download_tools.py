import asyncio
import inspect
import logging
from collections.abc import Iterable
from collections.abc import Mapping
from datetime import datetime
from datetime import time
from datetime import timedelta
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Annotated

import aioftp
import anyio
import cdsapi
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import PlainSerializer
from pydantic import SecretStr

from ogs_riverger.utils.area_selection import AreaSelection


EFAS_FTP_URL = "aux.ecmwf.int"
EFAS_FTP_PORT = 21
EFAS_FTP_DIR = "for_OGS"

# A type that represents a tuple of integers, which, when serialized, ensures
# that each element is formatted as a two-digit number. This means adding a
# leading zero to any single-digit numbers.
TwoDigitsPaddedTuple = Annotated[
    tuple[int, ...],
    PlainSerializer(lambda int_tuple: tuple(f"{d:0>2}" for d in int_tuple)),
]

EFAS_DOMAIN = AreaSelection(
    north=72.25,
    south=22.75,
    west=-25.25,
    east=50.25,
)


def build_efas_area_selection(
    *,
    north: float | None,
    south: float | None,
    east: float | None,
    west: float | None,
):
    return AreaSelection(
        north=north if north is not None else EFAS_DOMAIN.north,
        south=south if south is not None else EFAS_DOMAIN.south,
        east=east if east is not None else EFAS_DOMAIN.east,
        west=west if west is not None else EFAS_DOMAIN.west,
    )


class EfasDataSource(Enum):
    """
    EFAS data is available through three distinct services, each catering to
    different temporal datasets:

    1. **OPERATIVE Data (ECMWF FTP Server)**:
       This service provides the most recent data, including forecasts. New
       data is uploaded twice daily and remains available on the FTP server
       for two months.

    2. **FORECAST Data (CEMS CDSAPI System)**:
      After two months, data from the FTP server is migrated to the Copernicus
      Climate Data Store (CDSAPI).
      This dataset is accessible under the product `River discharge and
      related forecasted data by the European Flood Awareness System`-. Data
      in this product spans back to 10 October 2018.

    3. **HISTORICAL Data (CEMS CDSAPI System) **:
       CEMS also hosts a comprehensive historical dataset containing all EFAS
       data since 1992. This is the dataset that we use to generate a
       climatology for the rivers.

    .. _River discharge and related forecasted data by the EFAS
    https://ewds.climate.copernicus.eu/datasets/efas-forecast

    """

    OPERATIVE = "operative"
    FORECAST = "forecast"
    HISTORICAL = "historical"

    def get_cdsapi_request_class(self):
        match self:
            case EfasDataSource.OPERATIVE:
                raise ValueError(
                    "OPERATIVE data can not be downloaded from CDSApi"
                )
            case EfasDataSource.FORECAST:
                return ForecastCDSApiRequest
            case EfasDataSource.HISTORICAL:
                return HistoricalCDSApiRequest
            case _:
                raise ValueError(f'Unknown field: "{self}"')


class EfasCEMSDataFormat(Enum):
    """Specifies the file data formats available on the Copernicus Emergency
    Management Service (CEMS) from the European Flood Awareness System (EFAS).

    Attributes:
        NETCDF: Data is formatted in the NetCDF format.
        GRIB: Data is formatted in the GRIB format.
    """

    NETCDF = "netcdf"
    GRIB = "grib"


EfasCEMSDataFormatWrapper = Annotated[
    EfasCEMSDataFormat,
    PlainSerializer(lambda x: x.value),
]


class EfasCEMSDownloadFormat(Enum):
    """Specifies the format for downloading Copernicus Emergency Management
    Service (CEMS) data from the European Flood Awareness System (EFAS).

    Attributes:
        ZIP: Data is downloaded in a zipped file.
        UNARCHIVED: Data is downloaded in its original, unarchived format.
    """

    ZIP = "zip"
    UNARCHIVED = "unarchived"


EfasCEMSDownloadFormatWrapper = Annotated[
    EfasCEMSDownloadFormat,
    PlainSerializer(lambda x: x.value),
]


class BaseCDSApiRequest(BaseModel):
    """Common CDSApi values shared between historical and forecast requests.

    This class streamlines the creation of requests to the CDSApi by
    providing default values for fields that are unlikely to change.

    Use the "`.dump()`" method of the subclasses of this class to obtain
    a dictionary representation of the request that can be used inside
    the `cdsapi.Client.retrieve()` method.
    """

    model_config = ConfigDict(protected_namespaces=())

    data_format: EfasCEMSDataFormatWrapper = EfasCEMSDataFormat.NETCDF
    download_format: EfasCEMSDownloadFormatWrapper = EfasCEMSDownloadFormat.ZIP
    originating_centre: str = "ecmwf"
    variable: str = "river_discharge_in_the_last_6_hours"
    model_levels: str = "surface_level"
    area: AreaSelection | None = None

    def dump(self, by_alias=True, exclude_none=True):
        return super().model_dump(by_alias=by_alias, exclude_none=exclude_none)


class HistoricalCDSApiRequest(BaseCDSApiRequest):
    """A Request for the Historical EFAS CDSApi."""

    year: TwoDigitsPaddedTuple = Field(serialization_alias="hyear")
    month: TwoDigitsPaddedTuple = Field(
        serialization_alias="hmonth", default=tuple(m for m in range(1, 13))
    )
    day: TwoDigitsPaddedTuple = Field(
        serialization_alias="hday", default=tuple(d for d in range(1, 32))
    )
    system_version: str = "version_5_0"
    time: tuple[str, str, str, str] = ("00:00", "06:00", "12:00", "18:00")


class ForecastCDSApiRequest(BaseCDSApiRequest):
    """A Request for the forecast EFAS CDSApi."""

    year: TwoDigitsPaddedTuple
    month: TwoDigitsPaddedTuple = tuple(m for m in range(1, 13))
    day: TwoDigitsPaddedTuple = tuple(d for d in range(1, 32))

    system_version: str = "operational"
    product_type: str = "high_resolution_forecast"
    time: tuple[str, ...] = ("00:00", "12:00")
    leadtime_hour: tuple[int, ...] = (6, 12, 18, 24)


def get_cdsapi_client(
    url: str | None = None,
    key: SecretStr | None = None,
    client_logger: logging.Logger = None,
) -> cdsapi.Client:
    """Returns a cdsapi.Client instance.

    It also configures the returned client to use a specific logger (if
    submitted)

    Args:
        url: the url of the endpoint of the cdsapi. If it is None, it will be
            read from the ~/.cdsapi file (if exists)
        key: the key of the cdsapi user account. If it is None, it will be
            read from the ~/.cdsapi file (if exists)
        client_logger: Logger that the returned client
            will use to print its messages

    Returns:
        cdsapi.Client instance
    """
    if client_logger is not None:

        def debug_callback(*args, **kwargs):
            return client_logger.debug(*args, **kwargs)

        def info_callback(*args, **kwargs):
            return client_logger.info(*args, **kwargs)

        def warning_callback(*args, **kwargs):
            return client_logger.warning(*args, **kwargs)

        def error_callback(*args, **kwargs):
            return client_logger.error(*args, **kwargs)
    else:
        debug_callback = None
        info_callback = None
        warning_callback = None
        error_callback = None

    client_kwargs = {
        "debug_callback": debug_callback,
        "info_callback": info_callback,
        "warning_callback": warning_callback,
        "error_callback": error_callback,
    }

    if url is not None:
        client_kwargs["url"] = url
    if key is not None:
        client_kwargs["key"] = key.get_secret_value()

    cdsapi_client = cdsapi.Client(**client_kwargs)

    # This is a horrible hack that probably will become not necessary in the
    # next version of cdsapi. It removes the "logging decorator", which is a
    # context manager that changes the configuration of the logger
    if cdsapi_client.__class__.__name__.startswith("Legacy"):
        if hasattr(cdsapi_client, "logging_decorator"):
            cdsapi_client.logging_decorator = lambda x: x

    return cdsapi_client


def download_from_cdsapi(
    service: EfasDataSource,
    start_date: datetime,
    end_date: datetime,
    output_dir: PathLike,
    area: AreaSelection | None = None,
    *,
    cdsapi_client: cdsapi.Client | None = None,
    download_format: EfasCEMSDownloadFormat = EfasCEMSDownloadFormat.ZIP,
    file_data_format: EfasCEMSDataFormat = EfasCEMSDataFormat.NETCDF,
    output_file_mask: str = "efas_{SERVICE}_{DATE}.{FORMAT}{IS_ZIP}",
    date_format: str = "{YEAR}_{MONTH:02d}_{START_DAY:02d}-{END_DAY:02d}",
) -> tuple[Path, ...]:
    """Downloads data from the CDSAPI EFAS archives (forecast or historical).

    Due to limitations in the `cdsapi` interface, this function may
    need to  download multiple files. For example, if the `start_date`
    and `end_date` span different months, a separate file for each
    month must be downloaded.

    The function returns a tuple containing the paths of the
    downloaded files.

    Args:
        service: The service to download data from.
        start_date: The start date to download the data for.
        end_date: The end date
        output_dir: Path where the files will be saved.
        area: The area of the EFAS domain that will be downloaded. Defaults to
            `None`
        cdsapi_client: The cdsapi client that must be used to download the
            file. If it is `None`, this function will initialize a new client
            for the download.
        download_format: Specifies if the downloaded files should be a netcdf
            file or a GRIB
        file_data_format: Specifies if downloading zipped files or not
        output_file_mask: the "mask" that will be used to generate the name
            of the output files. It accepts the following placeholders:

            - {SERVICE} is the name of the service ("forecast" or "historical")
            - {DATE} is the range of dates for which the data is downloaded
            - {FORMAT} is the format of the file ("netcdf" or "grib")
            - {IS_ZIP} is a string that is empty if the downloaded file is not
              zipped, otherwise it is ".zip"

        date_format: the format of the date in the output file name. It accepts
            the following placeholders:

            - {YEAR} is the year of the file
            - {MONTH} is the month of the file
            - {START_DAY} is the start day of the file
            - {END_DAY} is the end day of the file

    Returns:
        A tuple with the paths of all the files that have been downloaded

    Raises:
        ValueError: the output directory does not exist or is not a directory.
    """
    logger = logging.getLogger(f"{__name__}.{inspect.stack()[0][3]}")

    request_class = service.get_cdsapi_request_class()

    logger.debug(
        'Executing function "%s" with service "%s"',
        inspect.stack()[0][3],
        service,
    )

    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        raise ValueError(
            "{} does not exist or is not a directory".format(output_dir)
        )

    logger.debug("Starting date is %s", start_date)
    logger.debug("End date is %s", end_date)

    if cdsapi_client is None:
        client_logger = logging.getLogger(
            f"{__name__}.{inspect.stack()[0][3]}.cdsapi"
        )
        cdsapi_client = get_cdsapi_client(client_logger=client_logger)

    years = range(start_date.year, end_date.year + 1)
    one_day = timedelta(days=1)

    retrieve_args = []
    for year in years:
        for month in range(1, 13):
            start_month_date = datetime(year, month, 1)
            if month != 12:
                end_month_date = datetime(year, month + 1, 1) - one_day
            else:
                end_month_date = datetime(year + 1, 1, 1) - one_day

            if start_month_date > end_date:
                logger.debug(
                    "Month %s of year %s will not be downloaded because it is "
                    "after the end_date (%s)",
                    month,
                    year,
                    end_date,
                )
                continue
            if end_month_date < start_date:
                logger.debug(
                    "Month %s of year %s will not be downloaded because it is "
                    "before the start_date (%s)",
                    month,
                    year,
                    start_date,
                )
                continue

            logger.debug(
                "Preparing request for month %s of year %s", month, year
            )

            start_month_date = max(start_date, start_month_date)
            end_month_date = min(end_date, end_month_date)

            start_day = start_month_date.day
            end_day = end_month_date.day
            is_zip_str = ""
            if download_format is EfasCEMSDownloadFormat.ZIP:
                is_zip_str = ".zip"
            file_date_str = date_format.format(
                YEAR=year,
                MONTH=month,
                START_DAY=start_day,
                END_DAY=end_day,
            )
            output_file_name = output_file_mask.format(
                SERVICE=service.value,
                DATE=file_date_str,
                FORMAT=file_data_format.value,
                IS_ZIP=is_zip_str,
            )
            output_file_path = output_dir / output_file_name

            days = tuple(range(start_day, end_day + 1))

            request = request_class(
                year=(year,),
                month=(month,),
                day=days,
                area=area,
                data_format=file_data_format,
                download_format=download_format,
            )

            retrieve_args.append(
                (f"efas-{service.value}", request, output_file_path)
            )

    def download_file(retrieve_arg):
        _request_name, _request, _output_file_path = retrieve_arg
        logger.debug(
            "Downloading file %s using the following request: %s",
            _output_file_path,
            _request,
        )
        cdsapi_client.retrieve(
            _request_name,
            _request.dump(),
            _output_file_path,
        )
        logger.debug("File %s has been downloaded", _output_file_path)
        return output_file_path

    saved_files = map(download_file, retrieve_args)

    return tuple(saved_files)


def download_yearly_data_from_cdsapi(
    service: EfasDataSource,
    year: int,
    output_file: PathLike,
    area: AreaSelection | None = None,
    *,
    cdsapi_client: cdsapi.Client | None = None,
    download_format: EfasCEMSDownloadFormat = EfasCEMSDownloadFormat.ZIP,
    file_data_format: EfasCEMSDataFormat = EfasCEMSDataFormat.NETCDF,
) -> None:
    """Downloads yearly EFAS data from the CDSAPI.

    This function operates similarly to `download_from_cdsapi`, but instead of
    downloading multiple files, it retrieves a single file containing all the
    data spanning an entire year.

    Args:
        service: The EFAS data source service to download from.
        year: The year for which data is to be downloaded.
        output_file (PathLike): The path where the downloaded data will be
            stored.
        area (AreaSelection, optional): The geographical area specified for
            the data download. Defaults to `None`.
        cdsapi_client (cdsapi.Client, optional): An existing CDSAPI client.
            If not provided, a new client will be created. Defaults to `None`.
        download_format (EfasCEMSDownloadFormat): The format to download the
            files (ZIP or UNARCHIVED). Defaults to
            `EfasCEMSDownloadFormat.ZIP`.
        file_data_format (EfasCEMSDataFormat): The format of the data files
            (NETCDF or GRIB). Defaults to `EfasCEMSDataFormat.NETCDF`.

    Returns:
        None
    """
    logger = logging.getLogger(f"{__name__}.{inspect.stack()[0][3]}")

    request_class = service.get_cdsapi_request_class()

    logger.debug(
        'Executing function "%s" with service "%s"',
        inspect.stack()[0][3],
        service,
    )
    request = request_class(
        year=(year,),
        month=range(1, 13),
        day=range(1, 32),
        area=area,
        data_format=file_data_format,
        download_format=download_format,
    )

    if cdsapi_client is None:
        client_logger = logging.getLogger(
            f"{__name__}.{inspect.stack()[0][3]}.cdsapi"
        )
        cdsapi_client = get_cdsapi_client(client_logger=client_logger)

    cdsapi_client.retrieve(
        f"efas-{service.value}", request.dump(), output_file
    )


class EfasOperationalDownloader:
    """This class manages and updates the local archive of EFAS (European Flood
    Awareness System) operational data by downloading newly available files
    from the official FTP server.

    The primary functionality of this class is provided by the
    `download_efas_operational_data` method, which retrieves all available
    files for a specified time interval. These files are stored in a designated
    directory specified during the object's initialization.

    This class incorporates a basic caching mechanism: if a file with the same
    name and size as one on the FTP server already exists in the specified
    directory, the download is skipped.

    EFAS files are derived from atmospheric models, and multiple files for the
    same time frame may be available on the FTP server, each generated using a
    different model. Within this context, the model used is referred to as the
    "version" of the EFAS file. Typically, two versions are available: "eud"
    and "dwd." Users can specify one or more desired versions during
    initialization, ensuring that only files of the selected versions are
    downloaded. Additionally, a "fallback version" can be specified:
    if a file for a particular version is unavailable, a file corresponding to
    the same date but from a fallback version will be downloaded.

    This class supports asynchronous operations, and its methods should be
    invoked accordingly.

    Args:
      data_dir: The directory where downloaded files will be stored.
      efas_user: The username used to authenticate with the FTP server.
      efas_password: The password used to authenticate with the FTP server.
      versions: The list of EFAS file versions to download, corresponding to
        the prefix of the file names.
      fallback_versions: A mapping of versions to fallback versions. If a file
        for a specified version is unavailable, the corresponding fallback
        version will be used.
      max_n_of_downloads: The maximum number of files to download concurrently.
      cache: A cache of locally available files, mapping file paths to their
        sizes.
    """

    def __init__(
        self,
        data_dir: Path,
        efas_user: str,
        efas_password: SecretStr,
        versions: Iterable[str] = ("eud",),
        fallback_versions: Mapping[str, str] | None = None,
        max_n_of_downloads: int = 1,
        cache: Mapping[Path, int] | None = None,
    ):
        if not data_dir.exists():
            raise IOError('Directory "{}" does not exist'.format(data_dir))
        if not data_dir.is_dir():
            raise IOError('Path "{}" is not a directory'.format(data_dir))

        if cache is None:
            efas_cache = {}
            efas_dir = Path(data_dir)
            if not efas_dir.exists():
                raise IOError('Directory "{}" does not exist'.format(efas_dir))
            if not efas_dir.is_dir():
                raise IOError('Path "{}" is not a directory'.format(efas_dir))
            for efas_file in efas_dir.iterdir():
                if not efas_file.is_file():
                    continue
                file_stat = efas_file.stat()
                file_size = file_stat.st_size
                efas_cache[Path(efas_file)] = file_size
        else:
            efas_cache = dict(cache)

        self.data_dir = data_dir
        self._user = efas_user
        self._password = efas_password

        self._versions = tuple(versions)
        self._fallback = dict(fallback_versions) if fallback_versions else {}
        self._max_n_of_downloads = int(max_n_of_downloads)
        self._cache = efas_cache

        self._semaphore = asyncio.Semaphore(self._max_n_of_downloads)

    @classmethod
    async def create(
        cls,
        efas_dir: PathLike,
        efas_user: str,
        efas_password: SecretStr,
        versions: Iterable[str] = ("eud",),
        fallback_versions: Mapping[str, str] | None = None,
        max_n_of_downloads: int = 1,
        cache: Mapping[Path, int] | None = None,
    ):
        """Asynchronous factory method for initializing an instance of this
        class.

        This method creates an object of the class while asynchronously
        populating the cache dictionary.
        """
        if cache is None:
            efas_cache = {}
            a_efas_dir = anyio.Path(efas_dir)
            if not await a_efas_dir.exists():
                raise IOError(
                    'Directory "{}" does not exist'.format(a_efas_dir)
                )
            if not await a_efas_dir.is_dir():
                raise IOError(
                    'Path "{}" is not a directory'.format(a_efas_dir)
                )
            async for efas_file in a_efas_dir.iterdir():
                if not await efas_file.is_file():
                    continue
                file_stat = await efas_file.stat()
                file_size = file_stat.st_size
                efas_cache[Path(efas_file)] = file_size
        else:
            efas_cache = dict(cache)

        return cls(
            Path(efas_dir),
            efas_user,
            efas_password,
            versions=versions,
            fallback_versions=fallback_versions,
            max_n_of_downloads=max_n_of_downloads,
            cache=efas_cache,
        )

    @staticmethod
    def _should_be_downloaded(
        file_name: str,
        available_files_on_server: dict[str, tuple[Path, int]],
        available_on_cache: dict[str, tuple[Path, int]],
    ) -> tuple[bool, Path | None]:
        """Determines whether a file on the server needs to be downloaded.

        This method implements the logic for deciding if a specified file
        should be downloaded. Several factors may lead to skipping the
        download, such as the file being unavailable on the server or the
        same information already existing in the local cache.

        The function returns a tuple of two elements:
        - The first element is a boolean indicating whether the file should be
          downloaded.
        - The second element of the tuple indicates whether the information in
          the file is already available. If available, this element contains
          the path to the local cached file with the desired information.
          If the data is missing, the element is 'None', signaling the need
          for a fallback strategy or a warning.

        If the first element of the returned tuple is `True`, the second one
        is always `None`.

        Args:
            file_name: The name of the file to evaluate.
            available_files_on_server: A mapping of file names to their paths
                and sizes on the remote server.
            available_on_cache: A mapping of file names to their paths and
                sizes in the local cache.

        Returns:
            A tuple of two elements where:
                - The first element is a boolean indicating whether the file
                  should be downloaded.
                - The second element is the `Path` to the current position of
                  the cached file (if it exists) or `None`.
        """
        logger = logging.getLogger(f"{__name__}.{inspect.stack()[0][3]}")

        logger.debug("Checking if a file named %s exists", file_name)
        if file_name not in available_files_on_server:
            logger.debug("File %s does not exist on the server", file_name)
            if file_name in available_on_cache:
                logger.debug(
                    "File %s does not exists on the server but it is "
                    "available on the cache; no need to download it",
                    file_name,
                )
                return False, available_on_cache[file_name][0]
            logger.debug(
                "File %s is missing; we will try to download a fallback",
                file_name,
            )
            return False, None

        file_server_path, remote_size = available_files_on_server[file_name]
        logger.debug(
            "File %s is available on server as %s and its size is %s",
            file_name,
            file_server_path,
            remote_size,
        )
        if file_name in available_on_cache:
            local_file_path, local_size = available_on_cache[file_name]
            logger.debug(
                "File %s is available into the cache as %s and its size is %s",
                file_name,
                local_file_path,
                local_size,
            )
            if remote_size != local_size:
                logger.info(
                    "The size of the file %s is different between the remote "
                    "FTP server (%s) and the local cache (%s); the file will "
                    "be rewritten",
                    file_name,
                    remote_size,
                    local_size,
                )
                return True, None
            logger.debug(
                "We already have a file named %s in cache with the same "
                "dimension; download is skipped",
                file_name,
            )
            return False, available_on_cache[file_name][0]

        logger.debug(
            "File %s is not available in the cache and will be downloaded",
            file_name,
        )
        return True, None

    async def _get_remote_server_available_files(
        self,
    ) -> dict[str, tuple[Path, int]]:
        """Checks which files are available on the server.

        Performs a `listdir` operation on the `EFAS_FTP_DIR` and retrieves the
        files stored in this directory. The method returns a dictionary that
        maps each file name to its corresponding path and size.

        Returns:
            A dictionary where keys are file names (str) and values are tuples
            of two elements such that:
            - the first element of the tuple is the file's path on the server.
            - the second element of the tuple is the file's size in bytes.
        """
        async with aioftp.Client.context(
            EFAS_FTP_URL,
            port=EFAS_FTP_PORT,
            user=self._user,
            password=self._password.get_secret_value(),
        ) as client:
            available_files = {
                p.name: (p, int(p_stat["size"]))
                for p, p_stat in await client.list(EFAS_FTP_DIR)
            }
        return available_files

    async def _single_download(
        self, remote_file_path: Path, retries: int = 5
    ) -> Path:
        """Downloads a single file from the remote server and saves it to the
        data directory.

        Args:
            remote_file_path: The path to the file on the remote server.
            retries: If the download fails, how many attempts to retry.

        Returns:
            The local path of the downloaded file.
        """
        logger = logging.getLogger(f"{__name__}.{inspect.stack()[0][3]}")
        logger.info("Downloading file %s", remote_file_path)
        for attempt in range(1, retries + 1):
            try:
                async with self._semaphore:
                    async with aioftp.Client.context(
                        EFAS_FTP_URL,
                        port=EFAS_FTP_PORT,
                        user=self._user,
                        password=self._password.get_secret_value(),
                        socket_timeout=30,
                        connection_timeout=30,
                    ) as client:
                        await client.download(remote_file_path, self.data_dir)
                break
            except aioftp.errors.StatusCodeError as e:
                if "425" in str(e) and attempt != retries:
                    logger.warning(
                        "Trying to download %s for the %d-th time due "
                        "to 425 error in the previous download",
                        attempt + 1,
                        remote_file_path,
                        exc_info=e,
                    )
                    await asyncio.sleep(attempt * 4)
                    continue
                raise

        output_path = self.data_dir / remote_file_path.name
        logger.info(
            "Download of file %s into %s completed",
            remote_file_path,
            output_path,
        )
        return output_path

    async def download_efas_operational_data(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> list[Path]:
        """Downloads all files from the FTP server for a specified time
        interval.

        This method constructs a list of expected file names and checks their
        availability on the remote server. Files that are available on the
        server and not present in the local cache will be downloaded.

        Args:
            start_time: The start of the time interval.
            end_time: The end of the time interval.
        """

        logger = logging.getLogger(f"{__name__}.{inspect.stack()[0][3]}")

        # We round start_time to the first multiple of 12th hours after
        # start_time
        start_day_midnight = datetime.combine(
            start_time.date(), time.min, start_time.tzinfo
        )
        if start_time > start_day_midnight:
            start_day_half_day = start_day_midnight + timedelta(hours=12)
            if start_time > start_day_half_day:
                start_time = start_day_midnight + timedelta(days=1)
            else:
                start_time = start_day_half_day

        logger.debug("Downloading EFAS NRT data starting from %s", start_time)

        time_steps = [start_time]
        while time_steps[-1] <= end_time:
            time_steps.append(time_steps[-1] + timedelta(hours=12))

        time_steps = tuple(time_steps)

        logger.debug(
            "We will try to download files for %s different times",
            len(time_steps),
        )

        available_files = await self._get_remote_server_available_files()
        available_in_cache = {
            p.name: (p, p_size) for p, p_size in self._cache.items()
        }

        returned_files = set()
        files_to_be_downloaded = set()
        for time_step in time_steps:
            time_str = time_step.strftime("%Y%m%d%H")
            for version in self._versions:
                expected_file_name = f"{version}.fc.dis_{time_str}.grb"
                logger.debug(
                    "Checking if a file named %s exists", expected_file_name
                )
                download_file, cache_position = self._should_be_downloaded(
                    expected_file_name, available_files, available_in_cache
                )
                if download_file:
                    files_to_be_downloaded.add(
                        available_files[expected_file_name][0]
                    )
                    continue

                if cache_position is None and version in self._fallback:
                    fallback_version = self._fallback[version]
                    fallback_file_name = (
                        f"{fallback_version}.fc.dis_{time_str}.grb"
                    )
                    logger.debug(
                        "Trying the fallback version %s: checking for file %s",
                        fallback_version,
                        fallback_file_name,
                    )
                    download_fallback, fallback_cache = (
                        self._should_be_downloaded(
                            fallback_file_name,
                            available_files,
                            available_in_cache,
                        )
                    )
                    if download_fallback:
                        files_to_be_downloaded.add(
                            available_files[fallback_file_name][0]
                        )
                        continue

                    if fallback_cache is not None:
                        returned_files.add(fallback_cache)
                    else:
                        logger.debug(
                            "Neither file %s not file %s exists on the "
                            "server; no file will be downloaded for "
                            "time-step %s",
                            expected_file_name,
                            fallback_file_name,
                            time_str,
                        )
                elif cache_position is None:
                    logger.debug(
                        "There is not fallback version for %s; no file will "
                        "be downloaded for time-step %s",
                        version,
                        time_str,
                    )
                else:
                    returned_files.add(cache_position)

        logger.info(
            "%s files will be downloaded from the EFAS FTP server",
            len(files_to_be_downloaded),
        )

        downloaded_files = await asyncio.gather(
            *(self._single_download(f) for f in files_to_be_downloaded),
            return_exceptions=True,
        )

        for file_remote_path, file_local_download in zip(
            files_to_be_downloaded, downloaded_files, strict=True
        ):
            if isinstance(file_local_download, Exception):
                logger.exception(
                    "Error while downloading file %s! This file has not been "
                    "downloaded! ",
                    file_remote_path,
                    exc_info=file_local_download,
                )
            else:
                returned_files.add(file_local_download)

        return sorted(list(returned_files))
