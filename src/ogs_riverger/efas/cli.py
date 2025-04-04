import argparse
import asyncio
import inspect
import logging
import re
from pathlib import Path
from tempfile import TemporaryDirectory

import ogs_riverger.efas.download_tools as efas_download
from ogs_riverger.efas.efas_config import read_efas_config_file
from ogs_riverger.efas.efas_manager import generate_efas_climatology
from ogs_riverger.efas.efas_manager import generate_efas_domain_file
from ogs_riverger.efas.efas_manager import read_efas_data_files
from ogs_riverger.settings import Settings
from ogs_riverger.utils.argparse_types import date_from_str
from ogs_riverger.utils.argparse_types import dir_to_be_created_if_not_exists
from ogs_riverger.utils.argparse_types import existing_dir_path
from ogs_riverger.utils.argparse_types import existing_file_path
from ogs_riverger.utils.argparse_types import path_inside_an_existing_dir

EFAS_COMMAND = "efas"


def add_efas_subcommands(subparsers):
    """Adds the efas section to the main CLI.

    Args:
        subparsers: The subparsers obtained by calling the method
            `add_subparsers` of the main parser
    """

    efas_parser = subparsers.add_parser(
        EFAS_COMMAND,
        help="Retrieve information about the rivers from the EFAS "
        "(European Flood Awareness System) data",
    )

    def define_bb_box(parser):
        parser.add_argument(
            "--bb-north",
            type=float,
            required=False,
            default=None,
            help="North boundary of the bounding box (in latitude degrees).",
        )
        parser.add_argument(
            "--bb-south",
            type=float,
            required=False,
            default=None,
            help="South boundary of the bounding box (in latitude degrees).",
        )
        parser.add_argument(
            "--bb-west",
            type=float,
            required=False,
            default=None,
            help="West boundary of the bounding box (in longitude degrees).",
        )
        parser.add_argument(
            "--bb-east",
            type=float,
            required=False,
            default=None,
            help="East boundary of the bounding box (in longitude degrees).",
        )

    subparsers = efas_parser.add_subparsers(dest="efas_action", required=True)

    download_forecast = subparsers.add_parser(
        "download-forecast",
        help="Downloads the forecast from the EFAS archive",
    )
    download_forecast.add_argument(
        "--start-date",
        "-s",
        type=date_from_str,
        required=True,
        help="The first day of the forecast",
    )
    download_forecast.add_argument(
        "--end-date",
        "-e",
        type=date_from_str,
        required=True,
        help="The last day of the forecast",
    )
    download_forecast.add_argument(
        "--output-dir",
        "-o",
        type=dir_to_be_created_if_not_exists,
        required=True,
        help="The path of the directory where the output files will be saved",
    )

    define_bb_box(download_forecast)

    download_climatology = subparsers.add_parser(
        "download-climatology",
        help="Download one file per year from the historical archive, "
        "typically used to create a climatology dataset.",
    )
    download_climatology.add_argument(
        "--first-year",
        "-s",
        type=int,
        required=True,
        help="The first year of the climatology",
    )
    download_climatology.add_argument(
        "--last-year",
        "-e",
        type=int,
        required=True,
        help="The last year of the climatology",
    )
    download_climatology.add_argument(
        "--output-dir",
        "-o",
        type=dir_to_be_created_if_not_exists,
        required=True,
        help="The path of the directory where the output files will be saved",
    )

    define_bb_box(download_climatology)

    convert_data = subparsers.add_parser(
        "convert-data",
        help="Convert downloaded EFAS files (from the "
        '"download-forecast", "download-operative", or '
        '"download-climatology" commands) into a single NetCDF4 file '
        "that can be opened with Xarray, with one value for each river",
    )

    convert_data.add_argument(
        "--input-files",
        "-i",
        type=existing_file_path,
        nargs="+",
        required=True,
        help="Specify one or more EFAS files downloaded from the archive "
        '(e.g., using the "download-forecast" command). Multiple files '
        "can be provided.",
    )

    convert_data.add_argument(
        "--output-file",
        "-o",
        type=path_inside_an_existing_dir,
        required=True,
        help="Specify the path for the output NetCDF4 file.",
    )

    convert_data.add_argument(
        "--config-file",
        "-c",
        type=existing_file_path,
        required=True,
        help="Provide the path to the JSON configuration file describing the "
        "rivers dataset.",
    )

    convert_data.add_argument(
        "--domain-file",
        "-d",
        type=existing_file_path,
        required=True,
        help="Specify the path to the EFAS domain file, which must be "
        'generated using the "generate-domain-file" command.',
    )

    generate_climatology = subparsers.add_parser(
        "generate-climatology",
        help="Given a directory containing historical EFAS files (one per "
        "year), generate a single file that stores the climatology of "
        "the dataset (i.e., the average across years for each date).",
    )

    generate_climatology.add_argument(
        "--input-dir",
        "-i",
        type=existing_dir_path,
        required=True,
        help="The path of the directory where the annual data files have "
        "been downloaded (usually, using the command "
        '"download-climatology")',
    )
    generate_climatology.add_argument(
        "--output-file",
        "-o",
        type=path_inside_an_existing_dir,
        required=True,
        help="The path of the output file",
    )

    generate_climatology.add_argument(
        "--config-file",
        "-c",
        type=existing_file_path,
        required=True,
        help="Provide the path to the JSON configuration file describing the "
        "rivers dataset.",
    )

    generate_climatology.add_argument(
        "--domain-file",
        "-d",
        type=existing_file_path,
        required=True,
        help="""
            The path of the EFAS domain file (generated using the command "
            generate-domain-file).
        """,
    )

    download_generate_climatology = subparsers.add_parser(
        "download-generate-climatology",
        help="This is equivalent to calling 'download-climatology' in a "
        "temporary directory, followed by 'generate-climatology' on the same "
        "directory. The downloaded files are deleted before the execution "
        "completes.",
    )

    download_generate_climatology.add_argument(
        "--first-year",
        "-s",
        type=int,
        required=True,
        help="The first year of the climatology",
    )
    download_generate_climatology.add_argument(
        "--last-year",
        "-e",
        type=int,
        required=True,
        help="The last year of the climatology",
    )

    download_generate_climatology.add_argument(
        "--output-file",
        "-o",
        type=path_inside_an_existing_dir,
        required=True,
        help="The path of the output file",
    )
    download_generate_climatology.add_argument(
        "--config-file",
        "-c",
        type=existing_file_path,
        required=True,
        help="Provide the path to the JSON configuration file describing the "
        "rivers dataset.",
    )

    download_generate_climatology.add_argument(
        "--domain-file",
        "-d",
        type=existing_file_path,
        required=True,
        help="""
            The path of the EFAS domain file (generated using the command "
            generate-domain-file).
        """,
    )

    define_bb_box(download_generate_climatology)

    download_operational = subparsers.add_parser(
        "download-operational",
        help="Downloads the operational files from the EFAS FTP server",
    )
    download_operational.add_argument(
        "--start-date",
        "-s",
        type=date_from_str,
        required=True,
        help="The first day of the data",
    )
    download_operational.add_argument(
        "--end-date",
        "-e",
        type=date_from_str,
        required=True,
        help="The last day of the data",
    )
    download_operational.add_argument(
        "--output-dir",
        "-o",
        type=dir_to_be_created_if_not_exists,
        required=True,
        help="The path of the directory where the output files will be saved",
    )
    download_operational.add_argument(
        "--n-downloads",
        "-n",
        type=int,
        required=False,
        default=1,
        help="The maximum number of concurrent downloads",
    )

    generate_domain_file = subparsers.add_parser(
        "generate-domain-file",
        help="Generates a files that contain the position of the points of "
        "the entire EFAS domain. This file will be used later to "
        "identify the position of the files that contain only a subset "
        "of the entire domain",
    )

    generate_domain_file.add_argument(
        "--output-file",
        "-o",
        type=path_inside_an_existing_dir,
        required=True,
        help="The path of the generated EFAS domain file",
    )


def efas_cli(args: argparse.Namespace, settings: Settings) -> int:
    """Executes the commands of the EFAS CLI.

    Args:
        args: the parsed command line arguments
        settings: The settings object containing the users and passwords that
          must be used to access the remote servers.

    Returns:
        The exit status of the function
    """
    logger = logging.getLogger(f"{__name__}.{inspect.stack()[0][3]}")

    action = args.efas_action
    logger.info('Executing action "%s"', action)

    match action:
        case "download-forecast":
            logger.info("Downloading forcast files...")
            bb_box = efas_download.build_efas_area_selection(
                north=args.bb_north,
                south=args.bb_south,
                west=args.bb_west,
                east=args.bb_east,
            )
            logger.debug("Using the following bounding box: %s", bb_box)

            client = efas_download.get_cdsapi_client(
                url=settings.CDS_API_URL,
                key=settings.CDS_API_KEY,
            )
            efas_download.download_from_cdsapi(
                service=efas_download.EfasDataSource.FORECAST,
                start_date=args.start_date,
                end_date=args.end_date,
                output_dir=args.output_dir,
                area=bb_box,
                cdsapi_client=client,
            )
            logger.info("Done")
        case "download-climatology":
            logger.info("Downloading climatology files...")
            bb_box = efas_download.build_efas_area_selection(
                north=args.bb_north,
                south=args.bb_south,
                west=args.bb_west,
                east=args.bb_east,
            )
            client = efas_download.get_cdsapi_client(
                url=settings.CDS_API_URL,
                key=settings.CDS_API_KEY,
            )
            for year in range(args.first_year, args.last_year + 1):
                output_path = (
                    args.output_dir / f"efas_historical_{year}.netcdf4.zip"
                )
                logger.info(
                    "Downloading historical file for year %s into %s",
                    year,
                    output_path,
                )
                efas_download.download_yearly_data_from_cdsapi(
                    service=efas_download.EfasDataSource.HISTORICAL,
                    year=year,
                    output_file=output_path,
                    area=bb_box,
                    cdsapi_client=client,
                )
                logger.info("File %s downloaded!", output_path)
            logger.info("Done")
        case "convert-data":
            logger.info(
                "The following files will be processed: %s",
                ", ".join([str(p) for p in args.input_files]),
            )
            config_content = read_efas_config_file(args.config_file)
            river_dataset = read_efas_data_files(
                args.input_files,
                config_content,
                efas_domain_file=args.domain_file,
            )

            logger.info("Saving file %s", args.output_file)
            river_dataset.to_netcdf(args.output_file)
            logger.info("Done")
        case "generate-climatology":
            climatology_files = {}
            mask = re.compile(
                r"^efas_historical_(?P<year>[0-9]{4}).netcdf4.zip$"
            )
            for f in args.input_dir.iterdir():
                if f.is_dir():
                    continue
                mask_match = mask.match(f.name)
                if mask_match is None:
                    continue
                year = int(mask_match.group("year"))
                climatology_files[year] = f

            config_content = read_efas_config_file(args.config_file)

            if len(climatology_files) == 0:
                logger.error("No files found in directory %s", args.input_dir)
                return 1
            generate_efas_climatology(
                climatology_files,
                config_content,
                args.output_file,
                efas_domain_file=args.domain_file,
            )
            logger.info("Done")
        case "download-generate-climatology":
            config_content = read_efas_config_file(args.config_file)
            bb_box = efas_download.build_efas_area_selection(
                north=args.bb_north,
                south=args.bb_south,
                west=args.bb_west,
                east=args.bb_east,
            )
            client = efas_download.get_cdsapi_client(
                url=settings.CDS_API_URL,
                key=settings.CDS_API_KEY,
            )

            with TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)
                downloaded_files = {}
                for year in range(args.first_year, args.last_year + 1):
                    current_download_path = (
                        temp_dir / f"efas_historical_{year}.netcdf4.zip"
                    )
                    logger.info(
                        "Downloading historical file for year %s into %s",
                        year,
                        current_download_path,
                    )
                    efas_download.download_yearly_data_from_cdsapi(
                        service=efas_download.EfasDataSource.HISTORICAL,
                        year=year,
                        output_file=current_download_path,
                        area=bb_box,
                        cdsapi_client=client,
                    )
                    downloaded_files[year] = current_download_path
                    logger.info("File %s downloaded!", current_download_path)

                generate_efas_climatology(
                    downloaded_files, config_content, args.output_file
                )
            logger.info("Done")
        case "download-operational":
            if settings.EFAS_FTP_USER is None:
                raise ValueError(
                    "The environment variable 'EFAS_FTP_USER' is not set. "
                    "Please export an environment variable named "
                    '"EFAS_FTP_USER" with the username for the '
                    "EFAS FTP server."
                )
            if settings.EFAS_FTP_PASSWORD is None:
                raise ValueError(
                    "The environment variable 'EFAS_FTP_PASSWORD' is not set. "
                    "Please export an environment variable named "
                    '"EFAS_FTP_PASSWORD" with the password for the '
                    "EFAS FTP server."
                )

            async def download_operational():
                efas_downloader = (
                    await efas_download.EfasOperationalDownloader.create(
                        args.output_dir,
                        settings.EFAS_FTP_USER,
                        settings.EFAS_FTP_PASSWORD,
                        fallback_versions={"eud": "dwd"},
                        max_n_of_downloads=args.n_downloads,
                    )
                )

                await efas_downloader.download_efas_operational_data(
                    start_time=args.start_date,
                    end_time=args.end_date,
                )

            asyncio.run(download_operational())
        case "generate-domain-file":
            output_file = args.output_file
            logger.info("Generating EFAS domain file")
            generate_efas_domain_file(output_file)
            logger.info("Done")

    return 0
