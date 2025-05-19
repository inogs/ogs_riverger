import inspect
import logging
from collections.abc import Iterable
from contextlib import ExitStack
from datetime import datetime
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal
from uuid import uuid1
from zipfile import ZipFile

import numpy as np
import xarray as xr

import ogs_riverger.efas.download_tools as efas_download
from ogs_riverger.efas.download_tools import EfasDataSource
from ogs_riverger.read_config import RiverConfigElement


class InvalidEfasFile(Exception):
    pass


def _get_best_unique_element(
    v: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Given an array `v`` that may contain repetitions of the same elements,
    returns a tuple of two arrays. The first one is a sorted array where every
    element of `v` appears just one, the second one is an array of indices
    `index` such that `v[index]` is equal to the first array we return. An
    index `i` of `index` is selected such that `weights[i] >= weights[j]` for
    every other `j` such that `v[i] == v[j]`.

    This function is similar to `np.unique(return_index=True)` and behaves in
    the same way if `weights == np.arange(len(v))[::-1]` (or any other
    array in descending order).
    """
    logger = logging.getLogger(f"{__name__}.{inspect.stack()[0][3]}")

    # Here we sort the v and the weights accordingly
    sorting_indices = np.argsort(v)
    weights = weights[sorting_indices]
    v = v[sorting_indices]

    # Now we check the position of each element of v; indeed,
    # unique_weights[i] contains the index of the first occurrence of the value
    # unique_v[i] inside the vector `v`; since valid_time is sorted,
    # from `unique_weights[i]` to `unique_weights[i + 1]` there are all the
    # values with a specific value. There we have to choose the one with
    # the maximum weight
    unique_v, unique_weights = np.unique(v, return_index=True)
    logger.debug(
        "There are %s unique values; from %s to %s",
        len(unique_v),
        unique_v[0],
        unique_v[-1],
    )

    # Here we save the indices of the elements that we select
    corresponding_indices = np.empty_like(unique_v, dtype=np.int32)

    # We convert unique_weights to a list because otherwise the static
    # typechecker of Pycharm wrongly reports an error
    unique_weights: list[int] = list(unique_weights)
    for i, i1 in enumerate(unique_weights):
        i2 = len(v) if i == len(unique_weights) - 1 else unique_weights[i + 1]

        logger.debug(
            "There are %s repetition of the element %s",
            i2 - i1,
            unique_v[i],
        )
        # From i1 to i2 we have the same value; now we choose
        # the index of the element with the maximum weight. This means
        # that weight[index_when_sorted] is the maximum weight for the value
        # we are considering. We also have to take into account that this
        # index is valid only after we have sorted the arrays. To retrieve the
        # original ones, we must use the "sorting_indices" array to go back
        # to the original position
        best_value_local_index = int(np.argmax(weights[i1:i2]))
        index_when_sorted = i1 + best_value_local_index

        corresponding_indices[i] = sorting_indices[index_when_sorted]

    return unique_v, corresponding_indices


def _find_slice(original: np.ndarray, sliced: np.ndarray) -> tuple[int, int]:
    """Finds the start and end indices of a slice obtained from a 1D array.

    This function takes an original 1D array `v` in ascending or descending
    order and another array `w` that has been obtained by performing a slice
    of `v` (i.e., `w = v[i:j]`). It calculates and returns the two indices
    `i` and `j` that have been used to get the slice `w` from the original
    array `v`.

    Args:
        original (np.ndarray): The original 1D array in ascending or descending
            order.
        sliced (np.ndarray): The sliced array obtained from the original array
            (w = v[i:j]).

    Returns:
        tuple[int, int]: A tuple containing the start index `i` and end
            index `j` used for the slice.

    Raises:
        ValueError: If the sliced array is not a valid slice of the original
            array.

    Example:
        >>> v = np.array([1, 2, 3, 4, 5])
        >>> w = v[1:4]
        >>> _find_slice(v, w)
        (1, 4)
    """
    if len(sliced) > len(original):
        raise ValueError(
            f"The size of the sliced array ({len(sliced)}) is greater than "
            f"the size of the original one ({len(original)})"
        )

    # We consider the first entry of sliced, and we look for the element of
    # `original` that is closest
    i = np.argmin(np.abs(original - sliced[0]))

    # Convert `i` to a standard integer (from a np.int32 or np.int64)
    i = int(i)

    # In this case, the original array ends before we found a corresponding
    # element for each value of the slice
    if i > len(original) - len(sliced):
        raise ValueError(
            f"The sliced array is not a valid slice of the original array. "
            f"The first element of the sliced array is {sliced[0]} but it "
            f"has position {i} of {len(original)} in the original array. "
            f"Since sliced has length {len(sliced)}, it can not be a valid "
            "slice of the original one."
        )

    j = i + sliced.size

    # Check if sliced is a valid slice of original
    if not np.allclose(original[i:j], sliced, atol=0.5e-3):
        raise ValueError(
            "The sliced array is not a valid slice of the original array."
        )

    return i, j


def generate_efas_domain_file(output_file: PathLike) -> Path:
    """Generate a single NetCDF4 file containing the coordinates of the entire
    EFAS domain.

    This file serves as a reference for identifying the indices of EFAS cells
    within subsets of the domain.
    It downloads data for a reference date, discards the content, and retains
    only the coordinates.

    Args:
        output_file: Path to the output file
    """
    reference_date = datetime(2018, 1, 1)

    logger = logging.getLogger(f"{__name__}.{inspect.stack()[0][3]}")

    with TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        downloaded_files = efas_download.download_from_cdsapi(
            service=EfasDataSource.HISTORICAL,
            start_date=reference_date,
            end_date=reference_date,
            output_dir=temp_dir,
            area=None,
            cdsapi_client=None,
            download_format=efas_download.EfasCEMSDownloadFormat.UNARCHIVED,
            file_data_format=efas_download.EfasCEMSDataFormat.GRIB,
        )

        if len(downloaded_files) != 1:
            raise AssertionError(
                f"The code is expected to download exactly one file; "
                f"{len(downloaded_files)} files have been downloaded"
            )
        downloaded_file = downloaded_files[0]

        with xr.open_dataset(downloaded_file, decode_timedelta=False) as ds:
            latitude = ds.latitude.values
            longitude = ds.longitude.values

    coords = xr.Dataset(coords={"latitude": latitude, "longitude": longitude})

    logger.info("Writing file %s", output_file)
    coords.to_netcdf(output_file, format="NETCDF4")

    return Path(output_file)


def read_efas_data_files(
    input_files: Iterable[Path],
    config_content: Iterable[RiverConfigElement],
    efas_domain_file: PathLike,
) -> xr.Dataset:
    """Reads and merges the content of multiple downloaded EFAS data files.

    EFAS data files are distributed as GRIB or NetCDF files, optionally
    compressed inside some zip archives.

    The function reads the different kinds of files, merges them,
    applies the runoff factor described in the CSV configuration file (if
    not specified otherwise), and returns an xArray dataset.

    Args:
        input_files: An iterable of paths to the input files
        config_content: An iterable that produces the configuration of each
            river we must consider
        efas_domain_file: The path of a file containing the coordinates of the
            efas domain (generated by the `generate_efas_domain_file` function)

    Returns:
        An Xarray dataset containing the EFAS rivers discharge. The
        dataset contains only one 2-dimensional variable named
        `dis06`. The first axis of the variable is the index of
        the river. The second one is the time.
        It also contains a one-dimensional variable named "computation_date"
        that contains the date of the computation.
    """
    # This function is just a wrapper that decompresses the (optionally)
    # zipped files and calls _read_unzipped_efas_files
    logger = logging.getLogger(f"{__name__}.{inspect.stack()[0][3]}")

    # Is there any zip file?
    zip_files = len([f for f in input_files if f.suffix.lower() == ".zip"]) > 0
    if not zip_files:
        logger.debug(
            "No zip files found in the list of files that must be read; "
            "they will be read as they are: %s",
            zip_files,
        )
        return _read_unzipped_efas_files(
            input_files,
            config_content,
            efas_domain_file,
        )

    logger.debug(
        "Some EFAS files are zipped and must be decompressed before being "
        "read."
    )
    uncompressed_files = []
    with TemporaryDirectory() as t:
        t = Path(t)
        logger.debug("Temporary directory %s will be used", t)

        for f in input_files:
            if f.suffix.lower() != ".zip":
                logger.debug(
                    "File %s will be read as is (already uncompressed)", f
                )
                uncompressed_files.append(f)
                continue

            decompressed_dir = t / (f.stem + "___" + uuid1().hex)
            decompressed_dir.mkdir(parents=False, exist_ok=False)
            logger.debug("Decompressing file %s into %s", f, decompressed_dir)
            with ZipFile(f, "r") as zip_ref:
                zip_ref.extractall(decompressed_dir)
                if len(zip_ref.namelist()) > 1:
                    raise ValueError(
                        f"The zip file {f} contains more than one file; "
                        "only one file is expected"
                    )
                if len(zip_ref.namelist()) == 0:
                    raise ValueError(
                        f"The zip file {f} does not contain any file"
                    )
                data_file_name = zip_ref.namelist()[0]
                zip_ref.extract(data_file_name, decompressed_dir)
                decompressed_data_file = decompressed_dir / data_file_name
                logger.debug(
                    "File %s has been decompressed into %s",
                    f,
                    decompressed_data_file,
                )
                uncompressed_files.append(decompressed_data_file)

        output_data = _read_unzipped_efas_files(
            uncompressed_files,
            config_content,
            efas_domain_file,
        )

    logger.debug("Directory %s has been deleted", t)
    return output_data


def _read_single_efas_file(dataset: xr.Dataset) -> xr.Dataset:
    """Determines the type of EFAS file and processes it accordingly.

    This function examines the structure and contents of the input dataset to
    identify its type. Based on the file characteristics, it calls the
    corresponding processing function, such as
    `_read_efas_operative_grib_file` for operative GRIB files or
    `_read_efas_historical_file` for historical NetCDF files.

    Args:
        dataset (xr.Dataset): An xarray dataset containing the EFAS data to be
            analyzed.

    Returns:
        xr.Dataset: The processed EFAS data in a standard format.

    Raises:
        InvalidEfasFile: If the file structure or its provided content does not
            match the expected EFAS formats.
    """
    logger = logging.getLogger(f"{__name__}.{inspect.stack()[0][3]}")

    if "id" not in dataset.dims:
        raise InvalidEfasFile(
            f'The current file has no "id" dimension; the current '
            f"dimensions are: {list(dataset.dims)}"
        )

    if len(dataset.dims) > 2:
        # Files with more than two dimensions usually are forecast files, where
        # we have two dimensions for the time; one is the time when the
        # computation has been performed, the other one is the time of the
        # model
        if set(dataset.dims) == {"id", "step", "time"}:
            return _read_efas_forecast_file(dataset)
        raise InvalidEfasFile(
            f"The current file has unexpected dimensions: {list(dataset.dims)}"
        )

    if "time" in dataset.coords and len(dataset["time"].dims) == 0:
        logger.debug(
            'The file has a coordinate "time" that contains only one element'
        )
        if "step" in dataset.dims:
            logger.debug(
                'The second dimension of the file (beside "id") is "step"; '
                "This file will be considered an operative grib file"
            )
            return _read_efas_operative_grib_file(dataset)

    if "valid_time" in dataset.coords and "valid_time" in dataset.dims:
        logger.debug(
            'The file has a coordinate "valid_time" that has the role of the '
            "time; it is a netcdf historical file"
        )
        return _read_efas_historical_file(dataset)

    if (
        "time" in dataset.coords
        and "time" in dataset.coords
        and "valid_time" in dataset.coords
    ):
        logger.debug(
            'The file has a coordinate named "time"; usually this means that '
            "is a GRIB historical file"
        )
        return _read_efas_historical_file(dataset)

    raise InvalidEfasFile(
        f"The EFAS file has a format that this software does not recognize:\n"
        f"{dataset}"
    )


def _read_efas_forecast_file(dataset: xr.Dataset):
    """Read the content of a forecast EFAS file.

    This function is called by `_read_single_efas_file` when the file has
    been identified as a forecast file.
    """
    logger = logging.getLogger(f"{__name__}.{inspect.stack()[0][3]}")
    # These files are the most difficult ones. They contain all the forecasts,
    # i.e., for the same time of the model we have the forecast produced 6
    # hours before, 18 hours before, and so on... We must remove this
    # redundancy by selecting only the most reliable forecast (the one that is
    # closest to the computation date).
    # "time" is the computational date
    # "valid_time" is the time of the model

    # Here we add an axis to "time" so that its shape is the same of
    # "valid_time" (this is the "step" dimension in the original dataset)
    time, valid_time = np.broadcast_arrays(
        dataset.time.values[:, np.newaxis], dataset.valid_time.values
    )

    # We remove the second dimension so that all the arrays are 1D now
    time = time.ravel()
    valid_time = valid_time.ravel()

    _, corresponding_indices = _get_best_unique_element(valid_time, time)

    logger.debug("Association completed! Now we perform the slicing...")
    # Here instead we unravel the indices, obtaining the correct indices
    # for the dataset
    time_indices, step_indices = np.unravel_index(
        corresponding_indices, dataset.valid_time.shape
    )

    # We slice the original data to take only the values that we have decided
    # that we want
    time_indices = xr.DataArray(time_indices, dims=("valid_time",))
    step_indices = xr.DataArray(step_indices, dims=("valid_time",))

    dataset_sliced = dataset.isel(time=time_indices, step=step_indices)

    # And then we can rename time -> computation_date and valid_time -> time
    logger.debug('Renaming "time" into "computation_date"...')
    dataset_sliced["computation_date"] = dataset_sliced["time"]
    del dataset_sliced["time"]

    logger.debug('Renaming "valid_time" into "time"...')
    dataset_sliced = dataset_sliced.rename({"valid_time": "time"})

    logger.debug("The dataset is read!")
    return dataset_sliced


def _read_efas_operative_grib_file(dataset: xr.Dataset):
    """Read the content of an operative EFAS file.

    This function is called by `_read_single_efas_file` when the file has
    been identified as an operative EFAS file downloaded from the FTP server.
    """
    logger = logging.getLogger(f"{__name__}.{inspect.stack()[0][3]}")

    logger.debug("Looking for the 2D variable that contains the data...")
    var_name = None
    for d_var in dataset.variables:
        if len(dataset.variables[d_var].dims) == 2:
            logger.debug('Found a 2d variable: "%s"', d_var)
            if var_name is not None:
                raise InvalidEfasFile(
                    f"The current file has more than one variable with two "
                    f'dimensions; these two variables are "{d_var}" and '
                    f'"{var_name}"'
                )
            var_name = d_var
    if var_name is None:
        raise InvalidEfasFile(
            "The current file has no variable with two dimensions"
        )
    logger.debug("The variable with the data is named %s ", var_name)

    computation_time = dataset.time.values
    logger.debug(
        "The time reference value for the file is %s", computation_time
    )

    times = dataset.step.values + computation_time

    # Let us broadcast computation_time to the same shape of "time"
    computation_time_array = np.empty_like(times, dtype=computation_time.dtype)
    computation_time_array[:] = computation_time

    if "valid_time" not in dataset.coords:
        raise InvalidEfasFile(
            'EFAS grib files are expected to have a "valid_time" coordinate'
        )

    if not np.array_equal(times, dataset.coords["valid_time"].values):
        raise InvalidEfasFile(
            "Expected times are different from the ones saved in the "
            '"valid_time" coordinate'
        )
    del dataset["time"]

    time = xr.DataArray(dataset["valid_time"].values, dims=("time",))

    dis06 = xr.DataArray(dataset[var_name].values.T, dims=("id", "time"))

    return xr.Dataset(
        {
            "latitude": dataset["latitude"],
            "longitude": dataset["longitude"],
            "dis06": dis06,
            "computation_date": xr.DataArray(
                computation_time_array, dims=("time",)
            ),
        },
        coords={"time": time, "id": dataset["id"]},
        attrs=dataset.attrs,
    )


def _read_efas_historical_file(dataset):
    """Read the content of an historical EFAS file.

    This function is called by `_read_single_efas_file` when the file has
    been identified as an historical file.
    """
    logger = logging.getLogger(f"{__name__}.{inspect.stack()[0][3]}")

    time_values = dataset.valid_time.values
    time = xr.DataArray(time_values, dims=("time",))

    if "time" in dataset.variables:
        # GRIB format
        logger.debug(
            'There is a variable named "time" in the file; probably it is a '
            "GRIB file"
        )
        computation_date = xr.DataArray(dataset.time.values, dims=("time",))
    else:
        # NETCDF format does not have the computation time
        logger.debug(
            'There is *NOT* a variable named "time" in the file; probably it '
            "is a netcdf file"
        )
        # We create a time variable by shifting the values of valid_time of 6
        # hours
        computation_date = xr.DataArray(
            time_values - np.datetime64(6 * 3600, "s"), dims=("time",)
        )

    logger.debug("Allocating the DataArray with the new values")
    dis06_dims = [
        d if d != "valid_time" else "time" for d in dataset["dis06"].dims
    ]
    dis06 = xr.DataArray(dataset["dis06"].values, dims=dis06_dims)

    logger.debug("Returning the dataset...")
    return xr.Dataset(
        {
            "latitude": dataset["latitude"],
            "longitude": dataset["longitude"],
            "dis06": dis06,
            "computation_date": computation_date,
        },
        coords={"time": time, "id": dataset["id"]},
        attrs=dataset.attrs,
    )


def _read_unzipped_efas_files(
    input_files: Iterable[Path],
    config_content: Iterable[RiverConfigElement],
    efas_domain_file: PathLike,
) -> xr.Dataset:
    """Reads and processes unzipped EFAS (European Flood Awareness System)
    files to extract data based on specified river locations, reshaping the
    data into a consumable format.

    This function takes EFAS files, configuration content, and the EFAS domain
    file as input.
    For each file provided, it extracts data based on configured river indices,
    validates that data aligns with the specified domain, and reshapes the
    output. The resulting datasets are concatenated along the time dimension,
    retaining the best forecast data, and returned as an xarray.Dataset.

    Args:
        input_files: A collection of file paths indicating the
            locations of the unzipped EFAS files to process.
        config_content: An iterable that generates the configuration for
            each EFAS river, including indices of river locations.
        efas_domain_file: A path to the EFAS domain file used for latitude and
            longitude referencing.

    Returns:
        xr.Dataset: A concatenated dataset containing processed EFAS data,
            reshaped and indexed by river and time.

    Raises:
        InvalidEfasFile: If a file's domain does not match the specified
            river indices, or if there are issues while reading a specific
            EFAS file.
    """
    logger = logging.getLogger(f"{__name__}.{inspect.stack()[0][3]}")

    input_files = tuple(input_files)
    if len(input_files) == 0:
        raise ValueError(
            "No input files were provided. This function needs at least one "
            "EFAS file to be processed."
        )
    river_configs = tuple(
        c for c in config_content if c.data_source.type == "EFAS"
    )
    n_rivers = len(river_configs)

    # Get the ids, latitude and longitude of the rivers
    river_ids = np.empty((n_rivers,), dtype=int)
    river_names = []
    lon_indices_array = np.empty_like(river_ids)
    lat_indices_array = np.empty_like(river_ids)
    for i, river in enumerate(river_configs):
        river_ids[i] = river.id
        river_names.append(river.name)
        lon_indices_array[i] = river.data_source.longitude_index
        lat_indices_array[i] = river.data_source.latitude_index

    # Transform the arrays with the indices of the latitude and the longitude
    # into two DataArrays
    lon_indices = xr.DataArray(
        lon_indices_array,
        dims=["id"],
        coords={"id": river_ids},
    )
    lat_indices = xr.DataArray(
        lat_indices_array,
        dims=["id"],
        coords={"id": river_ids},
    )

    # Open the file with the coordinates of the overall domain to understand
    # which part of the domain we downloaded
    with xr.open_dataset(efas_domain_file) as f:
        domain_latitudes = f.latitude.values
        domain_longitudes = f.longitude.values

    datasets = []
    for file_path in input_files:
        # This file could be a grib or a NetCDF; luckily, xarray supports both
        logger.debug('Opening file "%s"', file_path)
        # We use Dask here (chunks={}) because it is way more efficient than
        # standard xarray when executing the isel method.
        # By setting "decode_timedelta=True" we ensure that the values of the
        # step variable are decoded as timedelta64 objects (and we also
        # silence a warning)
        with xr.open_dataset(
            file_path, chunks={}, decode_timedelta=True
        ) as single_ds:
            dataset_latitudes = single_ds.latitude.values
            dataset_longitudes = single_ds.longitude.values

            i_lat1, i_lat2 = _find_slice(domain_latitudes, dataset_latitudes)
            i_lon1, i_lon2 = _find_slice(domain_longitudes, dataset_longitudes)

            # Check that the position of the rivers is coherent with the
            # domain of the file we have downloaded
            outside_lat = np.logical_or(
                lat_indices < i_lat1, lat_indices >= i_lat2
            )
            outside_lon = np.logical_or(
                lon_indices < i_lon1, lon_indices >= i_lon2
            )
            if np.any(outside_lat) or np.any(outside_lon):
                river_outside_index = np.nonzero(
                    (outside_lon | outside_lat).values
                )[0][0]
                river_name = river_names[river_outside_index]
                river_latitude = lat_indices.values[river_outside_index]
                river_longitude = lon_indices.values[river_outside_index]

                lat_sorted = np.sort(dataset_latitudes)
                lon_sorted = np.sort(dataset_longitudes)
                raise InvalidEfasFile(
                    f'The domain of the file "{file_path}" (latitudes '
                    f"from {lat_sorted[0]:.3f} to {lat_sorted[-1]:.3f} "
                    f"and longitudes from {lon_sorted[0]:.3f} to "
                    f"{lon_sorted[-1]:.3f}) does not contain the river "
                    f'"{river_name}", whose mouth has latitude '
                    f"{domain_latitudes[river_latitude]:.3f} and "
                    f"longitude {domain_longitudes[river_longitude]:.3f}."
                )

            logger.debug("Retrieving rivers' data from the map")
            file_content = single_ds.isel(
                longitude=lon_indices - i_lon1,
                latitude=lat_indices - i_lat1,
            ).load()

        # Remove the surface variable
        if "surface" in file_content:
            logger.debug('Removing coordinate "surface"')
            del file_content["surface"]

        try:
            original_dataset = _read_single_efas_file(file_content)
        except InvalidEfasFile as exception:
            raise InvalidEfasFile(
                f'Error while reading EFAS file "{file_path}"'
            ) from exception

        logger.debug(
            "Adding a new dataset to the collection: %s file have been read",
            len(datasets) + 1,
        )
        datasets.append(original_dataset)

    logger.debug("Concatenating the datasets...")
    concat_dataset = xr.concat(datasets, dim="time")

    logger.debug("Getting the best forecast (if needed)")
    _, slicing_indices = _get_best_unique_element(
        concat_dataset.time.values, concat_dataset.computation_date.values
    )

    efas_data = concat_dataset.isel(time=slicing_indices)

    return efas_data


def generate_efas_climatology(
    annual_files: dict[int, Path],
    config_content: Iterable[RiverConfigElement],
    output_file: Path | str,
    efas_domain_file: PathLike | None = None,
) -> None:
    """Generates a climatology for the EFAS dataset.

    This function writes a NetCDF file containing the climatological
    values (averaged across years) for each day of the year of the
    input files submitted.

    Args:
        annual_files: a dictionary that maps each year to the corresponding
            EFAS data archive
        config_content: An iterable that produces a RiverConfigElement for each
            river we must keep in our configuration
        output_file: The path to the output NetCDF file
        efas_domain_file: Optional path to the domain file of the EFAS

    Raises:
        ValueError: if the keys of the annual_files dictionary do not
            represent a continuous range of years (for example, from
            2012 to 2020)
    """
    logger = logging.getLogger(f"{__name__}.{inspect.stack()[0][3]}")

    with ExitStack() as cleanup_operations:
        if efas_domain_file is None:
            t = Path(cleanup_operations.enter_context(TemporaryDirectory()))
            efas_domain_file = t / "efas_domain.nc"
            generate_efas_domain_file(efas_domain_file)
        else:
            efas_domain_file = Path(efas_domain_file)

        years = tuple(sorted(annual_files.keys()))
        # We check if there is a "hole" in the dictionary that associates the
        # years with the respective files. If this is the case, we check which
        # one is the missing file and we raise an error
        if years != tuple(range(years[0], years[-1] + 1)):
            missing_year = next(
                y for y in range(years[0], years[-1] + 1) if y not in years
            )
            raise ValueError(
                "annual files shall contain one file for each year from {} "
                "to {}; year {} is missing".format(
                    years[0], years[-1], missing_year
                )
            )

        logger.debug(
            "%s files will be used to create the climatology (from %s to %s)",
            len(years),
            years[0],
            years[-1],
        )

        # After we execute this function, we can delete the efas_domain_file if
        # it was generated on the fly. Therefore, we close the context manager
        efas_data = read_efas_data_files(
            [annual_files[y] for y in years],
            config_content,
            efas_domain_file=efas_domain_file,
        )

    # Compute the daily average
    times = np.array(efas_data.time, dtype="datetime64[s]")
    # noinspection PyTypeChecker
    first_timestep: datetime = times[0].astype(datetime)
    # noinspection PyTypeChecker
    last_timestep: datetime = times[-1].astype(datetime)

    # As of the new CDS API update (September 26, 2024), the data
    # slicing method has changed.
    # For example, when requesting data for September 10, 2024,
    # instead of getting four time steps starting at midnight (00:00) on
    # the 10th, followed by 06:00, 12:00, and 18:00 of the same day,
    # the time steps now include 06:00, 12:00, 18:00 on September 10th
    # and midnight (00:00) of September 11th. Therefore, we adjust the
    # "left" and "right" boundaries to align with this new format.
    closed: Literal["left", "right"] = "left"
    if first_timestep.hour > 0 and last_timestep.hour == 0:
        closed = "right"

    logger.debug("Computing daily means as %s", closed)
    daily_dataset = efas_data.resample(time="1D", closed=closed).mean()

    logger.debug('Renaming variable "dis06" as "discharge"')
    daily_dataset["discharge"] = daily_dataset["dis06"]
    del daily_dataset["dis06"]

    # Compute month and day of each time
    daily_dataset = daily_dataset.assign_coords(
        month=("time", daily_dataset["time"].dt.month.values),
        day=("time", daily_dataset["time"].dt.day.values),
    )

    # This number identifies the same day among different years
    unique_date = daily_dataset.month * 100 + daily_dataset.day

    logger.debug("Computing means for each date")
    climatological_data = daily_dataset.groupby(unique_date).mean("time")
    climatological_data = climatological_data.rename({"group": "date"})

    climatological_data = climatological_data.assign_coords(
        month=("date", climatological_data.date.values // 100),
        day=("date", climatological_data.date.values % 100),
    )

    # Remove the date; we already have month and day
    climatological_data = climatological_data.drop_vars("date")

    logger.debug("Saving file %s", output_file)
    climatological_data.to_netcdf(
        output_file,
        encoding={"discharge": {"compression": "zlib", "complevel": 9}},
    )
