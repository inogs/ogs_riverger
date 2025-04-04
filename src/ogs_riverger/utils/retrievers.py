import inspect
import logging
from collections.abc import Awaitable
from io import BytesIO
from pathlib import Path
from typing import Any
from typing import Callable

from aiofiles import open as aio_open
from httpx import AsyncClient
from httpx import HTTPStatusError


async def http_retrieve(
    *,
    client: AsyncClient,
    url: str,
    params: dict[str, Any] | None = None,
    dest: Callable[[bytes], Awaitable[Any]],
) -> None:
    """
    Asynchronously retrieves data from a specified URL using a given HTTP
    client. The response data is streamed in chunks, and each chunk is
    processed by the provided callback function.
    The function ensures proper handling of HTTP status codes and logs
    the activity accordingly.

    Args:
        client: An instance of `AsyncClient` used to perform the
            HTTP request.
        url: The URL to send the GET request to.
        params: Optional dictionary containing query parameters to include
            in the request.
        dest: An asynchronous callback function that processes the
            chunks of the streamed response.
    """
    logger = logging.getLogger(f"{__name__}.{inspect.stack()[0][3]}")
    logger.debug("Async download of %s with params=%s", url, params)
    async with client.stream("GET", url=url, params=params) as stream:
        if stream.status_code != 200:
            logger.error(
                "Error fetching %s (status: %d): %s",
                stream.url,
                stream.status_code,
                (await stream.aread()).decode("utf-8", errors="ignore"),
            )
            raise HTTPStatusError(
                f"Stream returned status code {stream.status_code}.",
                request=stream.request,
                response=stream,
            )
        async for chunk in stream.aiter_bytes():
            await dest(chunk)


async def http_retrieve_on_file(
    *,
    client: AsyncClient,
    url: str,
    params: dict[str, Any] | None = None,
    dest: Path,
) -> None:
    """
    Asynchronously retrieves content from a specified HTTP URL and writes it
    to a file.

    This coroutine function performs an HTTP request to retrieve the content
    from a given URL and writes it directly to the specified destination file.
    The file is opened in binary write mode to accommodate binary data,
    and chunks of data received from the HTTP response are incrementally
    written to the file. The HTTP request is managed by the passed AsyncClient
    instance.

    Args:
        client: The asynchronous HTTP client used to make the request.
        url: The URL from which to retrieve the content.
        params: Optional query parameters to include in the request.
        dest: The file system path where the retrieved content will be stored.
    """
    logger = logging.getLogger(f"{__name__}.{inspect.stack()[0][3]}")
    logger.debug("Opening file %s", dest)
    async with aio_open(dest, "wb") as f:

        async def write_to_file(chunk: bytes) -> None:
            await f.write(chunk)

        await http_retrieve(
            client=client,
            url=url,
            params=params,
            dest=write_to_file,
        )


async def http_retrieve_in_memory(
    *,
    client: AsyncClient,
    url: str,
    params: dict[str, Any] | None = None,
) -> bytes:
    """
    Downloads data from the specified URL into an in-memory buffer instead
    of a file.

    Args:
        client (AsyncClient): The HTTP client to use for the request.
        url (str): The URL to fetch the content from.
        params (dict[str, Any] | None): Optional query parameters for the
        request.

    Returns:
        bytes: The downloaded content stored in memory.
    """
    logger = logging.getLogger(f"{__name__}.{inspect.stack()[0][3]}")
    logger.debug(
        "Starting in-memory download of %s with params=%s", url, params
    )

    buffer = BytesIO()

    async def write_to_buffer(chunk: bytes) -> None:
        buffer.write(chunk)

    await http_retrieve(
        client=client,
        url=url,
        params=params,
        dest=write_to_buffer,
    )

    logger.debug("Download complete, returning in-memory content")
    return buffer.getvalue()
