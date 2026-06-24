"""Lazy proxy for xarray.

Importing this module does *not* import xarray. The real import happens only
when an attribute (e.g. ``xr.Dataset``, ``xr.open_dataset``) is first
accessed, via the module-level ``__getattr__`` hook defined in PEP 562.

Usage::

    import ogs_riverger.utils.lazy_xarray as xr

    ds = xr.open_dataset(...)   # xarray is imported here, on first access
"""

_XARRAY = None


def __getattr__(name: str):
    global _XARRAY
    if _XARRAY is None:
        import xarray

        _XARRAY = xarray

    return getattr(_XARRAY, name)
