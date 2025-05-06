import pytest

from .fixtures import config_example  # noqa: F401
from .fixtures import config_file_example  # noqa: F401
from .fixtures import settings  # noqa: F401
from .fixtures import test_data_dir  # noqa: F401


def pytest_addoption(parser):
    parser.addoption(
        "--external-resources",
        action="store_true",
        dest="external-resources",
        default=False,
        help="enable tests about rivers that require external connections",
    )


def pytest_collection_modifyitems(config, items):
    # If "external-resources" is not submitted, then all the tests that are
    # marked with that keyword must be skipped
    if not config.getoption("--external-resources"):
        skipper = pytest.mark.skip(
            reason="Only run when --external-resources is given"
        )
        for item in items:
            print(list(item.keywords))
            if "external_resources" in item.keywords:
                item.add_marker(skipper)
