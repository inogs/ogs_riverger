import os
from pathlib import Path

import pytest

from ogs_riverger.settings import Settings


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


@pytest.fixture
def test_data_dir():
    return Path(os.path.dirname(__file__)) / "data"


@pytest.fixture(scope="session")
def settings():
    return Settings()
