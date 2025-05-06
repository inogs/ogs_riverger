from ogs_riverger.read_config import RiverConfig


def test_read_config(config_file_example):
    river_config = RiverConfig.from_json(config_file_example)
    assert len(river_config.root) == 6
