import dataclasses
import tomllib

from ecg_annotator.config import AppConfig, DEFAULTS, _SAMPLE_CONFIG, load_config


def test_defaults_returned_when_no_config_file_exists(tmp_path, monkeypatch):
    monkeypatch.setattr("ecg_annotator.config.CONFIG_FILE", tmp_path / "config.toml")
    config = load_config()
    assert config == DEFAULTS


def test_user_values_override_defaults(tmp_path, monkeypatch):
    config_file = tmp_path / "config.toml"
    config_file.write_text('distance_threshold = 0.1\namplitude_scale = 5.0\n')
    monkeypatch.setattr("ecg_annotator.config.CONFIG_FILE", config_file)

    config = load_config()
    assert config.distance_threshold == 0.1
    assert config.amplitude_scale == 5.0


def test_partial_config_falls_back_to_defaults(tmp_path, monkeypatch):
    config_file = tmp_path / "config.toml"
    config_file.write_text('distance_threshold = 0.2\n')
    monkeypatch.setattr("ecg_annotator.config.CONFIG_FILE", config_file)

    config = load_config()
    assert config.distance_threshold == 0.2
    assert config.default_confidence == DEFAULTS.default_confidence
    assert config.amplitude_scale == DEFAULTS.amplitude_scale


def test_sample_config_is_valid_toml():
    parsed = tomllib.loads(_SAMPLE_CONFIG)
    expected_keys = {f.name for f in dataclasses.fields(AppConfig)}
    assert set(parsed.keys()) == expected_keys
