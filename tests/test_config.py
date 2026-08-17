import pytest

from config import MAX_FRAMES, ConfigError, Settings


def test_defaults_match_spec():
    settings = Settings.from_env({})
    assert settings.model_repo_id == "MiniMaxAI/MiniMax-Music3"
    assert settings.hf_home == "/runpod-volume/huggingface-cache"
    assert settings.gpu_count == 2
    assert settings.sgl_port == 8000
    assert settings.server_startup_timeout_s == 1200.0
    assert settings.generation_timeout_s == 1500.0
    assert settings.max_concurrency == 1
    assert settings.max_duration_s == 360.0
    assert settings.default_format == "mp3"
    assert settings.default_bitrate == "192k"
    assert settings.base64_max_encoded_bytes == 9_500_000


def test_base_url_is_localhost_with_port():
    settings = Settings.from_env({"SGL_PORT": "9001"})
    assert settings.base_url == "http://127.0.0.1:9001"


def test_extra_args_split_on_whitespace():
    settings = Settings.from_env({"SGL_EXTRA_ARGS": "--max-running-requests 32"})
    assert settings.sgl_extra_args == ("--max-running-requests", "32")


def test_bucket_configured_requires_all_three_values():
    partial = Settings.from_env({"BUCKET_ENDPOINT_URL": "https://example.com"})
    assert partial.bucket_configured is False
    full = Settings.from_env(
        {
            "BUCKET_ENDPOINT_URL": "https://example.com",
            "BUCKET_ACCESS_KEY_ID": "key",
            "BUCKET_SECRET_ACCESS_KEY": "secret",
        }
    )
    assert full.bucket_configured is True


def test_non_numeric_value_is_a_config_error():
    with pytest.raises(ConfigError, match="SGL_PORT"):
        Settings.from_env({"SGL_PORT": "not-a-number"})


def test_gpu_count_must_be_positive():
    with pytest.raises(ConfigError, match="GPU_COUNT"):
        Settings.from_env({"GPU_COUNT": "0"})


def test_max_duration_cannot_exceed_model_limit():
    limit = MAX_FRAMES / 25
    with pytest.raises(ConfigError, match="MAX_DURATION_S"):
        Settings.from_env({"MAX_DURATION_S": str(limit + 1)})


def test_default_format_must_be_supported():
    with pytest.raises(ConfigError, match="DEFAULT_FORMAT"):
        Settings.from_env({"DEFAULT_FORMAT": "aiff"})


def test_hub_download_is_off_by_default():
    assert Settings.from_env({}).allow_hub_download is False
    assert Settings.from_env({"ALLOW_HUB_DOWNLOAD": "true"}).allow_hub_download is True


def test_blank_values_fall_back_to_defaults():
    settings = Settings.from_env({"DEFAULT_FORMAT": "  ", "SGL_HOST": ""})
    assert settings.default_format == "mp3"
    assert settings.sgl_host == "127.0.0.1"
