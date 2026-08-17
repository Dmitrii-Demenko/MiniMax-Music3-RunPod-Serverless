import os

import pytest

from config import Settings
from model_path import ModelPathError, missing_artifacts, resolve_model_path


def make_checkpoint(root):
    (root / "qwen_7B" / "qwen_7B").mkdir(parents=True)
    (root / "qwen_7B" / "qwen3-8B-tokenizer-music").mkdir(parents=True)
    (root / "flowmatching_vae.pth").write_bytes(b"x")
    (root / "dav.pth").write_bytes(b"x")
    return root


def test_complete_checkpoint_has_no_missing_artifacts(tmp_path):
    assert missing_artifacts(make_checkpoint(tmp_path)) == []


def test_missing_weight_file_is_reported(tmp_path):
    make_checkpoint(tmp_path)
    (tmp_path / "dav.pth").unlink()
    assert missing_artifacts(tmp_path) == ["dav.pth"]


def test_a_file_where_a_directory_is_expected_is_reported(tmp_path):
    make_checkpoint(tmp_path)
    tokenizer = tmp_path / "qwen_7B" / "qwen3-8B-tokenizer-music"
    tokenizer.rmdir()
    tokenizer.write_bytes(b"not a directory")
    assert missing_artifacts(tmp_path) == ["qwen_7B/qwen3-8B-tokenizer-music"]


def test_explicit_model_path_wins(tmp_path):
    root = make_checkpoint(tmp_path / "weights")
    settings = Settings.from_env({"MODEL_PATH": str(root)})
    assert resolve_model_path(settings) == str(root)


def test_incomplete_explicit_model_path_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    settings = Settings.from_env({"MODEL_PATH": str(tmp_path / "empty")})
    with pytest.raises(ModelPathError, match="qwen_7B"):
        resolve_model_path(settings)


def test_cached_model_snapshot_is_discovered(tmp_path):
    snapshot = (
        tmp_path / "hub" / "models--MiniMaxAI--MiniMax-Music3" / "snapshots" / "abc123"
    )
    make_checkpoint(snapshot)
    settings = Settings.from_env({"HF_HOME": str(tmp_path)})
    assert resolve_model_path(settings) == str(snapshot)


def test_newest_complete_snapshot_wins(tmp_path):
    base = tmp_path / "hub" / "models--MiniMaxAI--MiniMax-Music3" / "snapshots"
    old = make_checkpoint(base / "old")
    new = make_checkpoint(base / "new")
    os.utime(old, (1_000_000, 1_000_000))
    os.utime(new, (2_000_000, 2_000_000))
    settings = Settings.from_env({"HF_HOME": str(tmp_path)})
    assert resolve_model_path(settings) == str(new)


def test_incomplete_snapshot_is_skipped_for_a_complete_one(tmp_path):
    base = tmp_path / "hub" / "models--MiniMaxAI--MiniMax-Music3" / "snapshots"
    (base / "broken").mkdir(parents=True)
    good = make_checkpoint(base / "good")
    settings = Settings.from_env({"HF_HOME": str(tmp_path)})
    assert resolve_model_path(settings) == str(good)


def test_repo_id_is_the_last_resort(tmp_path):
    settings = Settings.from_env(
        {"HF_HOME": str(tmp_path), "ALLOW_HUB_DOWNLOAD": "1"}
    )
    assert resolve_model_path(settings) == "MiniMaxAI/MiniMax-Music3"


def test_missing_weights_without_hub_download_raises(tmp_path):
    settings = Settings.from_env({"HF_HOME": str(tmp_path)})
    with pytest.raises(ModelPathError, match="no usable checkpoint"):
        resolve_model_path(settings)
