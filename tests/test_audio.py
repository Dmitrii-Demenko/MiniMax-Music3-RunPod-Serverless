import io
import wave

import pytest

from audio import AudioError, probe_wav, transcode


def make_wav(seconds=0.5, sample_rate=32000, channels=2):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x01" * channels * int(sample_rate * seconds))
    return buffer.getvalue()


def test_probe_reads_the_wav_header():
    info = probe_wav(make_wav(seconds=1.2))
    assert info.sample_rate == 32000
    assert info.channels == 2
    assert info.duration_s == pytest.approx(1.2, abs=0.01)


def test_probe_reports_model_frames_at_25_fps():
    assert probe_wav(make_wav(seconds=2.0)).frames == 50


def test_probe_rejects_non_wav_bytes():
    with pytest.raises(AudioError):
        probe_wav(b"definitely not a wav file")


def test_wav_target_is_a_passthrough():
    data = make_wav()
    out, info = transcode(data, "wav", "192k")
    assert out == data
    assert info.sample_rate == 32000


def test_mp3_output_is_produced():
    out, info = transcode(make_wav(), "mp3", "192k")
    assert len(out) > 0
    assert out[:3] == b"ID3" or out[0] == 0xFF
    assert info.sample_rate == 32000


def test_flac_output_is_produced():
    out, _ = transcode(make_wav(), "flac", "192k")
    assert out[:4] == b"fLaC"


def test_opus_output_is_ogg_and_resampled_to_48k():
    out, info = transcode(make_wav(), "opus", "128k")
    assert out[:4] == b"OggS"
    assert info.sample_rate == 48000


def test_duration_survives_transcoding():
    _, info = transcode(make_wav(seconds=1.5), "mp3", "192k")
    assert info.duration_s == pytest.approx(1.5, abs=0.05)


def test_unknown_target_format_raises():
    with pytest.raises(AudioError, match="format"):
        transcode(make_wav(), "aiff", "192k")


def test_mp3_is_much_smaller_than_wav():
    wav = make_wav(seconds=2.0)
    mp3, _ = transcode(wav, "mp3", "128k")
    assert len(mp3) < len(wav)
