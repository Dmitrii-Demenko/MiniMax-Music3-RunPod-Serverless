import pytest

from config import Settings
from request_schema import REJECTED_PARAMETERS, GenerationRequest, RequestError, parse

SETTINGS = Settings.from_env({})
VALID = {"lyrics": "[Verse]\nline one", "prompt": "A lo-fi hip-hop track at 85 BPM"}


def test_minimal_request_uses_documented_defaults():
    request = parse(dict(VALID), SETTINGS)
    assert isinstance(request, GenerationRequest)
    assert request.max_new_tokens == 750  # 30 s default at 25 fps
    assert request.seed == 0
    assert request.format == "mp3"
    assert request.bitrate == "192k"


def test_duration_converts_to_frames_at_25_fps():
    assert parse({**VALID, "duration": 12}, SETTINGS).max_new_tokens == 300


def test_max_new_tokens_can_be_given_directly():
    assert parse({**VALID, "max_new_tokens": 1500}, SETTINGS).max_new_tokens == 1500


def test_duration_and_max_new_tokens_together_are_rejected():
    with pytest.raises(RequestError, match="mutually exclusive"):
        parse({**VALID, "duration": 30, "max_new_tokens": 750}, SETTINGS)


def test_duration_above_the_configured_maximum_is_rejected():
    with pytest.raises(RequestError, match="duration"):
        parse({**VALID, "duration": 361}, SETTINGS)


def test_max_new_tokens_above_the_model_limit_is_rejected():
    with pytest.raises(RequestError, match="9000"):
        parse({**VALID, "max_new_tokens": 9001}, SETTINGS)


def test_full_length_request_is_accepted():
    assert parse({**VALID, "duration": 360}, SETTINGS).max_new_tokens == 9000


def test_lyrics_and_prompt_are_required():
    with pytest.raises(RequestError, match="lyrics"):
        parse({"prompt": "a caption"}, SETTINGS)
    with pytest.raises(RequestError, match="prompt"):
        parse({"lyrics": "[Verse]\nline"}, SETTINGS)


def test_blank_lyrics_are_rejected():
    with pytest.raises(RequestError, match="lyrics"):
        parse({**VALID, "lyrics": "   \n  "}, SETTINGS)


def test_openai_style_aliases_are_accepted():
    request = parse({"input": "[Verse]\nline", "instructions": "a caption"}, SETTINGS)
    assert request.lyrics == "[Verse]\nline"
    assert request.prompt == "a caption"


def test_conflicting_alias_and_canonical_field_is_rejected():
    with pytest.raises(RequestError, match="both"):
        parse({**VALID, "input": "[Verse]\nsomething else"}, SETTINGS)


def test_matching_alias_and_canonical_field_is_accepted():
    request = parse({**VALID, "input": VALID["lyrics"]}, SETTINGS)
    assert request.lyrics == VALID["lyrics"]


@pytest.mark.parametrize("name", sorted(REJECTED_PARAMETERS))
def test_every_unsupported_parameter_is_rejected_with_its_own_code(name):
    with pytest.raises(RequestError) as excinfo:
        parse({**VALID, name: 0.7}, SETTINGS)
    assert excinfo.value.code == "unsupported_parameter"
    assert name in str(excinfo.value)


def test_streaming_is_rejected_but_stream_false_is_accepted():
    with pytest.raises(RequestError) as excinfo:
        parse({**VALID, "stream": True}, SETTINGS)
    assert excinfo.value.code == "unsupported_parameter"
    assert parse({**VALID, "stream": False}, SETTINGS).max_new_tokens == 750


def test_negative_seed_is_rejected():
    with pytest.raises(RequestError, match="seed"):
        parse({**VALID, "seed": -1}, SETTINGS)


def test_non_integer_seed_is_rejected():
    with pytest.raises(RequestError, match="seed"):
        parse({**VALID, "seed": "42"}, SETTINGS)


def test_unsupported_format_is_rejected():
    with pytest.raises(RequestError, match="format"):
        parse({**VALID, "format": "aiff"}, SETTINGS)


def test_bitrate_on_a_lossless_format_warns_instead_of_failing():
    request = parse({**VALID, "format": "flac", "bitrate": "320k"}, SETTINGS)
    assert any("bitrate" in warning for warning in request.warnings)


def test_malformed_bitrate_is_rejected():
    with pytest.raises(RequestError, match="bitrate"):
        parse({**VALID, "bitrate": "320kbps"}, SETTINGS)


def test_lyrics_warnings_are_carried_through():
    request = parse({**VALID, "lyrics": "[Verse] line one"}, SETTINGS)
    assert request.lyrics == "[Verse]\nline one"
    assert len(request.warnings) == 1


def test_non_object_input_is_rejected():
    with pytest.raises(RequestError, match="object"):
        parse(["lyrics"], SETTINGS)


def test_prompt_is_stripped():
    assert parse({**VALID, "prompt": "  a caption  "}, SETTINGS).prompt == "a caption"
