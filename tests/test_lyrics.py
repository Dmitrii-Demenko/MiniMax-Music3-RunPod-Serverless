from lyrics import normalize


def test_text_on_a_tag_line_is_split_and_warned():
    result, warnings = normalize("[Verse] Walking down the street")
    assert result == "[Verse]\nWalking down the street"
    assert len(warnings) == 1
    assert "[Verse]" in warnings[0]


def test_correct_lyrics_are_left_alone():
    text = "[Verse]\nWalking down the street\n[Chorus]\nAnd I keep on walking"
    result, warnings = normalize(text)
    assert result == text
    assert warnings == []


def test_lowercase_tags_are_handled():
    result, warnings = normalize("[verse] morning light")
    assert result == "[verse]\nmorning light"
    assert len(warnings) == 1


def test_tag_alone_on_its_line_is_untouched():
    result, warnings = normalize("[Intro]\n(instrumental)")
    assert result == "[Intro]\n(instrumental)"
    assert warnings == []


def test_unknown_bracketed_prefix_is_also_split():
    # The model applies the same rule to any leading bracket, so protect it too.
    result, warnings = normalize("[Whispered] don't go")
    assert result == "[Whispered]\ndon't go"
    assert len(warnings) == 1


def test_every_offending_line_produces_its_own_warning():
    result, warnings = normalize("[Verse] one\n[Chorus] two")
    assert result == "[Verse]\none\n[Chorus]\ntwo"
    assert len(warnings) == 2


def test_windows_line_endings_are_normalised():
    result, _ = normalize("[Verse]\r\nline one\r\n")
    assert "\r" not in result
    assert result == "[Verse]\nline one"


def test_brackets_inside_a_line_are_not_tags():
    text = "walking [slowly] down the street"
    result, warnings = normalize(text)
    assert result == text
    assert warnings == []


def test_empty_input_returns_empty_output():
    assert normalize("") == ("", [])


def test_leading_whitespace_before_a_tag_is_trimmed():
    result, warnings = normalize("  [Verse] line")
    assert result == "[Verse]\nline"
    assert len(warnings) == 1
