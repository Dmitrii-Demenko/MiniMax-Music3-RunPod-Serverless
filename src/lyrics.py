"""Lyrics normalisation.

MiniMax-Music3 normalises a line that starts with a structure tag down to the tag
alone and drops the rest of that line. So this input:

    [Verse] Walking down the street

reaches the model as `[start] [verse]` and the lyric is gone, with no warning.
We split such lines onto two lines and report what we changed.
"""

from __future__ import annotations

import re

# Documented structure tags, kept for reference and for callers that want to
# validate against them. Normalisation deliberately does not consult this set:
# the model applies the same rule to any leading bracket, so we protect all of them.
STRUCTURE_TAGS = frozenset(
    {
        "intro",
        "verse",
        "pre-chorus",
        "chorus",
        "post-chorus",
        "bridge",
        "instrumental",
        "solo",
        "outro",
    }
)

_TAG_LINE = re.compile(r"^(\s*\[[^\]\n]+\])[ \t]*(\S.*)$")


def normalize(text: str) -> tuple[str, list[str]]:
    """Put every structure tag on its own line.

    Returns the normalised lyrics and one warning per line that had to be split.
    """
    if not text:
        return "", []

    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: list[str] = []
    warnings: list[str] = []

    for line in lines:
        match = _TAG_LINE.match(line)
        if match is None:
            out.append(line)
            continue
        tag, remainder = match.group(1).strip(), match.group(2)
        out.append(tag)
        out.append(remainder)
        warnings.append(
            f"lyrics: text followed {tag} on the same line and would have been "
            f"dropped by the model; it was moved to its own line"
        )

    return "\n".join(out).strip("\n"), warnings
