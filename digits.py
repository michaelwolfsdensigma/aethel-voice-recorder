"""Seven-segment style ASCII digits for the recorder's time readout."""

from __future__ import annotations

# Each glyph is 5 rows tall. Built from heavy box characters so it reads
# like an engraved brass counter.
_GLYPHS = {
    "0": ["██████", "██  ██", "██  ██", "██  ██", "██████"],
    "1": ["   ██ ", "  ███ ", "   ██ ", "   ██ ", "  ████"],
    "2": ["██████", "    ██", "██████", "██    ", "██████"],
    "3": ["██████", "    ██", " █████", "    ██", "██████"],
    "4": ["██  ██", "██  ██", "██████", "    ██", "    ██"],
    "5": ["██████", "██    ", "██████", "    ██", "██████"],
    "6": ["██████", "██    ", "██████", "██  ██", "██████"],
    "7": ["██████", "    ██", "   ██ ", "  ██  ", "  ██  "],
    "8": ["██████", "██  ██", "██████", "██  ██", "██████"],
    "9": ["██████", "██  ██", "██████", "    ██", "██████"],
    ":": ["  ", "██", "  ", "██", "  "],
    ".": ["  ", "  ", "  ", "  ", "██"],
    " ": ["   ", "   ", "   ", "   ", "   "],
}

_ROWS = 5


def big(text: str) -> str:
    """Render a string of digits/':'/'.' as 5-line ASCII art."""
    lines = [""] * _ROWS
    for ch in text:
        glyph = _GLYPHS.get(ch, _GLYPHS[" "])
        for r in range(_ROWS):
            lines[r] += glyph[r] + " "
    return "\n".join(lines)


def format_time(seconds: float) -> str:
    """seconds -> MM:SS.t (or HH:MM:SS over an hour)."""
    if seconds >= 3600:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    m = int(seconds // 60)
    s = int(seconds % 60)
    t = int((seconds * 10) % 10)
    return f"{m:02d}:{s:02d}.{t}"
