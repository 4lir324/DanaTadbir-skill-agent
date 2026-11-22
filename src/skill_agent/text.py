"""Text normalization shared by the shortlist and the calculator (Persian-aware)."""

import re
import unicodedata

_CHAR_MAP = str.maketrans(
    {
        "ي": "ی",  # Arabic yeh -> Persian yeh
        "ك": "ک",  # Arabic kaf -> Persian kaf
        "‌": " ",  # zero-width non-joiner
        "‏": " ",  # right-to-left mark
        "‎": " ",  # left-to-right mark
    }
)
_DIGIT_MAP = {ord(d): str(i) for i, d in enumerate("۰۱۲۳۴۵۶۷۸۹")}
_DIGIT_MAP.update({ord(d): str(i) for i, d in enumerate("٠١٢٣٤٥٦٧٨٩")})
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def normalize(text: str) -> str:
    """Lowercase, unify Arabic/Persian letters, ASCII digits, drop diacritics."""
    text = unicodedata.normalize("NFKC", text).translate(_CHAR_MAP).translate(_DIGIT_MAP)
    text = "".join(ch for ch in text if not unicodedata.category(ch).startswith("Mn"))
    return text.lower()


def tokenize(text: str) -> list[str]:
    """Split normalized text into word tokens."""
    return _TOKEN_RE.findall(normalize(text))
