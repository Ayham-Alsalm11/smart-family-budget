"""Arabic text normalisation shared by the yes/no rules (§6.5) and the category keywords of search (§7.1).

Users write the same word with different hamza forms and diacritics, so matching on the raw text misses it.
"""
import re
import unicodedata

_HAMZA = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ؤ": "و", "ئ": "ي"})
_PUNCTUATION = re.compile(r"[^\w\s]|_")


def normalise(text: str) -> str:
    """Without diacritics, punctuation and hamza forms, in lower case, with single spaces."""
    text = "".join(c for c in unicodedata.normalize("NFC", text)
                   if unicodedata.category(c) not in ("Mn", "Cf") and c != "ـ")
    return " ".join(_PUNCTUATION.sub(" ", text.translate(_HAMZA).lower()).split())
