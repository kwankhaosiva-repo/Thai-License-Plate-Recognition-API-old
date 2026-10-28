from pydantic import BaseModel, field_validator, ValidationError
import re
from typing import Optional

# Thai Consonants Range: ก-ฮ (Ko Kai to Ho Nokhuk)
# User\x27s "N" = Number (Digit), "C" = Character (Thai Consonant).
THAI_CONSONANTS = r"[\u0E01-\u0E2E]"
# Strict Lao License Plate Consonants (Active 20 series consonants; excludes unused ຊ, ງ, ຖ, ປ)
LAO_PLATE_CONSONANTS = r"[ກຂຄຈຍດຕທນບຜພມຣລວສຫອຮ]"
LAO_CONSONANTS = LAO_PLATE_CONSONANTS
DIGIT = r"\d"

# Trailing digits regex pattern (strictly 1 to 4 digits at the end of plate)
PATTERN_TRAILING_DIGITS = re.compile(rf"({DIGIT}{{1,4}})$")

# Character to digit confusion maps for trailing numeric section
CONFUSION_TO_DIGIT_THAI = {
    "อ": "0", "ฮ": "0", "ถ": "0", "ด": "0", "O": "0", "o": "0", "D": "0", "Q": "0",
    "เ": "1", "า": "1", "ใ": "1", "ไ": "1", "I": "1", "l": "1", "|": "1", "๑": "1",
    "Z": "2", "z": "2", "๒": "2",
    "E": "3", "๓": "3",
    "A": "4", "๔": "4",
    "ส": "5", "S": "5", "s": "5", "๕": "5",
    "บ": "6", "ธ": "6", "b": "6", "G": "6", "๖": "6",
    "ฯ": "7", "T": "7", "t": "7", "๗": "7",
    "ม": "8", "ผ": "8", "พ": "8", "ฟ": "8", "ษ": "8", "ข": "8", "B": "8", "๘": "8",
    "ฎ": "9", "ฏ": "9", "g": "9", "q": "9", "๙": "9",
}

CONFUSION_TO_DIGIT_LAO = {
    "ອ": "0", "ຮ": "0", "ດ": "0", "O": "0", "o": "0", "D": "0", "Q": "0",
    "I": "1", "l": "1", "|": "1",
    "Z": "2", "z": "2",
    "E": "3",
    "A": "4",
    "ສ": "5", "S": "5", "s": "5",
    "ບ": "6", "b": "6", "G": "6",
    "T": "7", "t": "7",
    "ມ": "8", "ຜ": "8", "ພ": "8", "ຂ": "8", "B": "8",
    "g": "9", "q": "9",
}

# SEP = optional separator: single space, hyphen, or nothing (covers real-world OCR variations)
SEP = r"[\s-]?"

# Regex Patterns
# NCC NNNN: 1 digit + 2 Thai consonants + optional sep + 1-4 digits (new private series, e.g. 1กข 1234, 1กข-1234)
PATTERN_NCC_NNNN = re.compile(rf"^{DIGIT}{THAI_CONSONANTS}{{2}}{SEP}{DIGIT}{{1,4}}$")
# CC NNNN: 2 Thai consonants + optional sep + 1-4 digits (standard private, e.g. กข 1234, กข-1234)
PATTERN_CC_NNNN  = re.compile(rf"^{THAI_CONSONANTS}{{2}}{SEP}{DIGIT}{{1,4}}$")
# C NNNN: 1 Thai consonant + optional sep + 1-4 digits (antique/motorcycle, e.g. ณ 4100, ณ-4100)
PATTERN_C_NNNN   = re.compile(rf"^{THAI_CONSONANTS}{SEP}{DIGIT}{{1,4}}$")
# NC NNNN: 1 digit + 1 Thai consonant + optional sep + 1-4 digits (trailer/machinery, e.g. 5ศ-7856)
PATTERN_NC_NNNN  = re.compile(rf"^{DIGIT}{THAI_CONSONANTS}\s*[-]?\s*{DIGIT}{{1,4}}$")
# NN-NNNN: 2 digits + optional sep + 4 digits (truck/transport, e.g. 82-6990, 82 6990, 826990)
PATTERN_NN_NNNN  = re.compile(rf"^{DIGIT}{{2}}{SEP}{DIGIT}{{4}}$")
# NNNNN: 4-6 all-digit plates (police/official/government, e.g. 1234, 12345, 123456)
PATTERN_NNNNN    = re.compile(rf"^{DIGIT}{{4,6}}$")

# Lao Standard Regex: Strictly 2 valid Lao consonants in front followed by 1 to 4 digits (e.g., ກກ 0083, ກວ 8607)
PATTERN_LAO_STANDARD = re.compile(rf"^({LAO_PLATE_CONSONANTS}{{2}})\s*({DIGIT}{{1,4}})$")


class PlateLabelValidator(BaseModel):
    text: str
    
    @field_validator('text')
    @classmethod
    def validate_format(cls, v: str) -> str:
        v_stripped = v.strip()
        is_ncc = PATTERN_NCC_NNNN.match(v_stripped)
        is_cc  = PATTERN_CC_NNNN.match(v_stripped)
        is_c   = PATTERN_C_NNNN.match(v_stripped)
        is_nc  = PATTERN_NC_NNNN.match(v_stripped)
        is_nn  = PATTERN_NN_NNNN.match(v_stripped)
        is_num = PATTERN_NNNNN.match(v_stripped)
        if is_num:
            # Series 70-99 are strictly DLT commercial transport trucks/buses requiring NN-NNNN (6 digits).
            # A 5-digit string starting with 70-99 is an incomplete truck plate missing a digit, NOT an official plate.
            clean_num = v_stripped.replace("-", "").replace(" ", "")
            if len(clean_num) == 5 and re.match(r"^[1-9]\d", clean_num):
                is_num = None
        
        if not (is_ncc or is_cc or is_c or is_nc or is_nn or is_num):
            raise ValueError(
                f"Invalid plate format: \x27{v}\x27. Must match NCC NNNN, CC NNNN, C NNNN, NC NNNN, NN-NNNN, or NNNNN."
            )
        
        return v


def enforce_trailing_digits(text: str, country: str = "Thai") -> str:
    """
    Enforces that the trailing 1-4 characters of a license plate must strictly be numeric digits.
    Uses regex to extract the prefix (letters or transport prefix) and the trailing registration
    segment. If the trailing segment contains letter confusions or non-digits, it maps them
    to digits and strips non-digits, ensuring the trailing group is purely numbers.
    """
    if not text:
        return ""
    s = text.strip()
    bare = re.sub(r"[\s-]", "", s)
    has_hyphen = "-" in s
    conf_map = CONFUSION_TO_DIGIT_LAO if country.lower() == "laos" else CONFUSION_TO_DIGIT_THAI

    if country.lower() == "laos":
        # Lao: strictly 2 Lao consonants prefix + 1-4 trailing digits
        m = re.match(rf"^({LAO_PLATE_CONSONANTS}{{2}})(.+)$", bare)
        if m:
            prefix = m.group(1)
            raw_trailing = m.group(2)
            mapped_trailing = "".join(conf_map.get(ch, ch) for ch in raw_trailing)
            clean_digits = re.sub(r"\D", "", mapped_trailing)[:4]
            if clean_digits:
                return f"{prefix} {clean_digits}"
        return s

    # Thai: check standard prefixes
    m = re.match(rf"^({DIGIT}{THAI_CONSONANTS}{{2}})(.+)$", bare)
    if not m:
        m = re.match(rf"^({THAI_CONSONANTS}{{2}})(.+)$", bare)
    if not m:
        m = re.match(rf"^({DIGIT}{THAI_CONSONANTS})(.+)$", bare)
    if not m:
        m = re.match(rf"^({THAI_CONSONANTS})(.+)$", bare)
    if not m:
        m = re.match(rf"^({DIGIT}{{2}})(.+)$", bare)

    if m:
        prefix = m.group(1)
        raw_trailing = m.group(2)
        mapped_trailing = "".join(conf_map.get(ch, ch) for ch in raw_trailing)
        clean_digits = re.sub(r"\D", "", mapped_trailing)[:4]
        if clean_digits:
            sep = "-" if has_hyphen else " "
            return f"{prefix}{sep}{clean_digits}"

    return s


def is_valid_lao_plate(upload_text: str) -> bool:
    """Returns True if the text matches strict Lao plate syntax (2 Lao consonants + 1-4 digits)."""
    if not upload_text:
        return False
    clean = upload_text.strip().replace(" ", "")
    # Strict trailing digit check: must end with 1-4 digits
    if not re.search(r"\d{1,4}$", clean):
        return False
    return bool(PATTERN_LAO_STANDARD.match(clean))


def format_lao_plate(text: str) -> str:
    """Standardizes Lao plate text into canonical format: [2 Consonants] [1-4 Digits].
    Strictly enforces that the trailing registration characters (up to 4) are numbers.
    """
    if not text:
        return ""
    clean = text.strip().replace(" ", "")
    m = PATTERN_LAO_STANDARD.match(clean)
    if m:
        return f"{m.group(1)} {m.group(2)}"

    # Trailing digit regex enforcement
    enforced = enforce_trailing_digits(text, country="Laos")
    clean_enf = enforced.replace(" ", "")
    m_enf = PATTERN_LAO_STANDARD.match(clean_enf)
    if m_enf:
        return f"{m_enf.group(1)} {m_enf.group(2)}"

    return text.strip()


def is_valid_thai_plate(upload_text: str) -> bool:
    """Returns True if the text matches strict Thai plate rules and ends with 1-4 digits."""
    if not upload_text:
        return False
    clean = upload_text.strip().replace(" ", "").replace("-", "")
    # Trailing characters must be digits (1-4 digits or 4-6 for government)
    if not re.search(r"\d{1,4}$", clean) and not PATTERN_NNNNN.match(clean):
        return False
    return is_valid_plate(upload_text, country="Thai")


def is_valid_plate(upload_text: str, country: str = "Thai") -> bool:
    """Returns True if the text matches strict plate rules for the specified country."""
    if country == "Laos":
        return is_valid_lao_plate(upload_text)
    try:
        PlateLabelValidator(text=upload_text)
        return True
    except ValidationError:
        return False


def format_thai_plate(text: str) -> str:
    """Standardizes Thai plate text into canonical format with proper spacing:
    - NCC NNNN  (1 digit + 2 consonants + space + 1-4 digits)
    - NCC-NNNN  (1 digit + 2 consonants + hyphen + 1-4 digits) — preserves hyphen
    - CC NNNN   (2 consonants + space + 1-4 digits)
    - CC-NNNN   (2 consonants + hyphen + 1-4 digits) — preserves hyphen
    - C NNNN / C-NNNN  (1 consonant + 1-4 digits)
    - NC NNNN / NC-NNNN (1 digit + 1 consonant + 1-4 digits)
    - NN-NNNN   (2 digits + hyphen + 4 digits) — normalizes to hyphen form
    - NNNNN     (4-6 digits only)
    Strictly enforces that the trailing registration characters (up to 4) are numbers.
    """
    s = text.strip()
    bare = re.sub(r"[\s-]", "", s)
    has_hyphen = "-" in s

    # 1. Pattern: NCC-?NNNN (1 digit, 2 Thai consonants, optional sep, 1-4 digits)
    m_grp = re.match(rf"^({DIGIT}{THAI_CONSONANTS}{{2}})({DIGIT}{{1,4}})$", bare)
    if m_grp:
        sep = "-" if has_hyphen else " "
        return f"{m_grp.group(1)}{sep}{m_grp.group(2)}"

    # 2. Pattern: CC-?NNNN (2 Thai consonants, optional sep, 1-4 digits)
    m_grp = re.match(rf"^({THAI_CONSONANTS}{{2}})({DIGIT}{{1,4}})$", bare)
    if m_grp:
        sep = "-" if has_hyphen else " "
        return f"{m_grp.group(1)}{sep}{m_grp.group(2)}"

    # 3. Pattern: C-?NNNN (1 Thai consonant, optional sep, 1-4 digits)
    m_grp = re.match(rf"^({THAI_CONSONANTS})({DIGIT}{{1,4}})$", bare)
    if m_grp:
        sep = "-" if has_hyphen else " "
        return f"{m_grp.group(1)}{sep}{m_grp.group(2)}"

    # 4. Pattern: NC-?NNNN (1 digit, 1 Thai consonant, optional sep, 1-4 digits)
    m_grp = re.match(rf"^({DIGIT}{THAI_CONSONANTS})({DIGIT}{{1,4}})$", bare)
    if m_grp:
        sep = " - " if has_hyphen else " "
        return f"{m_grp.group(1)}{sep}{m_grp.group(2)}"

    # 5. Pattern: NN-NNNN (2 digits + 4 digits — commercial transport)
    m_grp = re.match(rf"^({DIGIT}{{2}})({DIGIT}{{4}})$", bare)
    if m_grp:
        return f"{m_grp.group(1)}-{m_grp.group(2)}"

    # 6. Pattern: NNNNN (4-6 digits — police/government)
    if PATTERN_NNNNN.match(bare):
        return bare

    # Fallback with Trailing Digit Regex Enforcement:
    enforced = enforce_trailing_digits(text, country="Thai")
    if enforced != text:
        bare_enf = re.sub(r"[\s-]", "", enforced)
        for pattern_func in [
            lambda b: re.match(rf"^({DIGIT}{THAI_CONSONANTS}{{2}})({DIGIT}{{1,4}})$", b),
            lambda b: re.match(rf"^({THAI_CONSONANTS}{{2}})({DIGIT}{{1,4}})$", b),
            lambda b: re.match(rf"^({THAI_CONSONANTS})({DIGIT}{{1,4}})$", b),
            lambda b: re.match(rf"^({DIGIT}{THAI_CONSONANTS})({DIGIT}{{1,4}})$", b),
            lambda b: re.match(rf"^({DIGIT}{{2}})({DIGIT}{{4}})$", b),
        ]:
            m_res = pattern_func(bare_enf)
            if m_res:
                sep = "-" if has_hyphen else " "
                return f"{m_res.group(1)}{sep}{m_res.group(2)}"

    return s
