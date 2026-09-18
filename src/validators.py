from pydantic import BaseModel, field_validator, ValidationError
import re
from typing import Optional

# Thai Consonants Range: \u0E01-\u0E2E (Ko Kai to Ho Nokhuk)
# Note: Some specialized plates might use other chars, but user said "NCC", "CC" (Char) which implies consonants.
# User's "N" = Number (Digit), "C" = Character (Thai Consonant).

THAI_CONSONANTS = r"[\u0E01-\u0E2E]"
# Strict Lao License Plate Consonants (Active 20 series consonants; excludes unused ຊ, ງ, ຖ, ປ)
LAO_PLATE_CONSONANTS = r"[ກຂຄຈຍດຕທນບຜພມຣລວສຫອຮ]"
LAO_CONSONANTS = LAO_PLATE_CONSONANTS
DIGIT = r"\d"

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
            if len(clean_num) == 5 and re.match(r"^[7-9]\d", clean_num):
                is_num = None
        
        if not (is_ncc or is_cc or is_c or is_nc or is_nn or is_num):
            raise ValueError(
                f"Invalid plate format: '{v}'. Must match NCC NNNN, CC NNNN, C NNNN, NC NNNN, NN-NNNN, or NNNNN."
            )
        
        return v

def is_valid_lao_plate(upload_text: str) -> bool:
    """Returns True if the text matches strict Lao plate syntax (2 Lao consonants + 1-4 digits)."""
    if not upload_text:
        return False
    clean = upload_text.strip().replace(" ", "")
    return bool(PATTERN_LAO_STANDARD.match(clean))


def format_lao_plate(text: str) -> str:
    """Standardizes Lao plate text into canonical format: [2 Consonants] [1-4 Digits]."""
    if not text:
        return ""
    clean = text.strip().replace(" ", "")
    m = PATTERN_LAO_STANDARD.match(clean)
    if m:
        return f"{m.group(1)} {m.group(2)}"
    return text.strip()


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
    """
    s = text.strip()
    # bare = all separators removed, used for pattern matching
    bare = re.sub(r"[\s-]", "", s)
    # detect if original had a hyphen (to preserve style for letter-plates)
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

    # 5. Pattern: NN-NNNN (2 digits + 4 digits — always canonical with hyphen)
    #    Only format as NN-NNNN if the original had a separator, OR it's exactly 6 digits
    #    AND starts with a known DLT commercial prefix (70-99). Pure 4-6 digit govt plates go to step 6.
    m_grp = re.match(rf"^({DIGIT}{{2}})({DIGIT}{{4}})$", bare)
    if m_grp:
        # If original already had a hyphen/space separator → definitely a truck plate
        if has_hyphen or " " in s:
            return f"{m_grp.group(1)}-{m_grp.group(2)}"
        # If 6 raw digits with no sep: only treat as NN-NNNN if DLT commercial prefix (70-99)
        if re.match(r"^[7-9]\d", bare):
            return f"{m_grp.group(1)}-{m_grp.group(2)}"
        # Otherwise fall through to NNNNN (govt/police)

    # 6. Pattern: NNNNN (4-6 digits — police/government, return bare digits)
    if PATTERN_NNNNN.match(bare):
        return bare

    return s


