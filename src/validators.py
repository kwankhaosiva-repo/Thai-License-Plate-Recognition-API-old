from pydantic import BaseModel, field_validator, ValidationError
import re
from typing import Optional

# Thai Consonants Range: \u0E01-\u0E2E (Ko Kai to Ho Nokhuk)
# Note: Some specialized plates might use other chars, but user said "NCC", "CC" (Char) which implies consonants.
# User's "N" = Number (Digit), "C" = Character (Thai Consonant).

THAI_CONSONANTS = r"[\u0E01-\u0E2E]"
LAO_CONSONANTS = r"[\u0E81-\u0EAE]"
DIGIT = r"\d"

# Regex Patterns
PATTERN_NCC_NNNN = re.compile(rf"^{DIGIT}{THAI_CONSONANTS}{{2}}\s?{DIGIT}{{1,4}}$") # e.g., 1กข 1234 or 1กข1234
PATTERN_CC_NNNN  = re.compile(rf"^{THAI_CONSONANTS}{{2}}\s?{DIGIT}{{1,4}}$")        # e.g., กข 1234 or ฮร 9960
PATTERN_C_NNNN   = re.compile(rf"^{THAI_CONSONANTS}[\s-]?{DIGIT}{{1,4}}$")             # e.g., ก 1234 or ส-5887
PATTERN_NC_NNNN  = re.compile(rf"^{DIGIT}{THAI_CONSONANTS}\s*[-]?\s*{DIGIT}{{1,4}}$")    # e.g., 5ศ - 7856 or 5ศ 7856 (Trailer/Machinery/Special)
PATTERN_NN_NNNN  = re.compile(rf"^{DIGIT}{{2}}-{DIGIT}{{4}}$")                      # e.g., 82-6990 (truck / trailer)
PATTERN_NNNNN    = re.compile(rf"^{DIGIT}{{4,6}}$")                                 # e.g., 12345 (police / official)

# Lao Standard Regex: Strictly 2 Lao consonants in front followed by 1 to 4 digits (e.g., ກກ 0083, ກວ 8607)
PATTERN_LAO_STANDARD = re.compile(rf"^({LAO_CONSONANTS}{{2}})\s*({DIGIT}{{1,4}})$")

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
    - NCC NNNN (1-4 digits)
    - CC NNNN (1-4 digits)
    - C NNNN / C-NNNN (1-4 digits)
    - NN-NNNN (1-4 digits)
    - NNNNN (1-6 digits)
    """
    s = text.strip()
    clean = s.replace(" ", "")

    # 1. Pattern: NCC NNNN (1 digit, 2 Thai consonants, 1-4 digits)
    m = PATTERN_NCC_NNNN.match(clean)
    if m:
        m_grp = re.match(rf"^({DIGIT}{THAI_CONSONANTS}{{2}})({DIGIT}{{1,4}})$", clean)
        if m_grp:
            return f"{m_grp.group(1)} {m_grp.group(2)}"

    # 2. Pattern: CC NNNN (2 Thai consonants, 1-4 digits)
    m = PATTERN_CC_NNNN.match(clean)
    if m:
        m_grp = re.match(rf"^({THAI_CONSONANTS}{{2}})({DIGIT}{{1,4}})$", clean)
        if m_grp:
            return f"{m_grp.group(1)} {m_grp.group(2)}"

    # 3. Pattern: C NNNN or C-NNNN (1 Thai consonant, 1-4 digits)
    m = PATTERN_C_NNNN.match(s)
    if m:
        sep = "-" if "-" in s else " "
        m_grp = re.match(rf"^({THAI_CONSONANTS})[\s-]?({DIGIT}{{1,4}})$", s)
        if m_grp:
            return f"{m_grp.group(1)}{sep}{m_grp.group(2)}"

    # 3.5. Pattern: NC NNNN or NC-NNNN (1 digit, 1 Thai consonant, 1-4 digits, e.g. 5ศ - 7856)
    m = PATTERN_NC_NNNN.match(clean)
    if m:
        sep = " - " if "-" in s else " "
        m_grp = re.match(rf"^({DIGIT}{THAI_CONSONANTS})[\s-]?({DIGIT}{{1,4}})$", clean)
        if m_grp:
            return f"{m_grp.group(1)}{sep}{m_grp.group(2)}"

    # 4. Pattern: NN-NNNN (Commercial trucks with or without hyphen)
    m = PATTERN_NN_NNNN.match(clean)
    if m:
        m_grp = re.match(rf"^({DIGIT}{{2}})-?({DIGIT}{{1,4}})$", clean)
        if m_grp:
            return f"{m_grp.group(1)}-{m_grp.group(2)}"
    elif re.match(r"^\d{6}$", clean):
        return f"{clean[:2]}-{clean[2:]}"

    # 5. Pattern: NNNNN (Police / government all digits)
    m = PATTERN_NNNNN.match(clean)
    if m:
        return clean

    return s


