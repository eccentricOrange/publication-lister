import re
from typing import Any

def sanitize_venue_name(venue: Any) -> str:
    """
    Sanitizes venue name string or list of venue search strings into a safe, clean, human-readable short code.
    - If venue is a list, uses the first non-empty string in the list.
    - Extracts acronym before '(' if parentheses are present (e.g., 'IROS (IEEE/RSJ...)' -> 'IROS').
    - Replaces slashes, backslashes, colons, and invalid path characters with underscores.
    - Ensures no path traversal or path delimiter issues occur.
    """
    if not venue:
        return "UNKNOWN_VENUE"

    if isinstance(venue, list):
        clean_items = [str(x).strip() for x in venue if str(x).strip()]
        if not clean_items:
            return "UNKNOWN_VENUE"
        venue_str = clean_items[0]
    else:
        venue_str = str(venue)

    # Extract short tag before '(' if parentheses exist
    raw = venue_str.split("(")[0].strip() if "(" in venue_str else venue_str.strip()

    # Replace invalid path characters: / \ : ? * < > | and spaces
    clean = re.sub(r'[/\\:\?\*\<\>\|\s]+', '_', raw).strip('_')

    if not clean:
        clean = re.sub(r'[/\\:\?\*\<\>\|\s]+', '_', venue_str).strip('_')

    return clean.upper()

