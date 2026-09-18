import re

def sanitize_venue_name(venue: str) -> str:
    """
    Sanitizes venue name string into a safe, clean, human-readable short code for filesystem paths and file names.
    - Extracts acronym before '(' if parentheses are present (e.g., 'IROS (IEEE/RSJ...)' -> 'IROS').
    - Replaces slashes, backslashes, colons, and invalid path characters with underscores.
    - Ensures no path traversal or path delimiter issues occur.
    """
    if not venue:
        return "UNKNOWN_VENUE"

    # Extract short tag before '(' if parentheses exist
    raw = venue.split("(")[0].strip() if "(" in venue else venue.strip()

    # Replace invalid path characters: / \ : ? * < > | and spaces
    clean = re.sub(r'[/\\:\?\*\<\>\|\s]+', '_', raw).strip('_')

    if not clean:
        clean = re.sub(r'[/\\:\?\*\<\>\|\s]+', '_', venue).strip('_')

    return clean.upper()

