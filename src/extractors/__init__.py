"""
Data extractors module for publication venues.
"""
from src.extractors.base import BaseExtractor
from src.extractors.openalex import OpenAlexExtractor
from src.extractors.ieee_xplore import IEEEExtractor
from src.extractors.scopus import ScopusExtractor
from src.extractors.conference_schedule import ConferenceScheduleExtractor

__all__ = [
    "BaseExtractor",
    "OpenAlexExtractor",
    "IEEEExtractor",
    "ScopusExtractor",
    "ConferenceScheduleExtractor",
]
