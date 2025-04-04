"""Class for storing information about an excerpt from a note."""

import os
import logging

from src.utils_file import get_root_dir, find_page_number
from src.utils_llm import llm_call
from src.utils_string import get_timestamp_as_string

# Set up logger
logger = logging.getLogger('writing_assistant')

class NoteExcerpt:
    """Class for storing information about an excerpt from a note."""
    
    def __init__(
        self,
        note,
        text: str,
        index: int,
    ):
        """Initialize a NoteExcerpt object."""
        self.note = note
        self.text = text
        self.index = index
        self.in_db = False