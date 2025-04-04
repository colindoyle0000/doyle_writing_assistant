"""Class for storing information about an excerpt from a source."""

import os
import logging

from src.utils_file import get_root_dir, find_page_number
from src.utils_llm import llm_call
from src.utils_string import get_timestamp_as_string

# Set up logger
logger = logging.getLogger('writing_assistant')

class SourceExcerpt:
    """Class for storing information about an excerpt from a source."""
    
    def __init__(
        self,
        source,
        text: str,
        index: int,
        page: int = None,
    ):
        """Initialize a SourceExcerpt object."""
        self.source = source
        self.text = text
        self.index = index
        self.page_number = page
        self.adjusted_page_number = None
        self.in_db = False
    
    def get_page_number(self):
        """Get the page number of the excerpt."""
        self.page_number = find_page_number(self.text, self.source.file_path)
    
    def get_adjusted_page_number(self):
        """Get the adjusted page number of the excerpt based on starting page of article."""
        # Check if self.source.fields has a key 'pages'
        if 'pages' in self.source.fields and isinstance(self.source.fields['pages'], int):
                self.adjusted_page_number = int(self.source.fields['pages']) + self.page_number
        else:
            self.adjusted_page_number = self.page_number
