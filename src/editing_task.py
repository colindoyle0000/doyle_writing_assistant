"""EditingTask class for editing drafts."""

import os
import logging

from src.utils_file import get_root_dir
from src.utils_llm import (
    LLMSettings,
)
from src.utils_string import get_timestamp_as_string
from src.section import Section

# Set up logger
logger = logging.getLogger('writing_assistant')

class EditingTask:
    """Class for editing a draft."""
    
    def __init__(
        self,
        task_name: str,
        project,
    ):
        """Initialize an ExecutiveEditor object."""
        self.task_name = task_name
        self.project = project
        
        # Edited draft variables
        self.llm_edits_lst = []
        self.last_llm_edit = ""
        
        # Added citations variables
        self.citation_notes_lst = []
        self.last_citation_notes = ""
        
        # Quotes variables
        self.quotes_notes_lst = []
        self.last_quotes_notes = ""
        
        # Plagiarism variables
        self.plagiarism_notes_lst = []
        self.last_plagiarism_notes = ""
        
        # Bluebook variables
        self.missing_bluebook_info_lst = []
        self.bluebooked_drafts_lst = []
        self.last_bluebooked_draft = ""
        
        # Markdown draft variables
        self.md_drafts_lst = []
        self.last_md_draft = ""
    
    def edit_draft_style(self):
        """Edit a draft for style."""
        pass
    
    def add_citations(self):
        """Add citations to the draft."""
        pass
    
    def check_quotes(self):
        """Check to make sure that any quotes in the draft can be found in the sources."""
        pass
    
    def check_plagiarism(self):
        """Check the draft for text that is a copy of source text or very similar to source text."""
        pass
    
    def bluebook_citations(self):
        """Convert citations to Bluebook format."""
        pass
    
    def convert_to_markdown(self):
        """Convert the draft to Markdown format.
        This method parses the draft, turning every instance of "/foonote{...}" into a Markdown footnote, starting with [^1] and incrementing by 1.
        """
        pass