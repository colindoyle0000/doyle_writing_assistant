"""Class for storing information about a note."""

import os
import logging

from src.source_excerpt import SourceExcerpt

from src.utils_llm import llm_call
from src.utils_string import get_timestamp_as_string
from src.utils_file import get_root_dir, extract_from_pdf, extract_from_docx, extract_from_md, extract_text_from_multiple_files
from src.utils_tokens import list_to_token_list, string_to_token_list
from src.note_excerpt import NoteExcerpt

# Set up logger
logger = logging.getLogger('writing_assistant')

class Note:
    """Class for storing information about a note."""
    
    def __init__(
        self,
        filename: str,
        file_path: str,
        project,
    ):
        """Initialize a Note instance."""
        self.filename = filename
        self.file_path = file_path
        self.project = project
        logger.info(f"Created Note instance for {self.filename}")
        # Extract text from the note file whether it is a PDF, DOCX, or MD file.
        self.text = extract_text_from_multiple_files(self.file_path)
        self.excerpt_token_lst = []
        self.excerpts = []
        
    def __repr__(self):
        return f"Note(filename={self.filename}, file_path={self.file_path})"
    
    def create_excerpts(self):
        """Create excerpts from the note."""
        # Create a list of token-sized excerpts from the extracted text.
        self.excerpt_token_lst = []
        self.excerpt_token_lst = string_to_token_list(
            self.text,
            chunk_size=self.project.llm_settings.chunk_size,
            chunk_overlap=self.project.llm_settings.chunk_overlap,
            )
        logger.info(f"Created excerpt token list for {self.filename}. Number of excerpts: {len(self.excerpt_token_lst)}")
        # Create a list of Excerpt instances from the token list.
        self.excerpts = []
        for i, text in enumerate(self.excerpt_token_lst):
            excerpt = NoteExcerpt(
                note=self,
                text=text,
                index=i,
            )
            self.excerpts.append(excerpt)
        logger.info(f"Created {len(self.excerpts)} excerpts for {self.filename}")