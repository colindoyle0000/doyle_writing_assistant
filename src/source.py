"""Class for storing information about a source."""

import os
import logging
import bibtexparser

from src.source_excerpt import SourceExcerpt

from src.utils_llm import llm_call
from src.utils_string import get_timestamp_as_string
from src.utils_file import get_root_dir, extract_from_pdf, extract_from_docx, extract_from_md, extract_text_from_multiple_files
from src.utils_tokens import list_to_token_list, string_to_token_list


# Set up logger
logger = logging.getLogger('writing_assistant')

class Source:
    """Class for storing information about a source."""
    
    def __init__(
        self,
        entry,
        project,
    ):
        """Initialize a Source instance from a BibTex entry."""
        self.entry_type = entry.get('ENTRYTYPE', None)
        self.citekey = entry.get('ID', None)
        self.fields = entry
        self.project = project
        self.text = ""
        self.excerpt_token_lst = []
        self.excerpts = []
        logger.info(f"Created Source instance for {self.citekey}")
        # Extract text from the PDF if the file path is provided
        self.file_path = entry.get('file', None)
        if self.file_path:
            self.text = extract_text_from_multiple_files(self.file_path)
        else:
            logger.warning(f"No PDF file path provided for {self.citekey}")

    def __repr__(self):
        return f"Source(citekey={self.citekey}, entry_type={self.entry_type})"
        
    def create_excerpts(self):
        """Create excerpts from the source."""
        # Clean up self.text to not include "<|endoftext|>" if present.
        self.text = self.text.replace("<|endoftext|>", "")
        # Create a list of token-sized excerpts from the extracted text.
        self.excerpt_token_lst = []
        self.excerpt_token_lst = string_to_token_list(
            self.text,
            chunk_size=self.project.llm_settings.chunk_size,
            chunk_overlap=self.project.llm_settings.chunk_overlap,
            )
        logger.info(f"Created excerpt token list for {self.citekey}. Number of excerpts: {len(self.excerpt_token_lst)}")
        # Create a list of Excerpt instances from the token list.
        self.excerpts = []
        for i, text in enumerate(self.excerpt_token_lst):
            excerpt = SourceExcerpt(
                source=self,
                text=text,
                index=i,
            )
            self.excerpts.append(excerpt)
        logger.info(f"Created {len(self.excerpts)} excerpts for {self.citekey}")
