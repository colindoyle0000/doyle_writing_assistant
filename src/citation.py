"""Citations class for a citation object within the FormatTask class."""

import os
import logging

# Set up logger
logger = logging.getLogger('writing_assistant')

class Citation:
    """Class for a citation object within the FormatTask class."""
    
    def __init__(
        self,
        original_text,
        footnote_num,
        location_cite,
        alone_in_footnote=True,
    ):
        """Initialize a Citation instance."""
        self.original_text = original_text
        self.footnote_num = footnote_num
        self.location_cite = location_cite
        self.alone_in_footnote = alone_in_footnote

        self.citekey = ""
        self.first_occurrence = True
        self.page_num = ""
        self.parenthetical = ""
        self.cite_in_draft = ""
        # Whether or not the cite in draft is id.
        self.is_id = False
        logger.info(f"Created Citation instance for {self.original_text}")
    
    def __repr__(self):
        return f"Citation(citekey={self.citekey}, Footnote={self.footnote_num})"
    
    def extract_citekey(self):
        """Extract the citekey from the original text.
        Citekeys always start with "@@" and end with a space, "}" "[", or ")".
        """
        # With self.original_text, find the first instance of "@@"
        citekey_start = self.original_text.find("@@")
        
        # Find position of each possible ending character
        space_pos = self.original_text.find(" ", citekey_start)
        brace_pos = self.original_text.find("}", citekey_start)
        bracket_pos = self.original_text.find("[", citekey_start)
        paren_pos = self.original_text.find("(", citekey_start)

        # Filter out -1 values and find minimum
        positions = [pos for pos in [space_pos, brace_pos, bracket_pos, paren_pos] if pos != -1]

        # If no ending character found, use end of string
        if not positions:
            citekey_end = len(self.original_text)
        else:
            citekey_end = min(positions)
        
        # Assign the result to self.citekey
        self.citekey = self.original_text[citekey_start + 2:citekey_end]
    
    def extract_page_number(self):
        """Extract the page number from the original text.
        Page numbers are always the first text contained in [] after the citekey.
        """
        citekey_start = self.original_text.find(self.citekey)
        page_num_start = self.original_text.find("[", citekey_start)
        
        # If no opening bracket found, return empty page number
        if page_num_start == -1:
            self.page_num = ""
            return
            
        page_num_end = self.original_text.find("]", page_num_start)
        
        # If no closing bracket found, return empty page number
        if page_num_end == -1:
            self.page_num = ""
            return
            
        self.page_num = self.original_text[page_num_start + 1:page_num_end]

    
    def extract_parenthetical(self):
        """Extract the parenthetical from the original text.
        Parentheticals are always the text contained in () after the citekey.
        """
        citekey_start = self.original_text.find(self.citekey)
        parenthetical_start = self.original_text.find("(", citekey_start)
        
        # If no opening parenthesis found, return empty parenthetical
        if parenthetical_start == -1:
            self.parenthetical = ""
            return
            
        parenthetical_end = self.original_text.find(")", parenthetical_start)
        
        # If no closing parenthesis found, return empty parenthetical
        if parenthetical_end == -1:
            self.parenthetical = ""
            return
            
        self.parenthetical = self.original_text[parenthetical_start + 1:parenthetical_end]