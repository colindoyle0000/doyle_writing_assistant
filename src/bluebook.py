"""BluebookCitation Class for storing bluebook citation information about a particular source."""

import os
import logging

# Set up logger
logger = logging.getLogger('writing_assistant')

class BluebookCitation:
    """Class for storing bluebook citation information about a particular source."""
    
    def __init__(
        self,
        citekey,
        bluebook="",
        shortcite="",
        hereinafter="",
        supra_eligible=True,
    ):
        """Initialize a BluebookCitation instance."""
        self.citekey = citekey
        self.bluebook = bluebook
        self.shortcite = shortcite
        self.hereinafter = hereinafter
        self.supra_eligible = supra_eligible
        logger.info(f"Created BluebookCitation instance for {self.citekey}")
    
    def __repr__(self):
        return f"BluebookCitation(citekey={self.citekey}, bluebook={self.bluebook})"
        