"""Bluebook dictionary class for storing a dictionary of bluebook citation class entries"""

import os
import logging
import time
import pickle

from src.utils_file import get_root_dir
from src.utils_string import get_timestamp_as_string

from src.bluebook import BluebookCitation

# Set up logger
logger = logging.getLogger('writing_assistant')

class BluebookDict:
    """Class for storing a dictionary of bluebook citation class entries."""
    
    def __init__(self):
        """Initialize a BluebookDict instance."""
        self.dict = {}
        logger.info("Created BluebookDict instance")
        self.pkl_directory = os.path.join(get_root_dir(), "data", "bluebook_dict")
        self.pkl_filepath = os.path.join(self.pkl_directory, "bluebook_dict.pkl")
    
    def add_citation(
        self,
        citekey,
        bluebook="",
        shortcite="",
        hereinafter="",
        supra_eligible=True,
    ):
        citation = BluebookCitation(
            citekey=citekey,
            bluebook=bluebook,
            shortcite=shortcite,
            hereinafter=hereinafter,
            supra_eligible=supra_eligible,
        )
        self.dict[citekey] = citation
    
    def save_pkl(
        self,
        name=None
        ):
        """Save the BluebookDict instance as a .pkl file."""
        os.makedirs(self.pkl_directory, exist_ok=True)
        if name is None:
            filepath = self.pkl_filepath
        else:
            filepath = os.path.join(self.pkl_directory, f"{name}.pkl")
        with open(filepath, 'wb') as pkl_file:
            pickle.dump(self, pkl_file)
        logger.info(f"Saved BluebookDict instance as {filepath}")
    
    def load_pkl(self):
        """Load the BluebookDict instance from a .pkl file."""
        try:
            if not os.path.exists(self.pkl_filepath):
                logger.error(f"Pickle file not found: {self.pkl_filepath}")
                return False
                
            with open(self.pkl_filepath, 'rb') as pkl_file:
                data = pickle.load(pkl_file)
                
                # Handle if data is another BluebookDict instance
                if isinstance(data, BluebookDict):
                    data = data.__dict__
                    
                # Validate data is dictionary
                if not isinstance(data, dict):
                    logger.error(f"Loaded data is not a dictionary: {type(data)}")
                    return False
                    
                self.__dict__.update(data)
                logger.info(f"Loaded BluebookDict instance from {self.pkl_filepath}")
                
        except Exception as e:
            logger.error(f"Error loading pickle file: {e}")
