"""Section class for writing one section of a writing task."""

import os
import logging

from src.utils_file import get_root_dir
from src.utils_llm import (
    LLMSettings,
    llm_call,
)
from src.utils_string import (
    get_timestamp_as_string,
    get_timestamp,
)

from src.utils_file import (
    get_root_dir, 
    extract_from_pdf, 
    extract_from_docx, 
    extract_from_md, 
    export_to_md, 
    save_outputs,
)

from src.task_utils import (
    replace_prompt_placeholders,
    retrieve_notes,
    retrieve_sources,
    reduce_notes_sources_for_tokens,
    replace_prompt_notes_sources,
    clean_up_prompt,
    hypo_retrieval,
)


from src.llm_call_data import LLMCallData

# Set up logger
logger = logging.getLogger('writing_assistant')

class Section:
    """Class for writing one section of a writing task."""
    
    def __init__(
        self,
        outline_text: str,
        index: int,
        writing_task,
    ):
        """Initialize a Section object."""
        self.outline_text = outline_text
        self.index = index
        self.writing_task = writing_task
        self.project = self.writing_task.project
        self.drafts_lst = []
        self.llm_call_data = []
        self.last_draft = ""
        self.text_of_prior_sections = ""