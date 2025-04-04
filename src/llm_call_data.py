"""LLMCallData class for storing information from an LLM call."""

import os
import logging

from src.utils_file import (
    get_root_dir,
    extract_from_pdf,
    extract_from_docx,
    extract_from_md,
    export_to_md,
    save_outputs,
)
from src.utils_llm import (
    LLMSettings,
    llm_call,
)
from src.utils_string import (
    get_timestamp,
    get_timestamp_as_string,
    
)
from src.task_utils import (
    replace_prompt_placeholders,
    retrieve_notes,
    retrieve_sources,
    reduce_notes_sources_for_tokens,
    replace_prompt_notes_sources,
    clean_up_prompt,
    hypo_retrieval
)
from src.utils_tokens import (
    num_tokens,
    list_to_token_list,
)

# Set up logger
logger = logging.getLogger('writing_assistant')

class LLMCallData:
    """Class for storing information from an LLM call."""
    
    def __init__(
        self,
        function_name: str,
        query="",
        hypo_query="",
        notes_lst=[],
        notes_metadata_lst=[],
        notes_results_dict={},
        sources_lst=[],
        sources_metadata_lst=[],
        sources_results_dict={},
        hypo_response=None,
        llm_response=None,
        ):
        """Initialize a CallData object."""
        self.function_name = function_name
        self.query = query
        self.hypo_query = hypo_query
        self.notes_lst = notes_lst
        self.notes_metadata_lst = notes_metadata_lst
        self.notes_results_dict = notes_results_dict
        self.sources_lst = sources_lst
        self.sources_metadata_lst = sources_metadata_lst
        self.sources_results_dict = sources_results_dict
        self.hypo_response = hypo_response
        self.llm_response = llm_response
        self.creation_timestamp = get_timestamp()
        self.creation_timestamp_string = get_timestamp_as_string()