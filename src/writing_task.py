"""WritingTask class for performing tasks in the writing stage."""

import logging
import os
import re
import copy
import PyPDF2
from docx import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from src.utils_tokens import num_tokens, list_to_token_list
from src.utils_llm import (
    llm_call, 
    query_db, 
    filter_results_by_similarity, 
    query_db_and_filter, 
    get_documents_from_results,
    get_metadata_from_results
)
from src.utils_file import (
    get_root_dir,
    extract_from_pdf,
    extract_from_docx,
    extract_from_md,
    export_to_md,
    save_outputs,
)
from src.utils_string import (
    get_timestamp_as_string,
    get_timestamp,
)
from src.source_excerpt import SourceExcerpt

from src.llm_call_data import LLMCallData

from src.section import Section


# Set up logger
logger = logging.getLogger('writing_assistant')

class WritingTask:
    """Class for performing tasks in the writing stage."""
    
    def __init__(
        self,
        task_name: str,
        project,
    ):
        """Initialize a WritingTask object."""
        self.task_name = task_name
        self.project = project
        
        # Notetaking variables
        self.llm_notes_lst = []
        self.last_llm_notes = ""
        
        # Outlining variables
        self.llm_outlines_lst = []
        self.last_llm_outline = ""
        self.last_llm_outline_lst = []
        
        # Writing variables
        self.outline = ""
        self.outline_lst = []
        self.sections_lst = []
        self.drafts_lst = []
        self.last_draft = ""
        
        # Edited draft variables
        self.llm_edits_lst = []
        self.last_llm_edit = ""
        
        # Citations variables
        self.citation_notes_lst = []
        self.last_citation_notes = ""
        self.claims_lst = []
        
        # Quotes variables
        self.quotes_notes_lst = []
        self.last_quotes_notes = ""
        
        # Plagiarism variables
        self.plagiarism_notes_lst = []
        self.last_plagiarism_notes = ""
        
        # LLM call variables
        # These are used to store data for current LLM call
        self.prompt_system = ""
        self.prompt_user = ""
        self.query = ""
        self.manual_notes = ""
        # Notes being passed to the LLM
        self.num_notes = 0
        self.notes_lst = []
        # Metadata on those notes
        self.notes_metadata_lst = []
        # The original retrieved notes before any reduction
        self.notes_results_dict = {}
        # Sources being passed to the LLM
        self.num_sources = 0
        self.sources_lst = []
        # Metadata on those sources
        self.sources_metadata_lst = []
        # The original retrieved sources before any reduction
        self.sources_results_dict = {}
        # Similarity threshold for filtering notes and sources
        self.similarity_threshold = 1.5
        # Reduce excerpts based on relevance to query?
        self.reduce_excerpts = False
        # Use the hypothetical retrieval method?
        self.use_hypo_query = True
        # All of the data from the LLM call
        self.llm_call_data = []
        
        self.creation_timestamp = get_timestamp()
        self.creation_timestamp_string = get_timestamp_as_string()
    
    # Set attributes
    def set_attributes(
        self,
        query=None,
        manual_notes=None,
        num_notes=None,
        num_sources=None,
        similarity_threshold=None,
        reduce_excerpts=None,
        use_hypo_query=None,
    ):
        """Set attributes for the WritingTask."""
        if query is not None:
            self.query = query
        if manual_notes is not None:
            self.manual_notes = manual_notes
        if num_notes is not None:
            self.num_notes = num_notes
        if num_sources is not None:
            self.num_sources = num_sources
        if similarity_threshold is not None:
            self.similarity_threshold = similarity_threshold
        if reduce_excerpts is not None:
            self.reduce_excerpts = reduce_excerpts
        if use_hypo_query is not None:
            self.use_hypo_query = use_hypo_query
    
    def set_attributes_from_file(
        self,
        manual_notes=False,
        outline=False,
        query=False,
    ):
        """Set attributes for the WritingTask from a file."""
        if manual_notes is True:
            manual_notes_path = os.path.join(self.project.project_path, "manual_notes.md")
            self.manual_notes = extract_from_md(manual_notes_path)
        if outline is True:
            outline_path = os.path.join(self.project.project_path, "outline.md")
            self.outline = extract_from_md(outline_path)
        if query is True:
            query_path = os.path.join(self.project.project_path, "query.md")
            self.query = extract_from_md(query_path)
      
    def replace_prompt_placeholders(self):
        """Replace placeholders in the system prompt."""
        # Replace general placeholders
        # Load general placeholders from prompts folder
        writing_style = extract_from_md(
            os.path.join(get_root_dir(), "data", "prompts", "writing_style.md")
        )
        preface = extract_from_md(
            os.path.join(get_root_dir(), "data", "prompts", "preface.md")
        )
        ideas = extract_from_md(
            os.path.join(get_root_dir(), "data", "prompts", "ideas.md")
        )
        # Replace general placeholders
        self.prompt_system = self.prompt_system.replace("{writing_style}", writing_style)
        self.prompt_system = self.prompt_system.replace("{preface}", preface)
        self.prompt_system = self.prompt_system.replace("{ideas}", ideas)
        
        # Replace project-specific placeholders
        self.prompt_system = self.prompt_system.replace("{project_name}", self.project.project_name)
        self.prompt_system = self.prompt_system.replace("{query}", self.query)
        
        if len(self.manual_notes) == 0:
            logger.debug("No manual notes.")
            self.prompt_system = self.prompt_system.replace("{manual_notes}", "")
        else:
            logger.debug("Manual notes found. Adding to prompt if needed.")
            self.prompt_system = self.prompt_system.replace("{manual_notes}", f"Here are the notes you've already taken on this:\n\n{self.manual_notes}\n\n")
        
        if self.project.abstract is None:
            logger.debug("No abstract.")
            self.prompt_system = self.prompt_system.replace("{abstract}", "")
        else:
            logger.debug("Abstract found. Adding to prompt if needed.")
            abstract = f"Our abstract is\n\n{self.project.abstract}\n\n"
            self.prompt_system = self.prompt_system.replace("{abstract}", abstract)

        if self.project.structure is None:
            logger.debug("No structure.")
            self.prompt_system = self.prompt_system.replace("{structure}", "")
        else:
            logger.debug("Structure found. Adding to prompt if needed.")
            structure = f"Our overall article structure is:\n\n{self.project.structure}\n\n"
            self.prompt_system = self.prompt_system.replace("{structure}", structure)
        
        if self.outline is None:
            logger.debug("No outline.")
            self.prompt_system = self.prompt_system.replace("{outline}", "")
        else:
            logger.debug("Outline found. Adding to prompt if needed.")
            outline = f"The outline for this part of the paper is:\n\n{self.outline}\n\n"
            self.prompt_system = self.prompt_system.replace("{outline}", outline)

    def replace_prompt_notes_sources(self):
        """Replace placeholders in the system prompt."""
        if len(self.notes_lst) == 0:
            self.prompt_system = self.prompt_system.replace("{notes}", "")
        else:
            notes = "\n\n".join(self.notes_lst)
            self.prompt_system = self.prompt_system.replace("{notes}", f"Excerpts from potentially relevant notes that we've taken are:\n\n{notes}\n\n")
        
        if len(self.sources_lst) == 0:
            self.prompt_system = self.prompt_system.replace("{sources}", "")
        else:
            sources = "\n\n".join(self.sources_lst)
            self.prompt_system = self.prompt_system.replace("{sources}", f"Excerpts from potentially relevant sources are:\n\n{sources}\n\n")
        
    def reduce_notes_sources_for_tokens(self):
        """Reduce notes and sources to fit within token limit."""
        # Calculate initial token count
        tokens_sum = num_tokens(self.prompt_system)
        tokens_sum += sum(num_tokens(note) for note in self.notes_lst)
        tokens_sum += sum(num_tokens(source) for source in self.sources_lst)    
        initial_notes = len(self.notes_lst)
        initial_sources = len(self.sources_lst)

        # Reduce until under token limit
        while tokens_sum > self.project.llm_settings.max_tokens:
            if len(self.notes_lst) > len(self.sources_lst) and self.notes_lst:
                note = self.notes_lst.pop()
                tokens_sum -= num_tokens(note)
            elif self.sources_lst:
                source = self.sources_lst.pop()
                tokens_sum -= num_tokens(source)
            else:
                break
        
        if initial_notes > len(self.notes_lst):
            logger.info(f"Reduced notes from {initial_notes} to {len(self.notes_lst)}")
        if initial_sources > len(self.sources_lst):
            logger.info(f"Reduced sources from {initial_sources} to {len(self.sources_lst)}")

    def clean_up_prompt(
        self
    ):
        """Run all prompt cleaning methods."""
        self.replace_prompt_placeholders()
        self.replace_prompt_notes_sources()
        self.reduce_notes_sources_for_tokens()

    def reduce_excerpt(
        self,
        excerpt_text: str,
    ):
        """Call on llm_cheap to copy from an excerpt only the information relevant to the query."""
        logger.info(f"Running reduce_excerpt.")
        # Build up the system prompt
        prompt_system = extract_from_md(
            os.path.join(get_root_dir(), "data", "prompts", "reduce_excerpt.md")
        )
        prompt_system += "\n\n"
        # Replace {excerpt_text} placeholder with the excerpt text
        prompt_system = prompt_system.replace("{excerpt_text}", excerpt_text)
        # Replace {query} placeholder with the query
        prompt_system = prompt_system.replace("{query}", self.query)
        # Create user prompt
        prompt_user = "Copy relevant information from the excerpt."
        # Set up prompts for the LLM
        prompts = [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt_user},
        ]
        # Call the LLM
        llm_response = llm_call(
            model=self.project.llm_settings.model_cheap,
            prompts=prompts,
            settings=self.project.llm_settings,
        )
        llm_call_data = LLMCallData(
            function_name="reduce_excerpt",
            query=self.query,
            llm_response=llm_response,
        )
        self.llm_call_data.append(llm_call_data)

        logger.info(f"Reduced excerpt from {num_tokens(excerpt_text)} to {num_tokens(llm_response.output)} tokens.")
        return llm_response

    def reduce_excerpts_lst(
        self,
        excerpts_lst,
    ):
        """Reduce a list of excerpts to only the information relevant to the query."""
        reduced_excerpts = []
        for excerpt in excerpts_lst:
            response = self.reduce_excerpt(
                excerpt_text=excerpt
            )
            reduced_excerpts.append(response.output)
        return reduced_excerpts
    
    # Retrieval methods
    
    def get_excerpt_metadata(
        self,
        metadata_entry: list,
    ):
        """Retrieve excerpt metadata from metadata list."""
        citekey = metadata_entry['citekey']
        excerpt_index = metadata_entry['excerpt_index']
        
        source = self.project.sources.sources_dict[citekey]
        excerpt = source.excerpts[excerpt_index]
        
        # If the adjusted page number is not set, set it
        if excerpt.adjusted_page_number is None:
            excerpt.get_page_number()
            excerpt.get_adjusted_page_number()
        page_number = excerpt.adjusted_page_number

        # Try to get title and author from source.fields dictionary, but default to empty string
        title = excerpt.source.fields.get('title', "")
        author = excerpt.source.fields.get('author', "")
        metadata = f"## Citekey: {citekey} \nTitle: {title} \nAuthor: {author} \nPage: {page_number} \n\n"
        return metadata

    def get_full_source_from_excerpt(
        self,
        excerpt_text: str,
    ):
        """Retrieve source from excerpt."""
        pass

    def retrieve_notes(
        self,
        query: str=None,
        ):
        """Retrieve notes from vector_db."""
        if query is not None:
            query = query
        else:
            query = self.query
                    
        logger.info(f"Running retrieve_notes for query: {query}")
        self.notes_results_dict = query_db(
            db=self.project.notes.notes_db,
            query_text=query,
            num_results=self.num_notes,
        )
        if self.similarity_threshold is not None:
            self.notes_results_dict = filter_results_by_similarity(
                results=self.notes_results_dict,
                similarity_threshold=self.similarity_threshold,
            )
        self.notes_metadata_lst = get_metadata_from_results(
            results=self.notes_results_dict,
        )
        
        self.notes_lst = get_documents_from_results(
            results=self.notes_results_dict,
        )
        if self.reduce_excerpts is True:
            self.notes_lst = self.reduce_excerpts_lst(
                excerpts_lst=self.notes_lst,
            )
        logger.info(f"Number of notes retrieved: {len(self.notes_lst)}")

    def retrieve_sources(
        self,
        query: str=None,
        ):
        """Retrieve excerpts from vector_db."""
        if query is not None:
            query = query
        else:
            query = self.query
        logger.info(f"Running retrieve_sources for query: {query}")
        # Retrieve sources from vector_db
        self.sources_results_dict = query_db(
            db=self.project.sources.sources_db,
            query_text=query,
            num_results=self.num_sources,
        )
        # Filter results by similarity
        if self.similarity_threshold is not None:
            self.sources_results_dict = filter_results_by_similarity(
                results=self.sources_results_dict,
                similarity_threshold=self.similarity_threshold,
            )
        # Get metadata and documents from results
        self.sources_metadata_lst = get_metadata_from_results(
            results=self.sources_results_dict,
        )
        # Get sources from results
        self.sources_lst = get_documents_from_results(
            results=self.sources_results_dict,
        )
        logger.info(f"Number of sources retrieved: {len(self.sources_lst)}")

        if self.reduce_excerpts is True:
            self.sources_lst = self.reduce_excerpts_lst(
                excerpts_lst=self.sources_lst,
            )

        # Add metadata to sources_lst
        for i, metadata_entry in enumerate(self.sources_metadata_lst):
            metadata = self.get_excerpt_metadata(
                metadata_entry=metadata_entry,
            )
            self.sources_lst[i] = metadata + self.sources_lst[i]

    def hypo_retrieval(self):
        """Call llm_cheap to write a hypothetical text and then retrieve similar notes and sources.
        Structure of prompt:
        {preface}
        
        Write a few paragraphs on the topic of:

        {query}
        """
        
        logger.info(f"Running hypo_retrieval for query: {self.query}")
        
        # Build up the system prompt
        prompt_system = extract_from_md(
            os.path.join(get_root_dir(), "data", "prompts", "hypo_retrieval.md")
        )
        # Replace {query} placeholder with the query
        prompt_system = prompt_system.replace("{query}", self.query)
        
        self.clean_up_prompt()
        
        # Create user prompt
        prompt_user = "Write this."
        # Set up prompts for the LLM
        prompts = [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt_user},
        ]
        

        
        # Call the LLM
        hypo_response = llm_call(
            model=self.project.llm_settings.model_cheap,
            prompts=prompts,
            settings=self.project.llm_settings,
        )
        # Retrieve notes
        if self.num_notes > 0:
            self.retrieve_notes(query=hypo_response.output)

        # Retrieve sources
        if self.num_sources > 0:
            self.retrieve_sources(query=hypo_response.output)
            
        save_outputs(
            directory=self.project.output_directory,
            filename="hypo_retrieval",
            responses=[hypo_response],
            meta_info=f"Query: {self.query} \n",
        )
        llm_call_data = LLMCallData(
            function_name="hypo_retrieval",
            query=self.query,
            notes_lst=self.notes_lst,
            sources_lst=self.sources_lst,
            notes_metadata_lst=self.notes_metadata_lst,
            sources_metadata_lst=self.sources_metadata_lst,
            notes_results_dict=self.notes_results_dict,
            sources_results_dict=self.sources_results_dict,
            llm_response=hypo_response,
        )
        self.llm_call_data.append(llm_call_data)


    
    def output_source_excerpts(
        self,
    ):
        """Save source excerpts to an .md file."""
        sources_text = ""
        # Combine the source excerpts into a single string
        for source in self.sources_lst:
            sources_text += source
            sources_text += "\n\n"
        # Save the source excerpts to an .md file
        export_to_md(
            directory=self.project.output_directory,
            filename=f"{self.task_name}_source_excerpts",
            text=sources_text,
        )

    def output_notes_excerpts(
        self,
    ):
        """Save notes excerpts to an .md file."""
        notes_text = ""
        # Combine the source excerpts into a single string
        for note in self.notes_lst:
            notes_text += note
            notes_text += "\n\n"
        # Save the source excerpts to an .md file
        export_to_md(
            directory=self.project.output_directory,
            filename=f"{self.task_name}_notes_excerpts",
            text=notes_text,
        )
    
    # Notetaking methods
        
    def take_notes_on_source_excerpts(
        self,
        cheap=False,
    ):
        """Take notes on source excerpts.
        Query should be full instruction on what to do.
        Example: "Take any notes on these sources that will help us to write a section of the paper on the topic of ______."
        
        Structure of prompt:
        {preface}
        
        {abstract}

        {query}

        [Instructions on how to write citations]
        
        {notes}

        {sources}

        {manual_notes}
        """
        
        # Build up the system prompt
        self.prompt_system = extract_from_md(
            os.path.join(get_root_dir(), "data", "prompts", "take_notes.md")
        )
        
        self.clean_up_prompt()
        
        # Create user prompt
        self.prompt_user = "Take notes on these sources."
        
        # Set up prompts for the LLM
        prompts = [
            {"role": "system", "content": self.prompt_system},
            {"role": "user", "content": self.prompt_user},
        ]
        # Call the LLM
        # If cheap is True, use the cheap model
        if cheap == True:
            model = self.project.llm_settings.model_cheap
        else:
            model = self.project.llm_settings.model
        
        llm_response = llm_call(
            model=model,
            prompts=prompts,
            settings=self.project.llm_settings,
        )
        
        # Grab the citekey from the sources (it's the first line of the source, ending with "\n")
        notes_str = ""
        for source in self.sources_lst:
            citekey = source.split("\n")[0]
            # Add the citekey to the notes
            notes_str += citekey
            notes_str += "\n"
        
        notes_str += llm_response.output
        
        self.llm_notes_lst.append(notes_str)
        self.last_llm_notes = notes_str
        
        # Output the notes to an .md file

        full_text = f"## Notes on sources\n\nTask: {self.task_name}\n\nNotes:\n\n{notes_str}"
        export_to_md(
            directory=self.project.output_directory,
            filename=f"{self.task_name}_notes_on_source_excerpts",
            text=full_text,
        )
        
        llm_call_data = LLMCallData(
            function_name="take_notes_on_source_excerpts",
            query=self.query,
            notes_lst=self.notes_lst,
            sources_lst=self.sources_lst,
            notes_metadata_lst=self.notes_metadata_lst,
            sources_metadata_lst=self.sources_metadata_lst,
            notes_results_dict=self.notes_results_dict,
            sources_results_dict=self.sources_results_dict,
            llm_response=llm_response,
        )
        self.llm_call_data.append(llm_call_data)
    
    def take_notes_on_sources_1_by_1(
        self,
        cheap=False,
    ):
        """Take notes on source excerpts one by one.
        Query should be full instruction on what to do.
        Example: "Take any notes from this source that will help us to write a section of the paper on the topic of ______."
        
        Structure of prompt:
        {preface}
        
        {abstract}

        {query}

        [Instructions on how to write citations]
        
        {notes}

        {sources}

        {manual_notes}
        """

        notes_str = ""
        # Build up the system prompt
        self.prompt_system = extract_from_md(
            os.path.join(get_root_dir(), "data", "prompts", "take_notes.md")
        )
        
        # Clean up the prompt, but only by replacing placeholders 
        # self.clean_up_prompt() would insert full list of notes and sources, which we don't want here
        self.replace_prompt_placeholders()
        
        # Create user prompt
        self.prompt_user = "Take notes on this source."
        

        # Call the LLM
        # If cheap is True, use the cheap model
        if cheap == True:
            model = self.project.llm_settings.model_cheap
        else:
            model = self.project.llm_settings.model
        
        for source in self.sources_lst:
            # Set up prompts for the LLM
            # Make prompt_system a copy of the self.prompt_system that can be modified without modifying self.prompt_system
            
            prompt_system_copy = copy.deepcopy(self.prompt_system)
            
            prompt_system_copy += f"\n\nThe source:\n\n{source}"

            prompts = [
                {"role": "system", "content": prompt_system_copy},
                {"role": "user", "content": self.prompt_user},
            ]
            
            llm_response = llm_call(
                model=model,
                prompts=prompts,
                settings=self.project.llm_settings,
            )
            # Grab the citekey from the source (it's the first line of the source, ending with "\n")
            citekey = source.split("\n")[0]
            # Add the citekey to the notes
            notes_str += citekey
            notes_str += "\n"
            notes_str += llm_response.output
            notes_str += "\n\n"
            
            llm_call_data = LLMCallData(
                function_name="take_notes_on_sources_1_by_1",
                query=self.query,
                notes_lst=self.notes_lst,
                sources_lst=self.sources_lst,
                notes_metadata_lst=self.notes_metadata_lst,
                sources_metadata_lst=self.sources_metadata_lst,
                notes_results_dict=self.notes_results_dict,
                sources_results_dict=self.sources_results_dict,
                llm_response=llm_response,
            )
            self.llm_call_data.append(llm_call_data)
            
        self.llm_notes_lst.append(notes_str)
        self.last_llm_notes = notes_str
        
        # Output the notes to an .md file
        
        full_text = f"## Notes on sources\n\nTask: {self.task_name}\n\nNotes:\n\n{notes_str}"
        export_to_md(
            directory=self.project.output_directory,
            filename=f"{self.task_name}_notes_on_sources",
            text=full_text,
        )

    def organize_notes(
        self,
        notes_lst: list=None,
    ):
        """Organize LLM notes.
        Query should be full instruction on what to do.
        Example: "Your job is to organize these notes on the topic of ____."
        
        Prompt structure:
        {preface}
        
        {query}
        
        [Instructions on how to write citations]
        
        {notes}
        
        {sources}
        
        {manual_notes}
        
        """
        # Build up the system prompt
        self.prompt_system = extract_from_md(
            os.path.join(get_root_dir(), "data", "prompts", "organize_notes.md")
        )
        
        self.notes_lst = []
        self.sources_lst = []
        if notes_lst is None:
            for note in self.llm_notes_lst:
                self.notes_lst.append(note)
        else:
            self.notes_lst = notes_lst
            
        self.clean_up_prompt()
        
        # Create user prompt
        self.prompt_user = "Organize these notes."

        # Set up prompts for the LLM
        prompts = [
            {"role": "system", "content": self.prompt_system},
            {"role": "user", "content": self.prompt_user},
        ]
        # Call the LLM
        llm_response = llm_call(
            model=self.project.llm_settings.model,
            prompts=prompts,
            settings=self.project.llm_settings,
        )
        
        self.llm_notes_lst.append(llm_response.output)
        self.last_llm_notes = llm_response.output
        
        save_outputs(
            directory=self.project.output_directory, 
            filename=f"{self.task_name}_organize_notes", 
            responses=llm_response,
        )
        llm_call_data = LLMCallData(
            function_name="organize_notes",
            query=self.query,
            notes_lst=self.notes_lst,
            llm_response=llm_response,
        )
        self.llm_call_data.append(llm_call_data)
        
    # Outlining methods
    
    def outline_cot(
        self,
    ):
        """Chain of thoughts for writing an outline.
        self.query should be the part of the paper that you need to take notes on outlining.
        self.manual_notes should be any notes that the LLM should consider when writing the outline.
        
        Structure of prompt:
        {preface}
        {ideas}
        It’s your job to take notes on an outline for part of the paper.

        {abstract}

        {structure}

        The part of the paper that you need to take notes on outlining is:

        {query}

        As a first step, take some notes on the different points that ought to be included in this part of the paper. What do we need to make sure to convey to our readers? What may or may not be worth including?

        Don’t include in the outline topics that will be addressed in other parts of the paper.

        Don’t number the parts of the outline.

        {notes}

        {sources}

        {manual_notes}

        Let’s think this through step-by-step.
        """
        # Build up the system prompt
        self.prompt_system = extract_from_md(
            os.path.join(get_root_dir(), "data", "prompts", "outline_cot.md")
        )
        
        self.clean_up_prompt()
        
        # Create user prompt
        self.prompt_user = "Take some notes on the different points that ought to be included in this part of the paper."
        
        # Set up prompts for the LLM
        prompts = [
            {"role": "system", "content": self.prompt_system},
            {"role": "user", "content": self.prompt_user},
        ]
        # Call the LLM
        llm_response = llm_call(
            model=self.project.llm_settings.model,
            prompts=prompts,
            settings=self.project.llm_settings,
        )
        self.llm_outlines_lst.append(llm_response.output)
        self.last_llm_outline = llm_response.output
        
        save_outputs(
            directory=self.project.output_directory, 
            filename=f"{self.task_name}_outline_cot", 
            responses=llm_response,
            meta_info=f"Number of notes: {self.num_notes} \nNumber of sources: {self.num_sources}\n\n",
        )
        llm_call_data = LLMCallData(
            function_name="outline_cot",
            query=self.query,
            llm_response=llm_response,
            notes_lst=self.notes_lst,
            sources_lst=self.sources_lst,
            notes_metadata_lst=self.notes_metadata_lst,
            sources_metadata_lst=self.sources_metadata_lst,
            notes_results_dict=self.notes_results_dict,
            sources_results_dict=self.sources_results_dict,
        )
        self.llm_call_data.append(llm_call_data)

    def outline_write(self):
        """Write an outline.
        self.query should be the part of the paper that you need to outline.
        self.manual_notes should be any notes that the LLM should consider when writing the outline.
        
        Structure of prompt:
        {preface}
        {ideas}
        It’s your job to draft an outline for part of the paper.

        {abstract}

        {structure}

        The part of the paper that you need to outline is:

        {query}

        Your output should consist only of the different points to be made in this section of the paper with each line representing a paragraph-sized point that should be made. Do not include any text in your output except for the outline. Each line of the outline should be the topic sentence of a new paragraph.

        Don’t include in the outline topics that will be addressed in other parts of the paper.

        Don’t number the parts of the outline.

        {notes}

        {sources}

        {manual_notes}
        """
        # Build up the system prompt
        self.prompt_system = extract_from_md(
            os.path.join(get_root_dir(), "data", "prompts", "outline_write.md")
        )
        
        self.clean_up_prompt()
        
        # Create user prompt
        self.prompt_user = "Write an outline."
        
        # Set up prompts for the LLM
        prompts = [
            {"role": "system", "content": self.prompt_system},
            {"role": "user", "content": self.prompt_user},
        ]
        # Call the LLM
        llm_response = llm_call(
            model=self.project.llm_settings.model,
            prompts=prompts,
            settings=self.project.llm_settings,
        )
        self.llm_outlines_lst.append(llm_response.output)
        self.last_llm_outline = llm_response.output
        
        save_outputs(
            directory=self.project.output_directory, 
            filename=f"{self.task_name}_outline_write", 
            responses=llm_response,
            meta_info=f"Number of notes: {self.num_notes} \nNumber of sources: {self.num_sources}\n\n",
        )
        llm_call_data = LLMCallData(
            function_name="outline_write",
            query=self.query,
            llm_response=llm_response,
            notes_lst=self.notes_lst,
            sources_lst=self.sources_lst,
            notes_metadata_lst=self.notes_metadata_lst,
            sources_metadata_lst=self.sources_metadata_lst,
            notes_results_dict=self.notes_results_dict,
            sources_results_dict=self.sources_results_dict,
        )
        self.llm_call_data.append(llm_call_data)

    def outline_cot_and_write(self):
        """Chain of thoughts for writing an outline and then write the outline."""
        self.outline_cot()
        self.manual_notes = self.last_llm_outline
        self.outline_write()
    
    # Writing methods
    
    def set_outline(self, from_file=False):
        """Set the outline text.
        Can load from file or use the last outline generated by the LLM.
        """
        if from_file is True:
            outline_path = os.path.join(self.project.project_path, "outline.md")
            self.outline = extract_from_md(outline_path, "outline.md")
        else:
            self.outline = self.last_llm_outline
    
    def split_outline(self):
        """Split the outline into a list of sections."""
        # Start by using LLM to create *** separated sections
        # Build up the system prompt
        prompt_system = extract_from_md(
            os.path.join(get_root_dir(), "data", "prompts", "split_outline.md")
        )
        
        # Add the outline to the prompt
        prompt_system = prompt_system.replace("{outline}", self.outline)
        
        # Create user prompt
        prompt_user = "Split the outline into parts."
        
        # Set up prompts for the LLM
        prompts = [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt_user},
        ]
        # Call the LLM
        llm_response = llm_call(
            model=self.project.llm_settings.model_cheap, 
            prompts=prompts, 
            settings=self.project.llm_settings
        )
        llm_call_data = LLMCallData(
            function_name="split_outline",
            llm_response=llm_response,
        )
        self.llm_call_data.append(llm_call_data)
        
        self.outline = llm_response.output
        
        # Split the outline into a list based on '***' as the separator
        self.outline_lst = self.outline.split("***")
        # Remove any items from the list that are less than seven characters long
        self.outline_lst = [item for item in self.outline_lst if len(item) > 7]
        logger.info(f"Split outline into {len(self.outline_lst)} parts")
        outline_text_str = "# Outline Parts\n\n"
        for index, item in enumerate(self.outline_lst):
            outline_text_str += f"[{index}] {item}\n\n"
        export_to_md(
            directory=self.project.output_directory,
            filename=f"{self.task_name}_outline_parts",
            text=outline_text_str,
        )
    
    def setup_sections_from_outline(self):
        """Set up sections for writing."""
        # Create a Section object for each item in the outline_lst
        for index, line in enumerate(self.outline_lst):
            section = Section(
                outline_text=line,
                index=index,
                writing_task=self,
            )
            # Add the section to the sections_lst
            self.sections_lst.append(section)
    
    def write_section(
        self,
        index: int,
        # Boolean for whether to load notes from file
        from_file=False,
        model='claude-3-5-sonnet-20241022',
        num_sections_before: int=3,
    ):
        """Write a section.
        Can load notes from file using notes.md in the project directory.
        Otherwise, if either num_notes and num_sources are set, will retrieve notes and sources based on the outline text and take notes on them.
        Query should be the outline text for the section.
        Model is Claude 3.5 Sonnet by default because it is a good model for writing in a particular style.
        
        num_sections_before is the number of sections before the current section whose text should be included in the prompt.
        
        Structure of prompt:
        {section.prior_text}
        
        {preface}
        
        {ideas}
        
        It’s your job to write a section of this paper.

        {abstract}

        The outline for this part of the paper is:

        {outline}

        Within that outline, the topic that you need to write a paragraph about is:

        {query}
        
        [General writing instructions.]

        {manual_notes}

        {prior_text}
        """
        llm_notes_on_sources = ""
        llm_notes_on_notes = ""
        section = self.sections_lst[index]
        
        # Set query to the outline text for the section
        self.query = section.outline_text
        
        # If from_file is True, load notes from file
        if from_file is True:
            notes_path = os.path.join(self.project.project_path, "notes.md")
            self.manual_notes = extract_from_md(notes_path, "notes.md")
        else:
            # If self.num_sources is > 0, retrieve sources and take notes
            if self.num_sources > 0:
                self.retrieve_sources()
                self.query = f"Take any notes on these sources that will help us to write a section of the paper on the topic of {section.outline_text}."
                self.take_notes_on_source_excerpts()
                llm_notes_on_sources = f"## Notes on sources\n\n{self.last_llm_notes}\n\n"
            # If self.num_notes is > 0, retrieve notes and take notes
            if self.num_notes > 0:
                self.retrieve_notes()
                self.query = f"Take any notes on these notes that will help us to write a section of the paper on the topic of {section.outline_text}."
                self.take_notes_on_source_excerpts()
                llm_notes_on_notes = f"## Notes on notes\n\n{self.last_llm_notes}\n\n"
            # Make manual notes the combination of notes on sources and notes
            # If both are empty, it will just be an empty string.
            self.manual_notes = llm_notes_on_sources + llm_notes_on_notes
            # Reset the query back to the line in the outline
            self.query = section.outline_text
        
        # Build up the system prompt
        self.prompt_system = extract_from_md(
            os.path.join(get_root_dir(), "data", "prompts", "section_write.md")
        )

        # Replace {prior_text} placeholder with the prior text
        if index == 0:
            prior_text = ""
        else:
            prior_text = ""
            for i in range(index - num_sections_before, index):
                prior_text += self.sections_lst[i].last_draft
                prior_text += "\n\n"
            prior_text = f"The paragraphs immediately preceding the section you are writing are: \n\n{prior_text}."
        
        self.prompt_system = self.prompt_system.replace("{prior_text}", prior_text)
        
        self.clean_up_prompt()
        
        # Create user prompt
        self.prompt_user = "Write this section of the paper."
        
        # Set up prompts for the LLM
        prompts = [
            {"role": "system", "content": self.prompt_system},
            {"role": "user", "content": self.prompt_user},
        ]
        # Call the LLM
        llm_response = llm_call(
            model=model,
            prompts=prompts,
            settings=self.project.llm_settings,
        )
        section.drafts_lst.append(llm_response.output)
        section.last_draft = llm_response.output
        
        save_outputs(
            directory=self.project.output_directory, 
            filename=f"{self.task_name}_write_section_{index}", 
            responses=llm_response,
            meta_info=f"Number of notes: {self.num_notes} \nNumber of sources: {self.num_sources}\n\n",
        )
        
        llm_call_data = LLMCallData(
            function_name="write_section",
            query=self.query,
            notes_lst=self.notes_lst,
            sources_lst=self.sources_lst,
            notes_metadata_lst=self.notes_metadata_lst,
            sources_metadata_lst=self.sources_metadata_lst,
            notes_results_dict=self.notes_results_dict,
            sources_results_dict=self.sources_results_dict,
            llm_response=llm_response,
        )
        self.llm_call_data.append(llm_call_data)
        
    def write_sections(
        self,
        # Boolean for whether to load notes from file
        from_file=False,
        model='claude-3-5-sonnet-20241022',
        num_sections_before: int=3,
    ):
        """Write sections.
        Can load notes from file using notes.md in the project directory.
        Otherwise, if either num_notes and num_sources are set, will retrieve notes and sources based on the outline text and take notes on them.
        
        Model is Claude 3.5 Sonnet by default because it is a good model for writing in a particular style.
        
        num_sections_before is the number of sections before the current section whose text should be included in the prompt.
        """
        for index, section in enumerate(self.sections_lst):
            self.write_section(
                index=index,
                from_file=from_file,
                model=model,
                num_sections_before=num_sections_before,
            )
    
    def output_drafts_from_sections(
        self,
    ):
        """Output drafts from sections to one .md file."""
        # Combine the drafts into a single string
        full_draft = f"# Drafts\n\nTask: {self.task_name}\n\n"
        for section in self.sections_lst:
            full_draft += f"## {section.outline_text}\n\n"
            count = 1
            for draft in section.drafts_lst:
                full_draft += f"### Draft {count}\n\n{draft}\n\n"
                count += 1
        # Save the drafts to an .md file
        export_to_md(
            directory=self.project.output_directory,
            filename=f"{self.task_name}_section_drafts_combined",
            text=full_draft,
        )
    
    def write_draft(
        self,
        model='gpt-4o',
        ):
        """Write a draft (not from outline).
        self.query = the LLM's job
        Examples:
            "Write a draft of the introduction to the paper."
            "Write three paragraphs on the topic of ______."

        self.manual_notes will be used if num_notes and num_sources are 0
        
        self.num_notes - number of notes to retrieve
        self.num_sources - number of sources to retrieve
        
        Model is gpt-4o by default because it has more flexibility with context length than Claude.
        
        Structure of prompt:
        {preface}
        {ideas}
        It’s your job to write a draft of a part of this paper.
        {abstract}
        {query}
        [Writing instructions to focus only on this issue.]
        {manual_notes}
        """

        # Save the original query by setting query to a copy of self.query
        query = copy.deepcopy(self.query)
        llm_notes_on_sources = ""
        llm_notes = ""
        
        # If self.num_sources is > 0, retrieve sources and take notes
        if self.num_sources > 0:
            self.retrieve_sources()
            self.query = f"Take any notes on these sources that will help us to write a section of the paper on the topic of {self.query}."
            self.take_notes_on_source_excerpts()
            llm_notes_on_sources = f"## Notes on sources\n\n{self.last_llm_notes}\n\n"
        # If self.num_notes is > 0, retrieve notes and take notes
        if self.num_notes > 0:
            self.retrieve_notes()
            # Turn self.notes_lst into a string
            llm_notes = "## Notes\n\n" + "\n\n".join(self.notes_lst)
        if self.num_notes > 0 or self.num_sources > 0:
            # Make manual notes the combination of notes on sources and notes
            # If both are empty, it will just be an empty string.
            self.manual_notes = llm_notes_on_sources + llm_notes
        # Reset the query back to the original query
        self.query = query
        
        # Build up the system prompt
        self.prompt_system = extract_from_md(
            os.path.join(get_root_dir(), "data", "prompts", "write_draft.md")
        )
        
        self.clean_up_prompt()
        
        # Create user prompt
        self.prompt_user = "Write a draft of this part of the paper."
        
        # Set up prompts for the LLM
        prompts = [
            {"role": "system", "content": self.prompt_system},
            {"role": "user", "content": self.prompt_user},
        ]
        # Call the LLM
        llm_response = llm_call(
            model=model,
            prompts=prompts,
            settings=self.project.llm_settings,
        )
        self.drafts_lst.append(llm_response.output)
        self.last_draft = llm_response.output
        
        save_outputs(
            directory=self.project.output_directory, 
            filename=f"{self.task_name}_write_draft", 
            responses=llm_response,
            meta_info=f"Number of notes: {self.num_notes} \nNumber of sources: {self.num_sources}\n\n",
        )
        
        llm_call_data = LLMCallData(
            function_name="write_draft",
            query=query,
            notes_lst=self.notes_lst,
            sources_lst=self.sources_lst,
            notes_metadata_lst=self.notes_metadata_lst,
            sources_metadata_lst=self.sources_metadata_lst,
            notes_results_dict=self.notes_results_dict,
            sources_results_dict=self.sources_results_dict,
            llm_response=llm_response,
        )
        self.llm_call_data.append(llm_call_data)
    
    def write_draft_with_cot(
        self,
        model='gpt-4o',
        ):
        """Write a draft with a chain of thought.
        This method has a chain of thought call before running the write_draft method.
        self.query = the LLM's job
        Examples:
            "Write a draft of the introduction to the paper."
            "Write three paragraphs on the topic of ______."

        self.manual_notes will be used if num_notes and num_sources are 0
        
        self.num_notes - number of notes to retrieve
        self.num_sources - number of sources to retrieve
        
        Model is gpt-4o by default because it has more flexibility with context length than Claude.
        
        Structure of prompt:
        {preface}
        {ideas}
        It’s your job to write notes on what to include in a draft of a part of this paper.
        {abstract}
        {query}
        [Writing instructions to focus only on this issue.]
        {manual_notes}
        Let’s think this through step-by-step, recording your thoughts each step of the way.        
        """
        
        # Save the original query by setting query to a copy of self.query
        query = copy.deepcopy(self.query)
        llm_notes_on_sources = ""
        llm_notes = ""
        
        # If self.num_sources is > 0, retrieve sources and take notes
        if self.num_sources > 0:
            self.retrieve_sources()
            self.query = f"Take any notes on these sources that will help us to write a section of the paper on the topic of {self.query}."
            self.take_notes_on_source_excerpts()
            llm_notes_on_sources = f"## Notes on sources\n\n{self.last_llm_notes}\n\n"
        # If self.num_notes is > 0, retrieve notes and take notes
        if self.num_notes > 0:
            self.retrieve_notes()
            # Turn self.notes_lst into a string
            llm_notes = "## Notes\n\n" + "\n\n".join(self.notes_lst)
        if self.num_notes > 0 or self.num_sources > 0:
            # Make manual notes the combination of notes on sources and notes
            # If both are empty, it will just be an empty string.
            self.manual_notes = llm_notes_on_sources + llm_notes
        # Reset the query back to the original query
        self.query = query
        
        # Build up the system prompt
        self.prompt_system = extract_from_md(
            os.path.join(get_root_dir(), "data", "prompts", "write_draft_cot.md")
        )
        
        self.clean_up_prompt()
        
        # Create user prompt
        self.prompt_user = "Write notes on what to include in a draft of this part of the paper."
        
        # Set up prompts for the LLM
        prompts = [
            {"role": "system", "content": self.prompt_system},
            {"role": "user", "content": self.prompt_user},
        ]
        
        # Call the LLM
        llm_response = llm_call(
            model=model,
            prompts=prompts,
            settings=self.project.llm_settings,
        )
        
        self.llm_notes_lst.append(llm_response.output)
        self.last_llm_notes = llm_response.output
        
        save_outputs(
            directory=self.project.output_directory,
            filename=f"{self.task_name}_write_draft_cot",
            responses=llm_response,
        )
        
        llm_call_data = LLMCallData(
            function_name="write_draft_cot",
            query=self.query,
            notes_lst=self.notes_lst,
            sources_lst=self.sources_lst,
            notes_metadata_lst=self.notes_metadata_lst,
            sources_metadata_lst=self.sources_metadata_lst,
            notes_results_dict=self.notes_results_dict,
            sources_results_dict=self.sources_results_dict,
            llm_response=llm_response,
        )
        self.llm_call_data.append(llm_call_data)
        
        # Write the draft
        self.num_notes = 0
        self.num_sources = 0
        self.manual_notes = self.last_llm_notes
        self.write_draft(
            model=model,
        )


    # Editing methods
    
    def embellish_draft(
        self,
        from_file=False,
        draft=None,
        model='gpt-4o',
        ):
        """Embellish an existing draft by adding material from notes or sources.
        Query = The LLM's job.
        Example: "Add any details from your notes that you think would improve the draft. Be sure to keep the footnotes and citations."
        Default model is gpt-4o for flexibility.
        There are two LLM calls: one for chain-of-thought for edits, one for making edits.
        
        self.manual_notes will be used if num_notes and num_sources are 0
        
        self.num_notes - number of notes to retrieve
        self.num_sources - number of sources to retrieve
        
        CoT prompt structure:
        {preface}
        
        {ideas}
        
        It’s your job to rewrite a draft of a part of this paper.
        Your current task is to take notes on the edits you'd like to make to this draft.
        
        {query}
        
        {draft}
        
        {writing_style}
        
        [Citation instructions.]
        
        {manual_notes}
        
        Let’s think this through step-by-step, recording your thoughts each step of the way.        
        
        embellish_draft prompt structure:
        {preface}
        
        {ideas}
        
        It’s your job to rewrite a draft of a part of this paper based on the notes you've taken for edits to make.
        
        {draft}
        
        {writing_style}
        
        [Citation instructions.]
        
        {manual_notes}        
        """
        # CoT call
        # Set the draft
        if from_file==True:
            draft_path = os.path.join(self.project.project_path, "draft.md")
            draft = extract_from_md(draft_path, "draft.md")
            logger.info(f"Loaded draft from file.")
        elif draft is None:
            draft = self.last_draft
            logger.info(f"No draft specified. Loaded self.last_draft.")
        else:
            logger.info(f"Draft specified by input argument.")
        
        draft = f"Current draft: \n\n {draft}"
        
        # If self.num_sources is > 0, retrieve sources and take notes
        # Save the original query by setting query to a copy of self.query
        query = copy.deepcopy(self.query)
        llm_notes_on_sources = ""
        llm_notes = ""
        
        if self.num_sources > 0:
            self.retrieve_sources(query=draft)
            self.query = f"Take any notes on these sources that will help us to write a section of the paper on the topic of {self.query}."
            self.take_notes_on_source_excerpts()
            llm_notes_on_sources = f"## Notes on sources\n\n{self.last_llm_notes}\n\n"
        # If self.num_notes is > 0, retrieve notes and take notes
        if self.num_notes > 0:
            self.retrieve_notes(query=draft)
            # Turn self.notes_lst into a string
            llm_notes = "## Notes\n\n" + "\n\n".join(self.notes_lst)
        if self.num_notes > 0 or self.num_sources > 0:
            # Make manual notes the combination of notes on sources and notes
            # If both are empty, it will just be an empty string.
            self.manual_notes = llm_notes_on_sources + llm_notes
        # Reset the query back to the original query
        self.query = query
        
        # Build up the system prompt
        self.prompt_system = extract_from_md(
            os.path.join(get_root_dir(), "data", "prompts", "embellish_draft_cot.md")
        )
        
        # Replace {draft} placeholder with the draft
        self.prompt_system = self.prompt_system.replace("{draft}", draft)
        
        self.clean_up_prompt()
        
        # Create user prompt
        self.prompt_user = "Add any details from your notes that you think would improve the draft."
        
        # Set up prompts for the LLM
        prompts = [
            {"role": "system", "content": self.prompt_system},
            {"role": "user", "content": self.prompt_user},
        ]
        
        # Call the LLM
        llm_response = llm_call(
            model=model,
            prompts=prompts,
            settings=self.project.llm_settings,
        )
        
        self.llm_edits_lst.append(llm_response.output)
        self.last_llm_edit = llm_response.output
        
        save_outputs(
            directory=self.project.output_directory,
            filename=f"{self.task_name}_embellish_draft_cot",
            responses=llm_response,
        )
        
        llm_call_data = LLMCallData(
            function_name="embellish_draft_cot",
            query=self.query,
            llm_response=llm_response,
        )
        self.llm_call_data.append(llm_call_data)
        
        # Embellish draft call
        
        # Don't need to change draft variable because it is the same as prior LLM call
        # Manual notes should be the last LLM edit (output of the CoT call)
        self.manual_notes += f"\n\n Your notes on edits to make to this draft: {self.last_llm_edit}"
        
        # Build up the system prompt
        self.prompt_system = extract_from_md(
            os.path.join(get_root_dir(), "data", "prompts", "embellish_draft.md")
        )
        
        self.clean_up_prompt()
        
        # Create user prompt
        self.prompt_user = "Rewrite the draft based on the notes you've taken for edits to make."
        
        # Set up prompts for the LLM
        prompts = [
            {"role": "system", "content": self.prompt_system},
            {"role": "user", "content": self.prompt_user},
        ]
        
        # Call the LLM
        llm_response = llm_call(
            model=model,
            prompts=prompts,
            settings=self.project.llm_settings,
        )
        
        self.llm_edits_lst.append(llm_response.output)
        self.last_llm_edit = llm_response.output
        self.last_draft = llm_response.output
        
        save_outputs(
            directory=self.project.output_directory,
            filename=f"{self.task_name}_embellish_draft",
            responses=llm_response,
        )
        
        llm_call_data = LLMCallData(
            function_name="embellish_draft",
            query=self.query,
            llm_response=llm_response,
        )
        self.llm_call_data.append(llm_call_data)
        
    
    def edit_draft_style(
        self,
        from_file=False,
        draft=None,
        model='claude-3-5-sonnet-20241022',
        style=True,
        ):
        """Edit a draft for style.
        Query = The LLM's job.
        Example: "Edit this draft for style."
        Default model is Claude 3.5 Sonnet because it is a good model for editing.
        
        from_file is a boolean for whether to load the draft from a file.
        draft is the draft to edit. If from_file is true, draft will be ignored.
        If from_file is false and draft is None, self.last_draft will be used.
        
        style is a boolean for whether to include {writing_style} instructions in the prompt. Default is True.
        
        Structure of prompt:
        {preface}

        {ideas}

        It’s your job to edit a draft of a part of this paper.

        {query}
        
        {draft}

        {writing_style}

        {manual_notes}
        """
        # Set the draft
        if from_file==True:
            draft_path = os.path.join(self.project.project_path, "draft.md")
            draft = extract_from_md(draft_path, "draft.md")
            logger.info(f"Loaded draft from file.")
        elif draft is None:
            draft = self.last_draft
            logger.info(f"No draft specified. Loaded self.last_draft.")
        else:
            logger.info(f"Loaded draft from input argument.")
        
        draft = f"Current draft: \n\n {draft}"
        
        # Build up the system prompt
        self.prompt_system = extract_from_md(
            os.path.join(get_root_dir(), "data", "prompts", "edit_draft_style.md")
        )
        
        # Replace {draft} placeholder with the draft
        self.prompt_system = self.prompt_system.replace("{draft}", draft)
        
        # If style is False, remove the {writing_style} placeholder
        if style is False:
            self.prompt_system = self.prompt_system.replace("{writing_style}", "")
        
        self.clean_up_prompt()
        
        # Create user prompt
        self.prompt_user = "Edit this draft."
        
        # Set up prompts for the LLM
        prompts = [
            {"role": "system", "content": self.prompt_system},
            {"role": "user", "content": self.prompt_user},
        ]
        
        # Call the LLM
        llm_response = llm_call(
            model=model,
            prompts=prompts,
            settings=self.project.llm_settings,
        )
        
        self.llm_edits_lst.append(llm_response.output)
        self.last_llm_edit = llm_response.output
        self.last_draft = llm_response.output
        
        save_outputs(
            directory=self.project.output_directory,
            filename=f"{self.task_name}_edit_draft_style",
            responses=llm_response,
        )
        
        llm_call_data = LLMCallData(
            function_name="edit_draft_style",
            query=self.query,
            llm_response=llm_response,
        )
        self.llm_call_data.append(llm_call_data)
    
    def find_citation(
        self,
        ):
        """Find a citation to support a claim.
        self.query = the claim that needs a citation.
        
        self.manual_notes will be used if num_sources is 0
        
        Structure of the prompt:
        {preface}
        Please find a citation to support the following claim:
        {query}
        [Citation instructions.]
        """

        sources_str = ""
        # If self.num_sources is > 0, retrieve sources
        if self.num_sources > 0:
            self.retrieve_sources()
            sources_str = "## Sources\n\n" + "\n\n".join(self.sources_lst)
            self.manual_notes = sources_str
        
        # Build up the system prompt
        self.prompt_system = extract_from_md(
            os.path.join(get_root_dir(), "data", "prompts", "find_citation.md")
        )
        
        self.clean_up_prompt()
        
        # Create user prompt
        self.prompt_user = "Find a citation to support this claim."
        
        # Set up prompts for the LLM
        prompts = [
            {"role": "system", "content": self.prompt_system},
            {"role": "user", "content": self.prompt_user},
        ]
        # Call the LLM
        llm_response = llm_call(
            model=self.project.llm_settings.model,
            prompts=prompts,
            settings=self.project.llm_settings,
        )
        self.citation_notes_lst.append(llm_response.output)
        self.last_citation_notes = llm_response.output
        
        save_outputs(
            directory=self.project.output_directory, 
            filename=f"{self.task_name}_find_citation", 
            responses=llm_response,
            meta_info=f"Number of sources: {self.num_sources}\n\n",
        )
        
        llm_call_data = LLMCallData(
            function_name="find_citation",
            query=self.query,
            sources_lst=self.sources_lst,
            sources_metadata_lst=self.sources_metadata_lst,
            sources_results_dict=self.sources_results_dict,
            llm_response=llm_response,
        )
        self.llm_call_data.append(llm_call_data)
    
    def make_claims_lst(self):
        """Make a list of claims from manual notes."""
        # Take self.manual_notes and split it into a list of claims
        # Each claim should be separated by '\n\n'
        self.claims_lst = self.manual_notes.split("\n\n")
        # Remove any items from the list that are less than seven characters long
        self.claims_lst = [item for item in self.claims_lst if len(item) > 7]
        logger.info(f"Split manual notes into {len(self.claims_lst)} claims")
    
    def find_citation_lst(
        self,
        make_claims_lst=True,
        ):
        """For a list of claims, find citations to support each claim."""
        citations_str = ""
        if make_claims_lst is True:
            self.make_claims_lst()
        for claim in self.claims_lst:
            self.query = claim
            self.find_citation()
            citations_str += f"## Claim: {claim}\n\n{self.last_citation_notes}\n\n"
        export_to_md(
            directory=self.project.output_directory,
            filename=f"{self.task_name}_find_citation_lst",
            text=citations_str,
        )
    
    def check_quotes(self):
        """Check to make sure that any quotes in the draft can be found in the sources."""
        pass
    
    def check_plagiarism(self):
        """Check the draft for text that is a copy of source text or very similar to source text."""
        pass
    

    
