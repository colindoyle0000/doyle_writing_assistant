"Project class for supervising an entire project."

import os
import logging
import pickle

from src.utils_file import get_root_dir, extract_from_pdf, extract_from_docx, extract_from_md
from src.utils_llm import (
    LLMSettings,
)
from src.utils_string import get_timestamp_as_string
from src.section import Section
from src.notes import Notes
from src.sources import Sources
from src.writing_task import WritingTask
from src.format_task import FormatTask

from src.task_architect import Architect
from src.task_carpenter import Carpenter
from src.task_librarian import Librarian
# from src.task_madman import Madman
# from src.task_ee import EE
# from src.task_judge import Judge

from src.bluebook_dict import BluebookDict

# Set up logger
logger = logging.getLogger('writing_assistant')

class Project:
    """Class for supervising an entire project."""
    
    def __init__(
        self,
        project_name: str,
        notes_folder: str = "",
        notes_path: str = "",
        sources_folder: str = "",
        bib_file_path: str = "",
        llm_settings: LLMSettings = LLMSettings(),
    ):
        """Initialize a Project object."""
        self.project_name = project_name
        self.project_path = os.path.join(get_root_dir(), "data", "projects", project_name)
        # Notes folder should be the name of the folder where the notes are stored within data/notes/
        self.notes_folder = notes_folder
        # Notes path should be the path to the notes file
        # If notes_path exists, notes_folder is ignored when loading notes
        self.notes_path = notes_path
        # Sources folder should be the name of the folder where the sources are stored within data/sources/
        self.sources_folder = sources_folder
        # Path to the BibTex file
        self.bib_file_path = bib_file_path
        # Path to the .pkl file
        self.pkl_directory = os.path.join(self.project_path, "pkl")
        self.pkl_filepath = os.path.join(self.pkl_directory, f"{self.project_name}.pkl")
        self.bluebook_dict = BluebookDict()
        self.word_doc = None
        self.full_text = None
        self.abstract = None
        self.structure = None
        self.output_directory = os.path.join(self.project_path, "outputs")
        self.llm_settings = llm_settings
        self.notes = None
        self.sources = None
        self.writing_tasks_dict = {}
        self.format_tasks_dict = {}
    
    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove sources_db and notes_db from the state
        if 'sources_db' in state:
            del state['sources_db']
        if 'notes_db' in state:
            del state['notes_db']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
    
    def set_word_doc(self, word_doc_path):
        """Set the Word document for the project."""
        self.word_doc_path = word_doc_path
        self.full_text = extract_from_docx(self.word_doc_path, self.project_name)
    
    def set_abstract(self, abstract_path=None):
        """Set the abstract for the project."""
        if abstract_path is None:
            abstract_path = os.path.join(self.project_path, "abstract.md")
        self.abstract = extract_from_md(abstract_path, "abstract.md")
    
    def set_structure(self, structure_path=None):
        """Set the structure for the project."""
        if structure_path is None:
            structure_path = os.path.join(self.project_path, "structure.md")
        self.structure = extract_from_md(structure_path, "structure.md")
    
    def set_folders(self):
        """Create the project folders."""
        # Create the project folder
        os.makedirs(self.project_path, exist_ok=True)
        # Create the .pkl folder
        os.makedirs(self.pkl_directory, exist_ok=True)
        # Within project folder, create folders for outputs, tasks, notes, and sources
        os.makedirs(os.path.join(self.project_path, "outputs"), exist_ok=True)
        os.makedirs(os.path.join(self.project_path, "tasks"), exist_ok=True)
        os.makedirs(os.path.join(self.project_path, "notes"), exist_ok=True)
        os.makedirs(os.path.join(self.project_path, "sources"), exist_ok=True)
    
    # Loading notes and sources methods
    
    def load_project_notes(self):
        """Load notes for the writing task."""
        self.notes = Notes(project=self)
        self.notes.set_path()
        self.notes.load_notes_from_folder()
        self.notes.create_excerpts_from_notes()
        self.notes.create_chroma_docs()
        self.notes.create_notes_db()
    
    def add_new_notes(self):
        """Add new notes to the project."""
        self.notes.load_new_notes_from_folder()
        self.notes.create_excerpts_from_notes()
        self.notes.add_to_notes_db()
        
    def load_project_sources(self):
        """Load sources for the writing task."""
        self.sources = Sources(project=self)
        self.sources.load_sources_from_bib_file()
        self.sources.create_excerpts_from_sources()
        self.sources.create_chroma_docs()
        self.sources.preprocess_chroma_docs()
        self.sources.create_sources_db()
        
    def add_new_sources(self):
        """Add new sources to the project."""
        self.sources.load_new_sources_from_bib_file()
        self.sources.create_excerpts_from_sources()
        self.sources.add_to_sources_db()
    
    def restore_db(self):
        """Restore the database for the project."""
        self.notes.load_notes_db()
        self.sources.load_sources_db()

    def create_writing_task(
        self,
        task_name: str,
        ):
        """Create a new writing task."""
        writing_task = WritingTask(
            task_name=task_name,
            project=self,
        )
        self.writing_tasks_dict[task_name] = writing_task
    
    def create_format_task(
        self,
        task_name: str,
        ):
        """Create a new format task."""
        format_task = FormatTask(
            task_name=task_name,
            project=self,
        )
        self.format_tasks_dict[task_name] = format_task
    
    def create_madman_task(
        self,
        task_name,
        assignment_str=None,
        assignment_lst=None,
    ):  
        """Create a new Madman task."""
        madman_task = Madman(
            task_name=task_name,
            project=self,
        )
        self.writing_tasks_dict[task_name] = madman_task
    
    def create_architect_task(
        self,
        task_name,
    ):
        """Create a new Architect task."""
        architect_task = Architect(
            task_name=task_name,
            project=self,
        )
        self.writing_tasks_dict[task_name] = architect_task
    
    def create_librarian_task(
        self,
        task_name,
    ):
        """Create a new Librarian task."""
        librarian_task = Librarian(
            task_name=task_name,
            project=self,
        )
        self.writing_tasks_dict[task_name] = librarian_task
    
    def create_carpenter_task(
        self,
        task_name,
    ):
        """Create a new Carpenter task."""
        carpenter_task = Carpenter(
            task_name=task_name,
            project=self,
        )
        self.writing_tasks_dict[task_name] = carpenter_task
    
    
    def save_project_pkl(
        self, 
        pkl_directory=None,
        pkl_filename=None
    ):
        """Save the project as a .pkl file.
        
        Args:
            filepath_pkl (str): The directory path where the .pkl file will be saved.
            name (str): The name of the .pkl file.
        
        Raises:
            Exception: If saving fails due to file permission or path errors.
        """
        if pkl_directory is None:
            pkl_directory = self.pkl_directory
        os.makedirs(pkl_directory, exist_ok=True)
        if pkl_filename is None:
            pkl_filename = self.project_name
        pkl_filepath = os.path.join(pkl_directory, f"{pkl_filename}.pkl")
        with open(pkl_filepath, 'wb') as f:
            pickle.dump(self.__dict__, f)
            logger.info("Saved attributes to .pkl file: %s", pkl_filepath)
        self.pkl_filepath = pkl_filepath

    def load_project_pkl(self, pkl_filepath=None):
        """Load the project from a .pkl file.
        
        Args:
            filepath_pkl (str): The path to the .pkl file.
        
        Raises:
            Exception: If loading fails due to file permission or path errors.
        """
        if pkl_filepath is None:
            pkl_filepath = self.pkl_filepath
        with open(pkl_filepath, 'rb') as f:
            data = pickle.load(f)
            self.__dict__.update(data)
            logger.info("Loaded attributes from .pkl file: %s", pkl_filepath)
