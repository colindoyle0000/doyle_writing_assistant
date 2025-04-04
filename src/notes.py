"""Class for managing previously written human notes for the project."""

import os
import logging
import PyPDF2
from docx import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import chromadb
import chromadb.utils.embedding_functions as embedding_functions

import openai

from src.utils_file import get_root_dir, extract_from_pdf, extract_from_docx, extract_from_md, extract_text_from_multiple_files
from src.utils_llm import llm_call, create_db, load_db, add_to_db
from src.utils_string import get_timestamp
from src.utils_tokens import list_to_token_list, string_to_token_list
from src.note import Note


# Set up logger
logger = logging.getLogger('writing_assistant')

class Notes:
    """Class for managing previously written human notes for the project."""
    
    def __init__(self, project):
        """Initialize a Notes object."""
        self.project = project
        self.path = {}
        self.str = ""
        self.notes_lst = []
        self.token_lst = []
        self.notes_db = None
        self.notes_dict = {}
        self.chroma_docs = []
        self.chroma_metadata = []
        self.chroma_ids = []
        self.last_ingest = None

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
    
    def set_path(self):
        """Set the folder containing the notes."""
        if len(self.project.notes_path) > 0:
            # If the notes path is specified, use it
            self.path = self.project.notes_path
        else:
            self.path = os.path.join(
                get_root_dir(), "data", "notes", self.project.notes_folder
            )
        logger.info(f"Notes folder set to {self.path}")
        # Check if the folder exists
        if not os.path.exists(self.path):
            logger.error(f"Notes folder {self.path} does not exist")
            raise FileNotFoundError(f"Notes folder {self.path} does not exist")
    
    def load_notes_from_folder(self):
        """Load notes from the notes folder."""
        self.notes_dict = {}
        
        for filename in os.listdir(self.path):
        # If the filename has the text "ai_note" anywhere in its title, it's an AI-generated note and should be skipped
        # If the filename has the text "do_not_use" anywhere in its title, it's a note that should not be used and should be skipped
            if "ai_note" in filename or "do_not_use" in filename:
                logger.info(f"Skipping note {filename}")
                continue
            else:
                file_path = os.path.join(self.path, filename)
                note = Note(
                    filename=filename,
                    file_path=file_path,
                    project=self.project,
                )
                self.notes_dict[filename] = note
                logger.info(f"Loaded note {filename}")
        logger.info(f"Loaded {len(self.notes_dict)} notes from {self.path}")
        self.last_ingest = get_timestamp()
    
    def load_new_notes_from_folder(self):
        """Load notes from the notes folder if they've been modified since the last ingest."""
        for filename in os.listdir(self.path):
            # If the filename starts with ai_note, it's an AI-generated note and should be skipped
            if "ai_note" in filename or "do_not_use" in filename:
                logger.info(f"Skipping note {filename}")
                continue
            else:
                file_path = os.path.join(self.path, filename)
                if os.path.getmtime(file_path) > self.last_ingest:
                    note = Note(
                        filename=filename,
                        file_path=file_path,
                        project=self.project,
                    )
                    self.notes_dict[filename] = note
                    logger.info(f"Loaded note {filename}")
        logger.info(f"Project now has {len(self.notes_dict)} notes from {self.path}")
        self.last_ingest = get_timestamp()
    
    def create_excerpts_from_notes(self):
        """Create excerpts from the notes."""
        for filename, note in self.notes_dict.items():
            # If excerpts have not been created for the note, create them
            if len(note.excerpts) == 0:
                logger.info(f"Creating excerpts for note {filename}")
                note.create_excerpts()
    
    def create_chroma_docs(self):
        """Create a list of documents to be ingested into the Chroma database."""
        self.chroma_docs = []
        for filename, note in self.notes_dict.items():
            for excerpt in note.excerpts:
                doc = Document(
                    page_content=excerpt.text,
                    metadata={
                        "source": "note",
                        "filename": filename,
                        "filepath": note.file_path,
                        "full_text": note.text,
                        "excerpt_index": excerpt.index,
                    }
                )
                excerpt.in_db = True
                self.chroma_docs.append(doc)
    
    def create_notes_db(self):
        """Create a Chroma vector datbase of notes using OpenAI embeddings."""
        self.notes_db = create_db(
            name="notes_db",
            persist_directory=os.path.join(self.project.project_path, "vectordb", "notes"),
            chroma_docs=self.chroma_docs,
        )
    
    def load_notes_db(self):
        self.notes_db = load_db(
            name="notes_db",
            persist_directory=os.path.join(self.project.project_path, "vectordb", "notes"),
        )
    
    def add_to_notes_db(self):
        new_chroma_docs = []
        for filename, note in self.notes_dict.items():
            for excerpt in note.excerpts:
                if excerpt.in_db == False:
                    doc = Document(
                        page_content=excerpt.text,
                        metadata={
                            "source": "note",
                            "filename": filename,
                            "filepath": note.file_path,
                            "full_text": note.text,
                            "excerpt_index": excerpt.index,
                        }
                    )
                    excerpt.in_db = True
                    self.chroma_docs.append(doc)
                    new_chroma_docs.append(doc)
        if len(new_chroma_docs) == 0:
            logger.info("No new excerpts to add to the notes database.")
        else:
            self.notes_db = add_to_db(
            name="notes_db",
            persist_directory=os.path.join(self.project.project_path, "vectordb", "notes"),
            chroma_docs=new_chroma_docs,
        )
            
        logger.info(f"Added {len(new_chroma_docs)} new excerpts to the sources database.")

        
    
    def add_to_notes_db_old(self):
        excerpts_text_lst = []
        for text, excerpt in self.excerpts_dict.items():
            if excerpt.in_db == False:
                excerpts_text_lst.append(text)
                
        if len(excerpts_text_lst) == 0:
            logger.info("No new excerpts to add to the sources excerpts database.")
            return
        
        self.notes_db.add_texts(excerpts_text_lst)
        
        logger.info(f"Added {len(excerpts_text_lst)} excerpts to the notes excerpts database")
        for text, excerpt in self.excerpts_dict.items():
            excerpt.in_db = True
    
