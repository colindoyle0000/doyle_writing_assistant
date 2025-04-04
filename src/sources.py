"""Class for loading sources for the project."""

import os
import logging
import time
import bibtexparser
from bibtexparser.bparser import BibTexParser
import PyPDF2
import logging
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


from src.utils_file import get_root_dir
from src.utils_llm import llm_call, create_db, load_db, add_to_db, preprocess_text, verify_docs_for_db
from src.utils_string import get_timestamp_as_string

from src.source import Source


# Set up logger
logger = logging.getLogger('writing_assistant')

class Sources:
    """Class for managing sources for the project."""
    
    def __init__(self, project):
        """Initialize a Sources object."""
        self.project = project
        self.sources_dict = {}
        self.excerpts_dict = {}
        self.sources_db = None

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
        
    def load_sources_from_bib_file(self, bib_file_path=None):
        """Parse a .bib file and return a list of Source instances."""
        if bib_file_path is None:
            bib_file_path = self.project.bib_file_path
        with open(bib_file_path, 'r', encoding='utf-8') as bib_file:
            parser = BibTexParser()
            parser.ignore_nonstandard_types = False
            bib_database = bibtexparser.load(bib_file, parser=parser)

        self.sources_dict = {}
        for entry in bib_database.entries:
            source = Source(
                entry=entry,
                project=self.project
                )
            # Add the source to the sources dictionary
            self.sources_dict[source.citekey] = source
        logger.info(f"Project now has {len(self.sources_dict)} sources from {bib_file_path}")
        
    def load_new_sources_from_bib_file(self, bib_file_path=None):
        """Load sources from a .bib file if the citekey is not already in the sources dictionary."""
        if bib_file_path is None:
            bib_file_path = self.project.bib_file_path
        with open(bib_file_path, 'r', encoding='utf-8') as bib_file:
            parser = BibTexParser()
            parser.ignore_nonstandard_types = False
            bib_database = bibtexparser.load(bib_file, parser=parser)
        
        for entry in bib_database.entries:
            if entry['ID'] not in self.sources_dict:
                source = Source(
                    entry=entry,
                    project=self.project
                    )
                # Add the source to the sources dictionary
                self.sources_dict[source.citekey] = source
        logger.info(f"Project now has {len(self.sources_dict)} sources from {bib_file_path}")
    
    def create_excerpts_from_sources(self):
        """Create excerpts from the sources."""
        for citekey, source in self.sources_dict.items():
            # If excerpts have not been created for this source, create them
            if len(source.excerpts) == 0 and len(source.text) > 0:
                source.create_excerpts()
    
    #def create_excerpts_dict(self):
    #    """Create a dictionary of excerpts. Key is text of excerpt, value is Excerpt instance."""
    #    for citekey, source in self.sources_dict.items():
    #        for excerpt in source.excerpts:
    #            self.excerpts_dict[excerpt.text] = excerpt
    #    logger.info(f"Created excerpts dictionary. Number of excerpts: {len(self.excerpts_dict)}")
    
    def create_chroma_docs(self):
        """Create a list of documents to be ingested into the Chroma database."""
        self.chroma_docs = []
        for citekey, source in self.sources_dict.items():
            for excerpt in source.excerpts:
                doc = Document(
                    page_content=excerpt.text,
                    metadata={
                        "source": "source",
                        "citekey": citekey,
                        "excerpt_index": excerpt.index,
                    }
                )
                excerpt.in_db = True
                self.chroma_docs.append(doc)
                logger.info(f"Added excerpt with index {excerpt.index} from source {citekey} to chroma_docs.")
    
    def preprocess_chroma_docs(self):
        """Preprocess the Chroma documents."""
        for doc in self.chroma_docs:
            doc.page_content = preprocess_text(doc.page_content)
        self.chroma_docs = verify_docs_for_db(self.chroma_docs)
    
    def create_sources_db(self):
        """Create a Chroma vector datbase of sources using OpenAI embeddings."""
        self.sources_db = create_db(
            name="sources_db",
            persist_directory=os.path.join(self.project.project_path, "vectordb", "sources"),
            chroma_docs=self.chroma_docs,
        )
    
    def load_sources_db(self):
        """Load the sources database."""
        self.sources_db = load_db(
            name="sources_db",
            persist_directory=os.path.join(self.project.project_path, "vectordb", "sources"),
        )

    def add_to_sources_db(self):
        """Add new excerpts to the sources database."""
        new_chroma_docs = []
        for citekey, source in self.sources_dict.items():
            for excerpt in source.excerpts:
                if excerpt.in_db == False:
                    doc = Document(
                        page_content=excerpt.text,
                        metadata={
                            "source": "source",
                            "citekey": citekey,
                            "full_text": source.text,
                            "excerpt_index": excerpt.index,
                        }
                    )
                    excerpt.in_db = True
                    self.chroma_docs.append(doc)
                    new_chroma_docs.append(doc)
        if len(new_chroma_docs) == 0:
            logger.info("No new excerpts to add to the sources database.")
        else:
            self.sources_db = add_to_db(
            name="sources_db",
            persist_directory=os.path.join(self.project.project_path, "vectordb", "sources"),
            chroma_docs=new_chroma_docs,
        )
        logger.info(f"Added {len(new_chroma_docs)} new excerpts to the sources database.")

    def create_and_add_sources_db_in_chunks(self):
        """To keep from going over token limits, create and add to sources db in chunks."""
        # Turn self.chroma_docs into a list of lists
        # Each sublist will be a chunk of 1000 documents
        chroma_docs_chunks = [self.chroma_docs[i:i + 1000] for i in range(0, len(self.chroma_docs), 1000)]
        
        logger.info(f"Split chroma_docs into {len(chroma_docs_chunks)} chunks.")

        for i, chunk in enumerate(chroma_docs_chunks):
            logger.info(f"Adding chunk {i} to the sources database.")
            logger.info(f"Chunk {i} document citekey: {chunk[0].metadata['citekey']} excerpt index: {chunk[0].metadata['excerpt_index']}")
            # If this is the first chunk, create the sources_db
            if i == 0:
                self.sources_db = create_db(
                    name=f"sources_db",
                    persist_directory=os.path.join(self.project.project_path, "vectordb", "sources"),
                    chroma_docs=chunk,
                )
                logger.info(f"Added chunk {i} to the sources database.")
            # If this is not the first chunk, add to the sources_db
            else:
                self.sources_db = add_to_db(
                    name=f"sources_db",
                    persist_directory=os.path.join(self.project.project_path, "vectordb", "sources"),
                    chroma_docs=chunk,
                )
                logger.info(f"Added chunk {i} to the sources database.")
            time.sleep(10)

    def create_chroma_chunks(self):
        """To keep from going over token limits, create and add to sources db in chunks."""
        # Turn self.chroma_docs into a list of lists
        # Each sublist will be a chunk of 1000 documents
        chroma_docs_chunks = [self.chroma_docs[i:i + 1000] for i in range(0, len(self.chroma_docs), 1000)]
        logger.info(f"Split chroma_docs into {len(chroma_docs_chunks)} chunks.")
        return chroma_docs_chunks
    
    def create_db_from_chunk(self, chunk):
        """Create a database from a chunk of documents."""
        self.sources_db = create_db(
            name=f"sources_db",
            persist_directory=os.path.join(self.project.project_path, "vectordb", "sources"),
            chroma_docs=chunk,
        )
    
    def add_to_db_from_chunk(self, chunk):
        """Add a chunk of documents to the database."""
        self.sources_db = add_to_db(
            name=f"sources_db",
            persist_directory=os.path.join(self.project.project_path, "vectordb", "sources"),
            chroma_docs=chunk,
        )