"""
Utility functions for loading, saving, and exporting.

Functions
get_root_dir() -> str
    Returns the root directory for the package.
    Returns:
        A string representing the root directory of the package.
"""
import logging
import os
import PyPDF2
from bs4 import BeautifulSoup
from docx import Document

from src.utils_string import get_timestamp_as_string

# Set up logger
logger = logging.getLogger('writing_assistant')


class RootDirectoryNotFoundError(Exception):
    """Exception raised when the root directory is not found."""

    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.message = f"Root directory 'writing_assistant' not found within depth: {max_depth}"
        super().__init__(self.message)


def get_root_dir(max_depth=10):
    """Returns the root directory for the package."""
    # get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # get the root directory
    depth = 0
    while os.path.basename(current_dir) != 'writing_assistant' and depth < max_depth:
        current_dir = os.path.dirname(current_dir)
        depth += 1
    if os.path.basename(current_dir) == 'writing_assistant':
        return current_dir
    else:
        raise RootDirectoryNotFoundError(max_depth)

def extract_from_pdf(file_path, file):
    """Extract text from a PDF file."""
    try:
        with open(file_path, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
                text += "\n"
            logger.info(f"Extracted text from PDF {file}")
            return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF {file}: {e}")
        return ""

def extract_from_docx(file_path, file):
    """Extract text from a DOCX file."""
    try:
        doc = Document(file_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text
            text += "\n"
        logger.info(f"Extracted text from DOCX {file}")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from DOCX {file}: {e}")
        return ""

def extract_from_md(file_path, file="file"):
    """Extract text from a Markdown file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        logger.info(f"Extracted text from Markdown {file}")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from Markdown {file}: {e}")
        return ""

def extract_from_html(file_path, file="file"):
    """Extract text from an HTML file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Parse the HTML file with BeautifulSoup
            soup = BeautifulSoup(f, 'html.parser')
            
            # Extract the text from the HTML file
            text = soup.get_text(separator='\n', strip=True)
        logger.info(f"Extracted text from HTML {file}")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from HTML {file}: {e}")
        return ""
    
def extract_text_from_multiple_files(file_field):
        """Handle multiple files in the 'file' metadata field."""
        text = ""
        extracted = False
        file_paths = file_field.split(';')
        for file_path in file_paths:
            # Extract text from the file if it hasn't been extracted yet
            if extracted is False:
                file_path = file_path.strip()
                if file_path.endswith('.pdf'):
                    extracted_text = extract_from_pdf(file_path, file_path)
                elif file_path.endswith('.docx'):
                    extracted_text = extract_from_docx(file_path, file_path)
                elif file_path.endswith('.md'):
                    extracted_text = extract_from_md(file_path, file_path)
                elif file_path.endswith('.html'):
                    extracted_text = extract_from_html(file_path, file_path)
                else:
                    logger.warning(f"File {file_path} is not a .pdf, .docx, or .md file")
                    extracted_text = ""
                # If text has been extracted, stop extracting from other files in filepath
                if len(extracted_text) > 100:
                    extracted = True
                text += extracted_text
        return text

def find_page_number(excerpt_text, file_field):
    """Take a text excerpt and find the page number in the PDF or DOCX file."""
    try:
        logger.info(f"Searching for excerpt to identify page number in file(s): {file_field}")
        file_paths = file_field.split(';')
        for file_path in file_paths:
            file_path = file_path.strip()
            if file_path.endswith('.pdf'):
                place_in_excerpt = 0
                length_of_excerpt = len(excerpt_text)
                search_text = excerpt_text[place_in_excerpt:place_in_excerpt+250]
                tries = 0
                while tries < 10 and place_in_excerpt < length_of_excerpt:
                    with open(file_path, "rb") as pdf_file:
                        pdf_reader = PyPDF2.PdfReader(pdf_file)
                        for page_num in range(len(pdf_reader.pages)):
                            page = pdf_reader.pages[page_num]
                            page_text = page.extract_text()
                            if search_text in page_text:
                                logger.info(f"Found excerpt in PDF {file_path} on page {page_num + 1}")
                                return page_num + 1  # Page numbers are 1-based
                    place_in_excerpt += 150
                    tries += 1
                logger.info(f"Excerpt not found in PDF {file_path}, can't find page number.")
                return None  # Excerpt not found
            else:
                logger.warning(f"File {file_path} is not a .pdf file, can't search for page number.")
                return None
    except Exception as e:
        logger.warning(f"Failed to find page for excerpt in {file_path}: {e}")
        return None

def export_to_md(directory, filename, text):
    """Export text to a Markdown file."""
    timestamp = get_timestamp_as_string()
    file_path = os.path.join(directory, f"{filename}_{timestamp}.md")
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)
        logger.info(f"Exported text to Markdown {file_path}")
    except Exception as e:
        logger.error(f"Error exporting text to Markdown {file_path}: {e}")

def save_outputs(directory, filename: str, responses: list, meta_info: str = ""):
    """Save the outputs of the class LLM responses to a markdown file.
    
    Args:
        directory (str): The directory path where the markdown file will be saved.
        filename (str): The name of the markdown file.
        responses (list): List of response objects, each containing 'prompts' and 'output' attributes.

    Raises:
        Exception: If saving fails due to file permission or path errors.
    """
    # If responses is a single response, convert it to a list
    if not isinstance(responses, list):
        responses = [responses]
    timestamp = get_timestamp_as_string()
    filename = f"{filename}_{timestamp}.md"
    with open(os.path.join(directory, filename), 'w', encoding='utf-8') as f:
        f.write(f"# {filename} Prompts and Outputs\n\n")
        if len(meta_info) > 0:
            f.write("## Meta Info\n\n")
            f.write(meta_info)
            f.write("\n\n")
        for response in responses:
            f.write('## API Call Info\n\n')
            f.write(f"Model: {response.model}\nPrompt tokens: {response.prompt_tokens}\nCompletion tokens: {response.completion_tokens}\n\n")
            f.write("## Prompts\n\n")
            # Turn the prompts into a string
            prompts_str = ""
            for prompt in response.prompts:
                prompts_str += f"{prompt['role']}: {prompt['content']}\n\n"
            f.write(prompts_str)
            f.write("## Output\n\n")
            f.write(response.output)
            f.write("\n\n")