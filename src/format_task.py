"""FormatTask class for formatting text at final stage of writing."""

import logging
import os
import re
import copy

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

from src.citation import Citation

# Set up logger
logger = logging.getLogger('writing_assistant')

class FormatTask:
    """Class for formatting text at final stage of writing."""
    
    def __init__(
        self, 
        task_name: str, 
        project,
    ):
        """Initialize a FormatTask instance."""
        self.task_name = task_name
        self.project = project
        self.draft = None
        # Draft in markdown format
        self.draft_md = None
        
        # Citation variables
        self.draft_citations = None
        self.draft_citations_indexed = None
        self.draft_citations_inserted = None
        # List of each citation in the draft
        self.citations = []
        
        # Bluebook variables
        self.unique_citekeys = []
        # List of citation keys in the draft but not in the Bluebook dictionary
        self.not_in_bb_dict = []
        # List of citation keys in the draft and in the Bluebook dictionary but no bluebook citation
        self.no_bb_in_bb_dict = []
        # List of citation keys in the draft and in the Bluebook dictionary but no shortcite
        self.no_sc_in_bb_dict = []
        
        self.draft_bluebooked = None
        
    def convert_to_markdown(self):
        """Convert the draft to Markdown format.
        This method parses the draft, turning every instance of "/foonote{...}" into a Markdown footnote, starting with [^1] and incrementing by 1.
        """

        # Load draft from file
        draft_path = os.path.join(self.project.project_path, "draft.md")
        self.draft = extract_from_md(draft_path, "draft.md")
        logger.info(f"Loaded draft from file.")
        # Go through draft and replace "/footnote{...}" with Markdown footnotes
        # Regular expression to find all \footnote{...} patterns
        footnote_pattern = re.compile(r"\\footnote\{((?:[^{}]|\{[^{}]*\})*)\}")
        
        # Initialize footnote counter
        fn_counter = 1
        footnotes = []
        
        # Function to replace each footnote with a Markdown footnote
        def replace_footnote(match):
            nonlocal fn_counter
            footnote_content = match.group(1)
            
            # Add the footnote to the list of footnotes
            footnotes.append(f"[^{fn_counter}]: {footnote_content}")
            replacement = f"[^{fn_counter}]"
            fn_counter += 1
            return replacement
        
        # Replace all footnotes in the draft
        draft_with_footnotes = footnote_pattern.sub(replace_footnote, self.draft)
        
        
        # Append footnotes at the end of the draft
        draft_with_footnotes += "\n\n" + "\n".join(footnotes)
        
        self.draft_md = draft_with_footnotes
        
        def check_remaining_footnotes(self):
            """Check for any remaining \footnote patterns in the draft."""
            # Find all remaining footnote patterns
            footnote_pattern = re.compile(r'\\footnote\{([^}]*)\}')
            matches = list(footnote_pattern.finditer(self.draft_md))
            
            if matches:
                logger.warning(f"Found {len(matches)} remaining \\footnote patterns")
                problem_footnotes = ""
                
                for match in matches:
                    # Get 50 chars before and 50 after for context
                    start = max(0, match.start() - 50)
                    end = min(len(self.draft_md), match.end() + 50)
                    
                    context = self.draft_md[start:end]
                    problem_footnotes += f"--- Problem Footnote ---\n"
                    problem_footnotes += f"Location: {match.start()}\n"
                    problem_footnotes += f"Content: {match.group(0)}\n"
                    problem_footnotes += f"Context: ...{context}...\n\n"
                
                export_to_md(
                    directory=self.project.output_directory,
                    filename=f"{self.task_name}_problem_footnotes",
                    text=problem_footnotes,
                )
        
        check_remaining_footnotes(self)
        
        # Export to markdown file
        export_to_md(
            directory=self.project.output_directory,
            filename=f"{self.task_name}_draft_markdown",
            text=draft_with_footnotes,
        )
    
    def extract_citations(self):
        """Go through the draft and extract all citations.
        self.citations is a list of Citation instances.
        For each \cite{...} in the draft, create a Citation instance.
            original_text: the original text of the citation (whatever is inside \cite{...})
            footnote_num: the footnote number of the citation (go backwards from \cite{...} to find the footnote number, always in format [^{number}])
            location_cite: the location of the start of the citation in the draft
            alone_in_footnote: whether the citation is the only citation in the footnote (check for \cite{...} before and after until you hit a "]" or "[")
        Replace the text within \cite{...} with the index of the citation in self.citations.
        """
        
        # Function to extract the original text of the citation
        def extract_original_text(location):
            # Go forwards from location to find the original text of the citation
            original_text = ""
            i = location + 6
            while i < len(self.draft_citations):
                if self.draft_citations[i] == "}":
                    break
                original_text += self.draft_citations[i]
                i += 1
                # Log a warning if the length of the original text is greater than 300 and break
                if len(original_text) > 300:
                    logger.warning(f"Original text of citation is greater than 300 characters.")
                    break
            return original_text
        
        # Function to extract the footnote number from the draft
        def extract_footnote_num(location):
            # Go backwards from location to find the footnote number
            fn_num = ""
            i = location - 1
            while i >= 0:
                if self.draft_citations[i] == "^":
                    # Start of footnote number, go forwards until you hit a "]"
                    i += 1
                    while i < location:
                        if self.draft_citations[i] == "]":
                            break
                        fn_num += self.draft_citations[i]
                        i += 1
                    break
                i -= 1
            return fn_num
        
        # Function to determine whether the citation is alone in the footnote
        def is_alone_in_footnote(location):
            # Go forwards from location to find the next \cite{...}
            i = location + 6
            while i+6 < len(self.draft_citations):
                if self.draft_citations[i:i+6] == "\cite{":
                    return False
                if self.draft_citations[i+6] == "^":
                    break
                i += 1
            # Go backwards from location to find the previous \cite{...}
            i = location - 1
            while i >= 0:
                if self.draft_citations[i-6:i] == "\cite{":
                    return False
                if self.draft_citations[i] == "^":
                    break
                i -= 1
            return True
        
        self.draft_citations = copy.deepcopy(self.draft_md)
        # Regular expression to find all \cite{...} patterns
        cite_pattern = re.compile(r"\\cite\{((?:[^{}]|\{[^{}]*\})*)\}")
        # Find all \cite{...} patterns in the draft
        # Convert iterator to list immediately
        cites = list(cite_pattern.finditer(self.draft_md))
        
        # Initialize list of citations
        self.citations = []
        
        # Create new string for citations with indices
        self.draft_citations_indexed = ""
        last_end = 0

        # Process each citation and build string
        for idx, cite in enumerate(cites):
            # Add text before current citation
            self.draft_citations_indexed += self.draft_citations[last_end:cite.start()]
            # Add citation index
            self.draft_citations_indexed += "\cite{" + str(idx) + "}"
            # Update last position
            last_end = cite.end()
            
            # Create Citation instance
            citation = Citation(
                original_text=extract_original_text(cite.start()),
                footnote_num=extract_footnote_num(cite.start()),
                location_cite=cite.start(),
                alone_in_footnote=is_alone_in_footnote(cite.start()),
            )
            self.citations.append(citation)
        
        # Add remaining text after last citation
        if cites:
            self.draft_citations_indexed += self.draft_citations[cites[-1].end():]
        else:
            self.draft_citations_indexed = self.draft_citations
    
    def extract_citation_info(self):
        """Extract information from the citations."""
        for citation in self.citations:
            citation.extract_citekey()
            citation.extract_page_number()
            citation.extract_parenthetical()
            # Check if any prior citations have the same citekey
            # So only check citations before the current citation
            for prior_citation in self.citations[:self.citations.index(citation)]:
                if prior_citation.citekey == citation.citekey:
                    citation.first_occurrence = False
                    break
    
    def abbreviate_journal(self, journal_name):
        """Abbreviate a journal name according to Bluebook rules."""
        
        # Table T6
        # Substitute "Journals" or "Journal" with "J."
        journal_name = re.sub(r"Journals", "J.", journal_name)
        journal_name = re.sub(r"Journal", "J.", journal_name)
        # Substitute "Review" with "Rev."
        journal_name = re.sub(r"Review", "Rev.", journal_name)
        # Substitute "Laws" or "Law" with "L."
        journal_name = re.sub(r"Laws ", "L. ", journal_name)
        journal_name = re.sub(r"Law ", "L. ", journal_name)
        # If the last word is "Law", replace with "L."
        journal_name = re.sub(r" Law$", " L.", journal_name)
        # Substitute "and" with "&"
        journal_name = re.sub(r" and ", " & ", journal_name)
        # Subtitute "American" with "Am."
        journal_name = re.sub(r" American", " Am.", journal_name)
        # Substitute "Behavior" or "Behavioral" with "Behav."
        journal_name = re.sub(r"Behavior", "Behav.", journal_name)
        journal_name = re.sub(r"Behavioral", "Behav.", journal_name)
        # Substitute "Catholic" with "Cath."
        journal_name = re.sub(r"Catholic", "Cath.", journal_name)
        # Substitute "Business" with "Bus."
        journal_name = re.sub(r"Business", "Bus.", journal_name)
        # Substitute "Civil Liberties" or "Civil Liberty" with "Civ. Lib."
        journal_name = re.sub(r"Civil Liberties", "Civ. Lib.", journal_name)
        journal_name = re.sub(r"Civil Liberty", "Civ. Lib.", journal_name)
        # Substitute "Civil" with "Civ."
        journal_name = re.sub(r"Civil", "Civ.", journal_name)
        # Substitute "Commentary" with "Comment."
        journal_name = re.sub(r"Commentary", "Comment.", journal_name)
        # Substitute "Comparative" with "Compar."
        journal_name = re.sub(r"Comparative", "Compar.", journal_name)
        # Substitute "Computer" with "Comput."
        journal_name = re.sub(r"Computer", "Comput.", journal_name)
        # Substitute "Conference" with "Conf."
        journal_name = re.sub(r"Conference", "Conf.", journal_name)
        # Substitute "Constitutional" or "Constitution" with "Const."
        journal_name = re.sub(r"Constitutional", "Const.", journal_name)
        journal_name = re.sub(r"Constitution", "Const.", journal_name)
        # Substitute "Contemporary" with "Contemp."
        journal_name = re.sub(r"Contemporary", "Contemp.", journal_name)
        # Substitute "Criminal" with "Crim."
        journal_name = re.sub(r"Criminal", "Crim.", journal_name)
        # Substitute " East" or " Eastern" with " E."
        journal_name = re.sub(r" East", " E.", journal_name)
        journal_name = re.sub(r" Eastern", " E.", journal_name)
        # Substitute "Economic", "Economical", "Economics", "Economy" with "Econ."
        journal_name = re.sub(r"Economical", "Econ.", journal_name)
        journal_name = re.sub(r"Economics", "Econ.", journal_name)
        journal_name = re.sub(r"Economy", "Econ.", journal_name)
        journal_name = re.sub(r"Economic", "Econ.", journal_name)
        # Substitute "Education" with "Educ."
        journal_name = re.sub(r"Education", "Educ.", journal_name)
        # Substitute "Environmental" with "Env't"
        journal_name = re.sub(r"Environmental", "Env't", journal_name)
        # Substitute "Equality" with "Equal."
        journal_name = re.sub(r"Equality", "Equal.", journal_name)
        # Substitute "Forum" with "F."
        journal_name = re.sub(r"Forum", "F.", journal_name)
        # Substitute "Government" with "Gov't"
        journal_name = re.sub(r"Government", "Gov't", journal_name)
        # Substitute "Humanity" with "Human."
        journal_name = re.sub(r"Humanity", "Human.", journal_name)
        # Substitute "Human " with "Hum. "
        journal_name = re.sub(r" Human ", " Hum. ", journal_name)
        # Substitute "Immigration" with "Immigr."
        journal_name = re.sub(r"Immigration", "Immigr.", journal_name)
        # Substitute "Inequality" with "Ineq."
        journal_name = re.sub(r"Inequality", "Ineq.", journal_name)
        # Substitute "Information" with "Info."
        journal_name = re.sub(r"Information", "Info.", journal_name)
        # Substitute "Institute" with "Inst."
        journal_name = re.sub(r"Institute", "Inst.", journal_name)
        # Substitute "Intelligence" with "Intell."
        journal_name = re.sub(r"Intelligence", "Intell.", journal_name)
        # Substitute "International" with "Int'l"
        journal_name = re.sub(r"International", "Int'l", journal_name)
        # Substitute "Magaizne" with "Mag."
        journal_name = re.sub(r"Magazine", "Mag.", journal_name)
        # Substitute "Management" with "Mgmt."
        journal_name = re.sub(r"Management", "Mgmt.", journal_name)
        # Substitute "Medical" with "Med."
        journal_name = re.sub(r"Medical", "Med.", journal_name)
        # Substitute "Modern" with "Mod."
        journal_name = re.sub(r"Modern", "Mod.", journal_name)
        # Substitute "National" with "Nat'l"
        journal_name = re.sub(r"National", "Nat'l", journal_name)
        # Substitute "Natural" with "Nat."
        journal_name = re.sub(r"Natural", "Nat.", journal_name)
        # Substitute "Organization" with "Org."
        journal_name = re.sub(r"Organization", "Org.", journal_name)
        # Substitute "Political" with "Pol."
        journal_name = re.sub(r"Political", "Pol.", journal_name)
        # Substitute "Politics" with "Pol."
        journal_name = re.sub(r"Politics", "Pol.", journal_name)
        # Substitute "Privacy" with "Priv."
        journal_name = re.sub(r"Privacy", "Priv.", journal_name)
        # Substitute "Problems" with "Probs."
        journal_name = re.sub(r"Problems", "Probs.", journal_name)
        # Substitute "Proceedings" with "Proc."
        journal_name = re.sub(r"Proceedings", "Proc.", journal_name)
        # Substitute "Psychology" with "Psych."
        journal_name = re.sub(r"Psychology", "Psych.", journal_name)
        # Substitute "Public" with "Pub."
        journal_name = re.sub(r"Public", "Pub.", journal_name)
        # Substitute "Publication" with "Publ'n"
        journal_name = re.sub(r"Publication", "Publ'n", journal_name)
        # Substitute "Publishing" with "Publ'g"
        journal_name = re.sub(r"Publishing", "Publ'g", journal_name)
        # Substitute "Quarterly" with "Q."
        journal_name = re.sub(r"Quarterly", "Q.", journal_name)
        # Substitute "Research" with "Rsch."
        journal_name = re.sub(r"Research", "Rsch.", journal_name)
        # Substitute "Review" with "Rev."
        journal_name = re.sub(r"Review", "Rev.", journal_name)
        # Substitute "Rights" with "Rts."
        journal_name = re.sub(r" Rights", " Rts.", journal_name)
        # Substitute "School" with "Sch."
        journal_name = re.sub(r"School", "Sch.", journal_name)
        # Substitute "Science" with "Sci."
        journal_name = re.sub(r"Science", "Sci.", journal_name)
        # Substitute "Social" with "Soc."
        journal_name = re.sub(r"Social ", "Soc. ", journal_name)
        # Substitute "Society" with "Soc'y"
        journal_name = re.sub(r"Society", "Soc'y", journal_name)
        # Substitute "Sociology" with "Socio."
        journal_name = re.sub(r"Sociology", "Socio.", journal_name)
        # Substitute "South " and "Southern" with "S."
        journal_name = re.sub(r" South ", " S. ", journal_name)
        journal_name = re.sub(r" Southern", " S.", journal_name)
        # Substitute "Southeastern" or "Southeast" with "Se."
        journal_name = re.sub(r"Southeastern", "Se.", journal_name)
        journal_name = re.sub(r"Southeast", "Se.", journal_name)
        # Substitute "Southwestern" or "Southwest" with "Sw."
        journal_name = re.sub(r"Southwestern", "Sw.", journal_name)
        journal_name = re.sub(r"Southwest", "Sw.", journal_name)
        # Substitute "Statistics" or "Statistical" with "Stat."
        journal_name = re.sub(r"Statistics", "Stat.", journal_name)
        journal_name = re.sub(r"Statistical", "Stat.", journal_name)
        # Substitute "Studies" with "Stud."
        journal_name = re.sub(r"Studies", "Stud.", journal_name)
        # Substitute "Supreme Court" with "Sup. Ct."
        journal_name = re.sub(r"Supreme Court", "Sup. Ct.", journal_name)
        # Substitute "Survey" with "Surv."
        journal_name = re.sub(r"Survey", "Surv.", journal_name)
        # Substitute "Symposium" with "Symp."
        journal_name = re.sub(r"Symposium", "Symp.", journal_name)
        # Substitute "Technology", "Technical", "Technological" with "Tech."
        journal_name = re.sub(r"Technology", "Tech.", journal_name)
        journal_name = re.sub(r"Technical", "Tech.", journal_name)
        journal_name = re.sub(r"Technological", "Tech.", journal_name)
        # Substitute "Tribune" with "Trib."
        journal_name = re.sub(r"Tribune", "Trib.", journal_name)
        # Substitute "United States" with "U.S."
        journal_name = re.sub(r"United States", "U.S.", journal_name)
        # Substitute "University" with "Univ."
        journal_name = re.sub(r"University", "U.", journal_name)
        # Substitute "Urban" with "Urb."
        journal_name = re.sub(r"Urban", "Urb.", journal_name)
        # Substitute "Week" with "Wk."
        journal_name = re.sub(r"Week", "Wk.", journal_name)
        # Substitute "Weekly" with "Wkly."
        journal_name = re.sub(r"Weekly", "Wkly.", journal_name)
        # Substitute "Western" with "W."
        journal_name = re.sub(r"Western", "W.", journal_name)
        
        # Table T10
        # States
        journal_name = re.sub(r"Alabama", "Ala.", journal_name)
        journal_name = re.sub(r"Alaska", "Alaska", journal_name)
        journal_name = re.sub(r"Arizona", "Ariz.", journal_name)
        journal_name = re.sub(r"Arkansas", "Ark.", journal_name)
        journal_name = re.sub(r"California", "Cal.", journal_name)
        journal_name = re.sub(r"Colorado", "Colo.", journal_name)
        journal_name = re.sub(r"Connecticut", "Conn.", journal_name)
        journal_name = re.sub(r"Delaware", "Del.", journal_name)
        journal_name = re.sub(r"Florida", "Fla.", journal_name)
        journal_name = re.sub(r"Georgia", "Ga.", journal_name)
        journal_name = re.sub(r"Hawaii", "Haw.", journal_name)
        journal_name = re.sub(r"Idaho", "Idaho", journal_name)
        journal_name = re.sub(r"Illinois", "Ill.", journal_name)
        journal_name = re.sub(r"Indiana", "Ind.", journal_name)
        journal_name = re.sub(r"Iowa", "Iowa", journal_name)
        journal_name = re.sub(r"Kansas", "Kan.", journal_name)
        journal_name = re.sub(r"Kentucky", "Ky.", journal_name)
        journal_name = re.sub(r"Louisiana", "La.", journal_name)
        journal_name = re.sub(r"Maine", "Me.", journal_name)
        journal_name = re.sub(r"Maryland", "Md.", journal_name)
        journal_name = re.sub(r"Massachusetts", "Mass.", journal_name)
        journal_name = re.sub(r"Michigan", "Mich.", journal_name)
        journal_name = re.sub(r"Minnesota", "Minn.", journal_name)
        journal_name = re.sub(r"Mississippi", "Miss.", journal_name)
        journal_name = re.sub(r"Missouri", "Mo.", journal_name)
        journal_name = re.sub(r"Montana", "Mont.", journal_name)
        journal_name = re.sub(r"Nebraska", "Neb.", journal_name)
        journal_name = re.sub(r"Nevada", "Nev.", journal_name)
        journal_name = re.sub(r"New Hampshire", "N.H.", journal_name)
        journal_name = re.sub(r"New Jersey", "N.J.", journal_name)
        journal_name = re.sub(r"New Mexico", "N.M.", journal_name)
        journal_name = re.sub(r"New York", "N.Y.", journal_name)
        journal_name = re.sub(r"North Carolina", "N.C.", journal_name)
        journal_name = re.sub(r"North Dakota", "N.D.", journal_name)
        journal_name = re.sub(r"Ohio", "Ohio", journal_name)
        journal_name = re.sub(r"Oklahoma", "Okla.", journal_name)
        journal_name = re.sub(r"Oregon", "Or.", journal_name)
        journal_name = re.sub(r"Pennsylvania", "Pa.", journal_name)
        journal_name = re.sub(r"Rhode Island", "R.I.", journal_name)
        journal_name = re.sub(r"South Carolina", "S.C.", journal_name)
        journal_name = re.sub(r"South Dakota", "S.D.", journal_name)
        journal_name = re.sub(r"Tennessee", "Tenn.", journal_name)
        journal_name = re.sub(r"Texas", "Tex.", journal_name)
        journal_name = re.sub(r"Utah", "Utah", journal_name)
        journal_name = re.sub(r"Vermont", "Vt.", journal_name)
        journal_name = re.sub(r"Virginia", "Va.", journal_name)
        journal_name = re.sub(r"Washington", "Wash.", journal_name)
        journal_name = re.sub(r"West Virginia", "W. Va.", journal_name)
        journal_name = re.sub(r"Wisconsin", "Wis.", journal_name)
        journal_name = re.sub(r"Wyoming", "Wyo.", journal_name)
        
        # Cities
        journal_name = re.sub(r"Balitmore", "Balt.", journal_name)
        journal_name = re.sub(r"Boston", "Bos.", journal_name)
        journal_name = re.sub(r"Chicago", "Chi.", journal_name)
        journal_name = re.sub(r"Dallas", "Dal.", journal_name)
        journal_name = re.sub(r"District of Columbia", "D.C.", journal_name)
        journal_name = re.sub(r"Houston", "Hous.", journal_name)
        journal_name = re.sub(r"Los Angeles", "L.A.", journal_name)
        journal_name = re.sub(r"Miami", "Mia.", journal_name)
        journal_name = re.sub(r"New York", "N.Y.", journal_name)
        journal_name = re.sub(r"Philadelphia", "Phila.", journal_name)
        journal_name = re.sub(r"Phoenix", "Phx.", journal_name)
        journal_name = re.sub(r"San Francisco", "S.F.", journal_name)
        
        # Table T13
        # Institutional names
        journal_name = re.sub(r"Albany", "Alb.", journal_name)
        journal_name = re.sub(r"American Bar Association", "A.B.A.", journal_name)
        journal_name = re.sub(r"American Law Institute", "A.L.I.", journal_name)
        journal_name = re.sub(r"Journal of the American Medical Association", "JAMA", journal_name)
        journal_name = re.sub(r"American Medical Association", "AMA", journal_name)
        journal_name = re.sub(r"American University", "Am. U.", journal_name)
        journal_name = re.sub(r"Boston College", "B.C.", journal_name)
        journal_name = re.sub(r"Boston University", "B.U.", journal_name)
        journal_name = re.sub(r"Bringham Young University", "BYU", journal_name)
        journal_name = re.sub(r"Brooklyn", "Brook.", journal_name)
        journal_name = re.sub(r"Buffalo", "Buff.", journal_name)
        journal_name = re.sub(r"California L. Rev.", "Cal. L. Rev.", journal_name)
        journal_name = re.sub(r"Capital", "Cap.", journal_name)
        journal_name = re.sub(r"Chapman", "Chap.", journal_name)
        journal_name = re.sub(r"Cincinnati", "Cin.", journal_name)
        journal_name = re.sub(r"City University of New York", "CUNY", journal_name)
        journal_name = re.sub(r"Columbia", "Col.", journal_name)
        journal_name = re.sub(r"Cornell", "Corn.", journal_name)
        journal_name = re.sub(r"Denver", "Den.", journal_name)
        journal_name = re.sub(r"Detroit", "Det.", journal_name)
        journal_name = re.sub(r"Dickinson", "Dick.", journal_name)
        
        journal_name = re.sub(r"Harvard", "Harv.", journal_name)
        journal_name = re.sub(r"Loyola", "Loy.", journal_name)
        journal_name = re.sub(r"State", "St.", journal_name)
        # Quit here because this is too annoying to keep doing

        # Omit the words "a" "at" "in" "of" and "the"
        journal_name = re.sub(r"\b(a|at|in|of|the)\b", "", journal_name)
        
        
        return journal_name
    
    def reverse_author(self, author_name):
        """Reverse the order of an author name (often listed as "Last, First" and need it to be "First Last")."""
        # Split the author name by ", "
        author_parts = author_name.split(", ")
        # Reverse the order of the parts
        author_parts.reverse()
        # Join the parts back together with a space
        return " ".join(author_parts)
    
    def extract_author_last(self, author):
        """Split an auther name into first and last names, returning the last name."""
        # Extract author last name safely
        # If there is a comma in the author name, split by comma and get the first word
        if author and "," in author:
            author_last = author.split(",")[0]
        # If there is a space in the author name, split by space and get the last word
        elif author and " " in author:
            author_last = author.split()[-1]
        else:
            author_last = author
        return author_last

    
    def get_missing_bluebook_info(self):
        """Get a list of citations in the draft that our Bluebook dictionary does not have citation information for."""
        # Get a list of unique citekeys from the citations
        self.unique_citekeys = []
        for citation in self.citations:
            if citation.first_occurrence:
                self.unique_citekeys.append(citation.citekey)
            
        # Go through self.unique_citekeys and 
        # find those not in the Bluebook dictionary, adding them to self.not_in_bb_dict
        # find those in the Bluebook dictionary but with no bluebook citation, adding them to self.no_bb_in_bb_dict
        # find those in the Bluebook dictionary but with no shortcite, adding them to self.no_sc_in_bb_dict
        for citekey in self.unique_citekeys:
            if citekey not in self.project.bluebook_dict.dict:
                self.not_in_bb_dict.append(citekey)
            else:
                if self.project.bluebook_dict.dict[citekey].bluebook == "":
                    self.no_bb_in_bb_dict.append(citekey)
                if self.project.bluebook_dict.dict[citekey].shortcite == "":
                    self.no_sc_in_bb_dict.append(citekey)
        
        # Log the results
        logger.info(f"Missing Bluebook Dictionary entries for {len(self.not_in_bb_dict)} citekeys.")
        logger.info(f"No Bluebook citation for {len(self.no_bb_in_bb_dict)} citekeys.")
        logger.info(f"No shortcite for {len(self.no_sc_in_bb_dict)} citekeys.")
        
        # Export the results to a markdown file
        missing_info_md = f"# Missing Bluebook Information\n\n"
        missing_info_md += f"## Not in Bluebook Dictionary\n\n"
        missing_info_md += "\n".join([f"- {citekey}" for citekey in self.not_in_bb_dict])
        missing_info_md += "\n\n## No Bluebook Citation\n\n"
        missing_info_md += "\n".join([f"- {citekey}" for citekey in self.no_bb_in_bb_dict])
        missing_info_md += "\n\n## No Shortcite\n\n"
        missing_info_md += "\n".join([f"- {citekey}" for citekey in self.no_sc_in_bb_dict])
        
        export_to_md(
            directory=self.project.output_directory,
            filename=f"{self.task_name}_missing_bluebook_info",
            text=missing_info_md,
        )
        
        # Export a more detailed analysis of the results to a markdown file
        # For each citation missing Bluebook information, grab information about the citation from self.project.sources.sources_dict[citekey].fields
        # Use that information to create a draft of the bluebook citation information
        # Export all of this to a markdown file.
        
        missing_info_md = "# Detailed Missing Bluebook Information\n\n"
        for citekey in self.not_in_bb_dict:
            missing_info_md += f"## {citekey}\n\n"
            missing_info_md += f"### Source Information\n\n"
            try:
                source = self.project.sources.sources_dict[citekey]
            except KeyError:
                missing_info_md += f"Could not find source information for {citekey}.\n"
                continue
            
            for field, value in source.fields.items():
                missing_info_md += f"**{field}**: {value}\n"
            
            # Extract information for article
            entry_type = source.fields.get("ENTRYTYPE", "")
            if entry_type == "article":
                author = source.fields.get("author", "")
                title = source.fields.get("title", "")
                journal = source.fields.get("journaltitle", "")
                date = source.fields.get("date", "")
                if len(date) < 2:
                    date = source.fields.get("year", "")
                volume = source.fields.get("volume", "")
                pages = source.fields.get("pages", "")
                url = source.fields.get("url", "")
                
                if len(url) > 5:
                    url = f", {url}"
                
                author_last = self.extract_author_last(author)
                author = self.reverse_author(author)

                # Remove any "{" or "}" from the title
                title = title.replace("{", "").replace("}", "")
                journal = self.abbreviate_journal(journal)

                missing_info_md += f"\n\n### Draft Bluebook Citation\n\n"
                missing_info_md += "all_cites.add_citation(\n"
                missing_info_md += f"    citekey=\"{citekey}\",\n"
                missing_info_md += f"    bluebook=\"{author}, *{title}*, {volume} [{journal}]{{.smallcaps}} {pages}, [##] ({date}){url}\",\n"
                missing_info_md += f"    shortcite=\"{author_last}\"\n"
                missing_info_md += ")\n\n"
                missing_info_md += "\n\n"
            
            # Extract information for book
            elif entry_type == "book" or entry_type == "inbook" or entry_type == "incollection" or entry_type == "report":
                author = source.fields.get("author", "")
                title = source.fields.get("title", "")
                publisher = source.fields.get("publisher", "")
                date = source.fields.get("date", "")
                if len(date) < 2:
                    date = source.fields.get("year", "")
                
                author_last = self.extract_author_last(author)
                author = self.reverse_author(author)

                # Remove any "{" or "}" from the title
                title = title.replace("{", "").replace("}", "")

                missing_info_md += f"\n\n### Draft Bluebook Citation\n\n"
                missing_info_md += "all_cites.add_citation(\n"
                missing_info_md += f"    citekey=\"{citekey}\",\n"
                missing_info_md += f"    bluebook=\"[{author}, {title}]{{.smallcaps}} [##] ({date})\",\n"
                missing_info_md += f"    shortcite=\"[{author_last}]{{.smallcaps}}\"\n"
                missing_info_md += ")\n\n"
            
            # Extract information for webpage and anything else
            else:
                author = source.fields.get("author", "")
                title = source.fields.get("title", "")
                site = source.fields.get("site", "")
                date = source.fields.get("date", "")
                if len(date) < 2:
                    date = source.fields.get("year", "")
                url = source.fields.get("url", "")
                organization = source.fields.get("organization", "")
                
                author_last = self.extract_author_last(author)
                author = self.reverse_author(author)

                # Remove any "{" or "}" from the title
                title = title.replace("{", "").replace("}", "")

                missing_info_md += f"\n\n### Draft Bluebook Citation\n\n"
                missing_info_md += "all_cites.add_citation(\n"
                missing_info_md += f"    citekey=\"{citekey}\",\n"
                line = f"    bluebook=\"{author}, *{title}*, [{site} {organization}]{{.smallcaps}} ({date}), {url}\",\n"
                # Remove any double spaces from the line
                line = line.replace("  ", " ")
                line = line.replace("  ", " ")
                line = f"    {line}"
                missing_info_md += line
                missing_info_md += f"    shortcite=\"{author_last}\"\n"
                missing_info_md += ")\n\n"
        
        export_to_md(
            directory=self.project.output_directory,
            filename=f"{self.task_name}_missing_bluebook_info_detailed",
            text=missing_info_md,
        )   


    def format_citations(self):
        """Format the citations in the draft according to the Bluebook dictionary.
        This method goes through each citation in citations and formats it according to how it should appear in the draft.
        Depending on its position in the text, the citation may be formatted for a full cite, short cite, Id., or hereinafter.
        """
        for citation in self.citations:
            # Make sure there is a bluebook entry to work with
            if citation.citekey in self.project.bluebook_dict.dict:
                bb_entry = self.project.bluebook_dict.dict[citation.citekey]
            else:
                logger.warning(f"Could not find Bluebook information for {citation.citekey}.")
                citation.cite_in_draft = citation.original_text
                continue
            # If the citation is the first occurrence of the citekey, use the full bluebook citation
            if citation.first_occurrence:
                citation.cite_in_draft = bb_entry.bluebook
                # Replace '[##]' with the page number
                citation.cite_in_draft = citation.cite_in_draft.replace("[##]", citation.page_num)

            # Elif the citation with one index lower has the same citekey and is alone in the footnote, use "Id."
            elif self.citations[self.citations.index(citation) - 1].citekey == citation.citekey and citation.alone_in_footnote:
                citation.cite_in_draft = "*Id.*"
                if len(citation.page_num) > 0 and self.citations[self.citations.index(citation) - 1].page_num != citation.page_num:
                    citation.cite_in_draft += f" at {citation.page_num}"
            
            else:
                # Find the first occurrence of the citekey
                first_cite = ""
                for prior_citation in self.citations[:self.citations.index(citation)]:
                    if prior_citation.citekey == citation.citekey:
                        first_cite = prior_citation.footnote_num
                        break
                # If hereinafter is not empty, use hereinafter
                if len(bb_entry.hereinafter) > 0:
                    citation.cite_in_draft = f"{bb_entry.hereinafter}, *supra* note {first_cite}"
                    if len(citation.page_num) > 0:
                        citation.cite_in_draft += f", at {citation.page_num}"
                
                # Else, use shortcite
                else:
                    citation.cite_in_draft = f"{bb_entry.shortcite}, *supra* note {first_cite}"
                    if len(citation.page_num) > 0:
                        citation.cite_in_draft += f", at {citation.page_num}"
            
            # If self.parenthetical is not empty, add it to the citation
            if len(citation.parenthetical) > 0:
                citation.cite_in_draft += f" ({citation.parenthetical})"
            # Turn any double spaces into single spaces
            citation.cite_in_draft = citation.cite_in_draft.replace("  ", " ")
            # Strip leading and trailing whitespace
            citation.cite_in_draft = citation.cite_in_draft.strip()
    
    def insert_citations(self):
        """Insert the formatted citations back into the draft."""
        self.draft_citations_inserted = copy.deepcopy(self.draft_citations_indexed)
        for citation in self.citations:
            self.draft_citations_inserted = self.draft_citations_inserted.replace(f"\cite{{{self.citations.index(citation)}}}", citation.cite_in_draft)
        
        # Export to markdown file
        export_to_md(
            directory=self.project.output_directory,
            filename=f"{self.task_name}_draft_citations_inserted",
            text=self.draft_citations_inserted,
        )
    
    def format_through_bluebook(self):
        """Format the draft up to the Bluebook stage."""
        self.convert_to_markdown()
        self.extract_citations()
        self.extract_citation_info()
        self.get_missing_bluebook_info()
    
    def format_draft(self):
        """Format the draft by converting to Markdown, extracting citations, extracting citation information, getting missing Bluebook information, formatting citations, and inserting citations."""
        self.convert_to_markdown()
        self.extract_citations()
        self.extract_citation_info()
        self.get_missing_bluebook_info()
        self.format_citations()
        self.insert_citations()