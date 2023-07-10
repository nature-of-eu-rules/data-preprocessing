#!/usr/bin/env python
# coding: utf-8

"""
Script to extract the regulatory section (as well as the sentences) of
EU legislative documents (PDF / HTML) located on the EURLEX website. 
Website: http://eur-lex.europa.eu/
"""

import fitz
from bs4 import BeautifulSoup
import pandas as pd
from lexnlp.nlp.en.segments.sentences import get_sentence_list
import string 
from thefuzz import fuzz
from thefuzz import process
import os
import re
import argparse
import sys
from os.path import exists

argParser = argparse.ArgumentParser(description='EU Legislation Regulatory Text and Sentence Extractor')
required = argParser.add_argument_group('required arguments')
required.add_argument("-in", "--input", required=True, help="Path to directory containing PDF and / or HTML EU legislative documents as downloaded using code from: https://github.com/nature-of-eu-rules/data-extraction")
required.add_argument("-out", "--output", required=True, help="Path to a CSV file which should store extracted sentences from the regulatory part of the input EU legislative documents found in the input folder e.g. 'path/to/sentences.csv'. ")
args = argParser.parse_args()

def is_valid_output_dir_or_file(arg):
    if arg is None:
        return False, "No valid CSV output file specified. Type 'python extract_sentences.py -h' for usage help."
    else:
        if os.path.isdir(str(arg)):
            return True, ''
        else:
            if os.path.isdir(os.path.dirname(str(arg))):
                if str(os.path.basename(str(arg))).lower().endswith('.csv'):
                    return True, ''
                else:
                    return False, 'Not a valid output file extension. CSV expected. Type "python extract_sentences.py -h" for usage help.'
            else:
                return False, 'The specified directory for your output CSV file is not valid or does not exist. First create it. Type "python extract_sentences.py -h" for usage help.'

def is_valid_input_dir(arg):
    if arg is None:
        return False, "No valid input directory specified. Type 'python extract_sentences.py -h' for usage help."
    else:
        count = 0
        if os.path.isdir(str(arg)):
            for path in os.listdir(str(arg)):
                if os.path.isfile(os.path.join(str(arg), path)) and str(os.path.basename(os.path.join(str(arg), path))).lower()[-4] in ['.html', '.pdf']:
                    count += 1
            if count > 0:
                return True, ''
            else:
                return False, 'No valid .pdf or .html files found in input directory.'
        else:
            return False, 'The specified input directory is not valid or does not exist. First create it. Type "python extract_sentences.py -h" for usage help.'

is_valid_input_directory, inerrmsg = is_valid_input_dir(args.input)
is_valid_output_directory_or_file, errmsg = is_valid_output_dir_or_file(args.output)

if not is_valid_output_directory_or_file:
     sys.exit(errmsg)

if not is_valid_input_directory:
     sys.exit(inerrmsg)

INPUT_DIR = str(args.input)
OUTPUT_FILE = str(args.output)

# Dictionary of phrases which denote the start and end point
# of relevant text in the documents

BEGIN_PHRASE_R1 = "HAS ADOPTED THIS REGULATION"
BEGIN_PHRASE_R2 = "HAVE ADOPTED THIS REGULATION"
BEGIN_PHRASE_D1 = "HAS DECIDED AS FOLLOWS"
BEGIN_PHRASE_D2 = "HAVE ADOPTED THIS DECISION"
BEGIN_PHRASE_D3 = "HAS ADOPTED THIS DECISION"
BEGIN_PHRASE_L = "HAS ADOPTED THIS DIRECTIVE"
BEGIN_PHRASES = [BEGIN_PHRASE_R1, BEGIN_PHRASE_R2, BEGIN_PHRASE_D1, BEGIN_PHRASE_D2, BEGIN_PHRASE_D3, BEGIN_PHRASE_L]

# Other constants
EXCLUDED_PHRASES = ["shall apply", "shall mean", "this regulation shall apply", "shall be binding in its entirety and directly applicable in the member states", "shall be binding in its entirety and directly applicable in all member states", "shall enter into force", "shall be based", "within the meaning", "shall be construed", "shall take effect"]
EXCLUDED_START_PHRASES = ['amendments to decision', 'amendments to implementing decision', 'in this case,', 'in such a case,', 'in such cases,', 'in all other cases,']
START_TOKENS = ['Article', 'Chapter', 'Section', 'ARTICLE', 'CHAPTER', 'SECTION', 'Paragraph', 'PARAGRAPH']
END_PHRASES = ["Done at Brussels", "Done at Luxembourg", "Done at Strasbourg", "Done at Frankfurt"]
DEONTICS = ['shall ', 'must ', 'shall not ', 'must not ']
DIGITS = '0123456789'

# BEGIN: function definitions

def get_index_of_next_upper_case_token(sent_tokens, start_index = 3):
    """Gets index of first word (after the given start_index) in list of words
      which starts with an uppercase character.

        Parameters
        ----------
        sent_tokens: list
            List of words.
        start_index: int
            the starting index from which the function starts searching

        Returns
        -------
        i: int
            the first index after start_index which has a word starting with an uppercase character

    """
    for i in range(start_index, len(sent_tokens)):
        if sent_tokens[i][0].isupper():
            return i
    return -1

def is_valid_sentence(sent_text):
    """Determines whether a sentence in a text can possibly be regulatory.

        Parameters
        ----------
        sent_text: str
            The sentence.
        
        Returns
        -------
            True if the sentence could possibly be regulatory, False otherwise.

    """
    global DIGITS
    global EXCLUDED_PHRASES
    global EXCLUDED_START_PHRASES

    is_valid = True
    
    # Rule 1: sentence should not start with any punctuation character (or numerical digit)
    if sent_text[0] in (string.punctuation + DIGITS):
        is_valid = False
        
    # Rule 2: check if 'EN Official Journal' or 'PAGE' occurs at start of sentence (this indicates an invalid sentence)
    if sent_text.lower().strip().startswith('en official journal') or sent_text.strip().startswith('PAGE'):
        is_valid = False
        
    # Rule 3: sentence must be at least 15 non-space characters long (otherwise highly unlikely to be a sentence)
    if len(sent_text.replace(' ','')) < 15:
        is_valid = False

    # Rule 4: sentence must not include these phrases (these phrases indicate non-regulatory sentences)
    for phrase in EXCLUDED_PHRASES:
        if (phrase in sent_text.lower()) or (phrase in clean_sentence_pass2(sent_text).lower()):
            is_valid = False

    # Rule 5: sentence must not include these phrases AT THE START of the sentence        
    for start_phrase in EXCLUDED_START_PHRASES:
        if sent_text.lower().startswith(start_phrase):
            is_valid = False
        
    return is_valid
            
def clean_sentence_pass2(sent):
    """Formats a sentence to be more easily processed downstream for classifying them as regulatory or not.

        Parameters
        ----------
        sent: str
            The sentence.
        
        Returns
        -------
            The processed sentence.

    """
    global START_TOKENS

    # Remove unncessary tokens at beginning of sentence e.g.
    # "Article 4    Heading of Article... now starts the relevant part of the sentence"
    sent_tokens = sent.split()
    if sent_tokens[0].strip() in START_TOKENS:

        if sent_tokens[1].strip().isnumeric():
            if sent_tokens[2].strip()[0].isupper():
                # find position / index of next upper case token in sent
                i = get_index_of_next_upper_case_token(sent_tokens)
                if i > 2:
                    return ' '.join(sent_tokens[i:])
                else:
                    return ' '.join(sent_tokens[3:])
            else:
                return ' '.join(sent_tokens[2:])
        else:
            return ' '.join(sent_tokens)
    else:
        return ' '.join(sent_tokens)

def clean_sentence_pass1(sent_text):
    """Formats a sentence to be more easily processed downstream for classifying them as regulatory or not.

        Parameters
        ----------
        sent: str
            The sentence.
        
        Returns
        -------
            The processed sentence.

    """
    # Rule 1: remove ':' at the start of sentence (it is there because the begin_phrase sometimes includes ':' and sometimes not
    if sent_text[0] == ':':
        sent_text = sent_text[1:].strip()
        
    # Rule 2: remove regex 'Article [some number] C' where 'C' is a capital letter
    done = False
    while not done:
        pattern = re.compile(r"\bArticle \s*\d\d?\d?[a-z]?\s*[A-Z]")
        matches = re.findall(pattern, sent_text)
        if len(matches) == 0:
            done = True
        else:
            idx_lst_char = len(matches[0]) - 1
            sent_text = sent_text.replace(matches[0], matches[0][idx_lst_char])

    return sent_text.strip()
    
def extract_summary(text):
    """Formats a text string for easy sentence tokenization and labelling / classification later.

        Parameters
        ----------
        text: str
            The input text.
        
        Returns
        -------
            Formatted text.

    """
    sent_list = get_sentence_list(text)
    
    new_sent_list = []
    for sent in sent_list:
        tmp_sent = clean_sentence_pass1(sent)
        if is_valid_sentence(tmp_sent):
            new_sent_list.append(clean_sentence_pass2(tmp_sent))

    return '\n\n\n'.join(new_sent_list)

def extract_text_from_pdf(filename, begin_phrases=BEGIN_PHRASES, end_phrases=END_PHRASES):
    """ Extracts only the raw text of PDF document that occurs between the two given phrases. 
        
            Gives only the first occurrence
        
        Parameters
        ----------
        filename: str
            Input filename string.
        begin_phrases: list
            List of string phrases which denote the starting marker of where to start
            extracting text from in the PDF
        end_phrases: list
            List of string phrases which denote the ending marker of where to stop
            extracting text from in the PDF
        
        Returns
        -------
            Extracted and formatted text from the input PDF file

    """
    
    if filename.endswith('.pdf'):
        text = ""
        title = filename.split(".")[0].split("/")[-1]

        with fitz.open(filename) as doc:
            for page in doc:
                current_page_text = page.get_text(sort=True)
                text += current_page_text

    for bphrase in begin_phrases:
        for ephrase in end_phrases: 
            pattern = re.compile(f"(?<={bphrase})(.*?)(?={ephrase})", re.DOTALL)
            matches = re.findall(pattern, text)
            if len(matches) > 0:
                the_match = matches[0]
                the_match = the_match.replace("\n", " ")
                the_match = the_match.replace("­ ", "")
                simpler_text = extract_summary(the_match)
                return simpler_text
        
    return ''

def extract_text_from_html(filename, begin_phrases=BEGIN_PHRASES, end_phrases=END_PHRASES):
    """ Extracts only the raw text of HTML document that occurs between the two given phrases. 
        
            Gives only the first occurrence
        
        Parameters
        ----------
        filename: str
            Input filename string.
        begin_phrases: list
            List of string phrases which denote the starting marker of where to start
            extracting text from in the HTML
        end_phrases: list
            List of string phrases which denote the ending marker of where to stop
            extracting text from in the HTML
        
        Returns
        -------
            Extracted and formatted text from the input HTML file

    """
    
    if filename.endswith('.html'):
        title = filename.split(".")[0].split("/")[-1]

        # Opening the html file
        html_file = open(filename, "r")
        # Reading the file
        index = html_file.read()
        # Creating a BeautifulSoup object and specifying the parser
        s = BeautifulSoup(index, 'lxml')

        for bphrase in begin_phrases:
            for ephrase in end_phrases: 
                pattern = re.compile(f"(?<={bphrase})(.*?)(?={ephrase})", re.DOTALL)
                matches = re.findall(pattern, s.text)
                if len(matches) > 0:
                    the_match = matches[0]
                    the_match = the_match.replace("\n", " ")
                    the_match = the_match.replace("­ ", "")
                    simpler_text = extract_summary(the_match)
                    return simpler_text
        
    return ''

def remove_stop_words(text):
    """ Removes unwanted tokens from text
        
            This is a custom function for this dataset. The main purpose is for 
            doing more accurate or useful word counts of documents without
            taking into account stopwords or words that do not contain any 
            valuable meaning for this particular use case - i.e., identifying
            substantive regulatory statements or legal obligations in EU legislative text.
        
        Parameters
        ----------
        text: str
            Input text string.
        
        Returns
        -------
            Processed text without custom stopwords and of useful length.

    """

    stopwords = ['the', 'and', 'this', 'that', 'for', 'with', 'are', 'its', 'which', 'have', 'has', 'these', 'those', 'from', 'was', 'were', 'had', 'into', 'then']
    tokens = text.split()
    for i in range(0, len(tokens)):
        tokens[i] = re.sub(r'[^\w\s]', '', tokens[i]) # remove punctuation
        tokens[i] = tokens[i].replace(' ', '') # remove whitespace
        
    # remove stop words and words that are less than 3 characters long
    relevant_tokens = []
    for token in tokens:
        if (token.lower() not in stopwords) and (len(token) > 2):
            relevant_tokens.append(token)
    
    return ' '.join(relevant_tokens)
            
def get_doc_lengths(text):
    """ Calculates two metrics of document length: word count and sentence count

        Parameters
        ----------
        
        text: str
            Input text string.
        
        Returns
        -------
            word count: int,
                Number of substantive words in the given text

            sent_count: int,
                Number of sentences in the given text

    """
    
    sent_count = len(text.split('\n\n\n'))
    word_count = len(remove_stop_words(text).split())
    
    return word_count, sent_count

def get_deontic_type(sent, deontics=DEONTICS):
    """ Identifies which deontic words appear in a given sentence.

        Parameters
        ----------
        
        sent: str
            Input sentence
        deontics: list
            List of deontic words or phrases
        
        Returns
        -------
            Pipe-delimited string of deontic phrases in the sentence

    """
    result = []
    for deontic in deontics:
        if deontic in (" ".join(sent.split())):
            result.append(deontic)
    if len(result) == 0:
        return 'None'
    else:
        return ' | '.join(result)
    
def identify_info(filename, text, deontics=DEONTICS):   
    """ Extracts metadata and sentences from a document

        Parameters
        ----------
        
        filename: str
            Filename (not path) of document
        text: str
            Text in the document to extract metadata and sentences from
        
        Returns
        -------
            List of lists where each list is a row in a dataframe or table:
            [celex, sent, deontic, word_count, sent_count, doc_format]
            celex: identifier for document
            sent: a sentence extracted from that document
            deontic: pipe-delimited string which represents the list of deontic words in the sentence
            word_count: number of substative words in the document
            sentence_count: number of sentences in the document
            doc_format: PDF or HTML?

    """
    word_count, sentence_count = get_doc_lengths(text)
    rows = []
    sents = text.split('\n\n\n')
    doc_format = 'pdf' if filename.endswith('.pdf') else 'html'
    
    # Filter out sentences that include negative flags for regulatory text
    for sent in sents:
        exclude = False
        for item in EXCLUDED_PHRASES:
            if fuzz.ratio(sent.strip(), item) >= 90:
                exclude = True
        
        if not exclude:
            current_row = []
            current_row.append(filename.replace('.pdf','').replace('.html','')) # celex number (identifier) of document
            current_row.append(sent.strip()) # sentence text
            deontic_types = get_deontic_type(sent.strip())
            current_row.append(deontic_types) # deontic types in the sentence
            current_row.append(word_count) # word count in document
            current_row.append(sentence_count) # sentence count in document
            current_row.append(doc_format) # PDF or HTML?
            if deontic_types != 'None':
                rows.append(current_row)
        
    return rows

# END: function definitions
# BEGIN: process input and generate prepared data for:
# 1. Ground truth labelling by legal experts (regulatory (1) or constitutive (0) and attribute label)
#    -> Also used as training data for few shot text classifier
# 2. Evaluation of rule-based NLP dependency parser analysis algorithm (regulatory (1) or constitutive (0) and attribute label)

rows = []

# Process documents
with os.scandir(INPUT_DIR) as iter:
    for i, filename in enumerate(iter):
        if filename.name.lower().endswith('.pdf'): # PDFs
            new_doc = extract_text_from_pdf(os.path.join(INPUT_DIR, filename.name))
        elif filename.name.lower().endswith('.html'): # HTMLs
            new_doc = extract_text_from_html(os.path.join(INPUT_DIR, filename.name))
        rows.extend(identify_info(filename.name, new_doc))

# Write dataframe to file
df = pd.DataFrame(rows, columns=['celex', 'sent', 'deontic', 'word_count', 'sent_count', 'doc_format'])
df.to_csv(OUTPUT_FILE, index=False)
