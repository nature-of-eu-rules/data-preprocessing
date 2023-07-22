# data-preprocessing

Data pre-processing scripts for the [Nature of EU Rules project](https://research-software-directory.org/projects/the-nature-of-eu-rules-strict-and-detailed-or-lacking-bite). There are two scripts in this repo for extracting sentences from EU legislative documents, one includes batch processing by year to preserve results in case of premature termination or failure of the script. The scripts are described below:

#### Sentence Extractor

A script for extracting ***potentially*** regulatory sentences in EU legislative documents.

##### Input
 A directory containing either .pdf or .html (or both) EU legislative documents downloaded from [EURLEX](http://eur-lex.europa.eu/). A Python script for downloading such documents automatically is available [here](https://github.com/nature-of-eu-rules/data-extraction).
 
##### Description

The ```extract_sentences.py``` script:
 
 1. Extracts the regulatory part of the text in an EU legislative document (identified by key phrases marking the beginning and end of this text such as ```HAS ADOPTED THIS REGULATION``` and ```Done at Brussels``` respectively.
 2. Tokenizes this portion of the text into sentences using the [LexNLP](https://github.com/LexPredict/lexpredict-lexnlp) ```get_sentence_list``` function.
 3. Filters this list of sentences for those that are ***potentially*** regulatory in nature. These sentences should include a [deontic](https://www.rep.routledge.com/articles/thematic/deontic-modals/v-1) phrase (e.g. "shall", "shall not", "must", "must not") and must as far as possible use these phrases in a regulatory manner on some agent. E.g. "Member states shall apply this measure on..." is a positive example of a potentially regulatory sentence. The phrase "This regulation shall be binding in its entirety and directly applicable in the member states" is a negative example of a non-regulatory sentence in that it does not describe a specific legal obligation for a specific agent. Sentences of the latter kind are filtered out (as far as possible) using a predefined dictionary of phrases to exclude (including in the code itself). It does not matter at this stage if this filtering is one hundred percent accurate because we will later try to classify the output sentences as regulatory/non-regulatory in a different part of the project [here](https://github.com/nature-of-eu-rules/regulatory-statement-classification).

 ##### Output
 A CSV file with the following columns of data:

 | # | name | description | type | example value |
| :---: | :-------------: | :-------------: | :-------------: | :-------------: |
| 1 | celex  | [CELEX identifier]([CELEX](https://eur-lex.europa.eu/content/help/eurlex-content/celex-number.html)) for a specific EU legislative document | string  | 32019D0001
| 2 | sent | A unique sentence from the document identified by the CELEX number  | string | "Member states shall take measures to inform the Commission about..." |
| 3 | deontic | pipe-delimited list of deontic phrases used in this sentence  | string | "shall &#124; must not" |
| 4 | word_count | Number of unique words in the regulatory part of the document referred to by the CELEX number (minus predefined custom stopwords list)  | integer | 134 |
| 5 | sent_count | Number of unique sentences in the regulatory part of the text of the document referred to by the CELEX number | integer | 23 |
| 6 | doc_format | The format or file extension of the document referred to by the CELEX number | string | "HTML"|

#### Sentence Extractor (Batch)
The ```extract_sentences_batch.py``` script is functionally the same as ```extract_sentences.py``` except for splitting the input documents to process the input documents sequentially by year (1 batch per year) and saving the results of each batch to disk before moving to the next. This is to ensure saving of output data periodically to disk as it executes for longer runtimes, avoiding the problem that arises if the script runs for a long time and terminates prematurely without saving any results to disk, thereby requiring one to restart the processing from scratch again. The resulting output CSV data has still the same structure described in the previous section, but multiple such files are generated by the script (one for each document year).

#### Requirements
+ [Python](https://www.python.org/downloads/) 3.9.12+
+ A tool for checking out a [Git](http://git-scm.com/) repository.
+ A directory containing .html and / or .pdf EU legislative documents downloaded from [EURLEX](http://eur-lex.europa.eu/). For example, using the following [Python script](https://github.com/nature-of-eu-rules/data-extraction).

#### Usage steps for ```extract_sentences.py```

***Note:*** analogous steps can be followed to run ```extract_sentences_batch.py```

1. Get a copy of the code:

        git clone git@github.com:nature-of-eu-rules/data-preprocessing.git
    
2. Change into the `data-preprocessing/` directory:

        cd data-preprocessing/
    
3. Create new [virtual environment](https://docs.python.org/3/library/venv.html) e.g:

        python -m venv path/to/virtual/environment/folder/
       
4. Activate new virtual environment e.g. for MacOSX users type: 

        source path/to/virtual/environment/folder/bin/activate
        
5. Install required libraries for the script in this virtual environment:

        pip install -r requirements.txt

6. Check the command line arguments required to run the script by typing:

        python extract_sentences.py -h
        
        OUTPUT >
        
        usage: extract_sentences.py [-h] -in INPUT -out OUTPUT

        EU Legislation Regulatory Text and Sentence Extractor

        optional arguments:
        -h, --help            show this help message and exit

        required arguments:
        -in INPUT, --input INPUT
                                Path to directory containing PDF and / or HTML EU legislative documents as downloaded using code from: https://github.com/nature-of-eu-rules/data-extraction
        -out OUTPUT, --output OUTPUT
                                Path to a CSV file which should store extracted sentences from the regulatory part of the input EU legislative documents found in the input folder e.g. 'path/to/sentences.csv'.

7. Example usage: 

        python extract_sentences.py --input path/to/inputfiles/ --output path/to/output/file.csv
        

##### License

Copyright (2023) [Kody Moodley, The Netherlands eScience Center](https://www.esciencecenter.nl/team/dr-kody-moodley/)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
