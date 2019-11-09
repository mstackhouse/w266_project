'''
supplementalData.py
    Build a supplemental dataset for descriptions of each label
    in the dataset
'''
import nltk
import pandas as pd
import wikipedia
from constants import *

# Ensure that the punkt package exists
nltk.download('punkt')

# Build setence segmenter using NLTK predefined punkt
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle').tokenize

# Tracker for which search is returned
tracker = {
    'original': 0,
    'trimmed': 0,
    'failed': 0,
}

# Initialize a dictionary to track what labels go into the text
# Define necessary functions
def get_wiki_desc(text, sent_detector=None):
    ''' Search Wikipedia for a segment text and return the 
        first sentence of the summary. The first setence is determined '''
    
    # Setence detector must be a callable
    assert callable(sent_detector), \
           "Argument sent_detector must be a callable"
    
    # Search wikipedia
    try:
        desc = wikipedia.WikipediaPage(title = text).summary
        # Segment setentences and use first one
        return sent_detector(desc)[0]
    # Otherwise return None to allow boolean evaluation
    except Exception as e:
        return False

def get_supplemental_dec(label, sent_detector):
    ''' Returns tuple of the original text label and its 
        scraped description. The methodology for finding 
        a description is as follows:
            - The full text is searched and the first sentence 
              of the summary is used
            
            - If no page exists for the full text, then the last 
              word is removed and the search is reattempted. This 
              helps retrieve meaningful text for labels like 
              "Blood amylase increased", where a page does not exist 
              for the full text, but a page exists for "Blood amylase" 
              and the final word is simply descriptive
              
            - If removing the last word still fails to return a page, 
              then the label text itself is used as the description. 
              This is still highly beneficial in that many triggers 
              in the training data are the label itself
    '''
    
    # Copy since this might be updated
    text = label
    
    # Retrive the initial description
    desc = get_wiki_desc(text, sent_detector)
    
    # Mark original text success
    if desc:
        tracker['original'] += 1
    
    # If not found reattempt by removing last word
    if not desc:
        text = ' '.join(text.split(' ')[:-1])
        desc = get_wiki_desc(text, sent_detector)
        # Mark trimmed success
        if desc:
            tracker['trimmed'] += 1
    
    # Search failed - just use label
    if not desc:
        desc = label
        # Mark failure
        tracker['failed'] += 1
        
    return label, desc

def build_supplemental_data(label_list, sent_detector=None):
    ''' Iterate a list of labels and assemble the 
        supplemental dataset.
        Returns pandas.DataFrame
    '''
    
    # Initialize a dictionary to hold observations
    data_dict = {'label': [], 'desc':[]}
    
    # Counter
    i = 0
    
    # Loop the input list
    for label in label_list:
        # Search the term
        label, desc = get_supplemental_dec(label, sent_detector)
        # Add to the dictionary
        data_dict['label'] += [label]
        data_dict['desc'] += [desc]
        
        # Increment counter
        i += 1
        
        # progress marker
        if not i % 100:
            print(f'Completed {i} searches')
        
    # Return as a data frame
    return pd.DataFrame(data_dict)

if __name__ == "__main__":
    # Open the label set and store as list
    with open(f'{LABEL_DIR}labels.csv', 'r') as f:
        labels = [x.replace('\n', '') for x in f.readlines()]
    
    print(f'Found {len(labels)} to search.')
    print('Starting search...')
    # Build the supplemental dataset
    sup_data = build_supplemental_data(labels, tokenizer)

    # Write out to tab delemitted text
    sup_data.to_csv(f'{DATA_DIR}supplement_data.txt', sep="\t",
                    index=False)

    # Report the tracker
    print('Data assembly complete')
    print('Summary of search results:')
    print(tracker)

