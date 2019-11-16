'''
supplementalData.py
    Build a supplemental dataset for descriptions of each label
    in the dataset
'''
import re
import datetime as dt
import nltk
import pandas as pd
import wikipedia
from tqdm import tqdm
from constants import *
from buildVocab import CustAnalyzer


# Ensure that the punkt package exists
nltk.download('punkt')

# Build setence segmenter using NLTK predefined punkt
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle').tokenize

# Regex to use to search within disambiguation suggestions
reg = re.compile(r'.*\((medi\w*)\)', re.I)
chars = re.compile(r'[\x00-\x1f]+')

# Tracker for which search is returned
tracker = {
    'original': 0,
    'disambiguation': 0,
    'trimmed': 0,
    'trimmed_disambiguation': 0,
    'failed': 0,
    'total_searches': 0
}

def search_wiki(text, sent_detector=tokenizer):
    ''' Wrapper for the search call to Wikipedia
        Additionally splits by setence and only returns the first one'''
    
    # Setence detector must be a callable
    assert callable(sent_detector), \
           "Argument sent_detector must be a callable"
    
    # Mark that a search occured
    tracker['total_searches'] += 1
    
    # Run the search
    desc = chars.sub(' ',
                     wikipedia.WikipediaPage(title = text).summary
    )
    
    # Split by setences and return the first one
    return sent_detector(desc)[0], desc

def get_wiki_desc(text, original_term=None):
    ''' Returns tuple of the original text label and its 
        scraped description. The methodology for finding 
        a description is as follows:
            - The full text is searched and the first sentence 
              of the summary is used
            
            - If no page exists for the full text, then either:
                - A disambiguation error occured. If any disambiguation
                  suggestion contained "medi", use that and research, 
                  otherwise take the first alternate suggestion if less
                  than 10 suggestions were available. More than that 
                  is too ambiguous so consider the search failed
                - If no pages existed, a word is removed and the search 
                  is reattempted. This helps retrieve meaningful text 
                  for labels like "Blood amylase increased", where a 
                  page does not exist for the full text, but a page exists 
                  for "Blood amylase" and the final word is simply descriptive

            - If removing the last word still fails to return a page, 
              then the label text itself is used as the description. 
              This is still highly beneficial in that many triggers 
              in the training data are the label itself
    '''
    
    # Mark if this is a trimmed search and save
    # original text
    if original_term:
        trimmed = True
    else:
        trimmed = False
        original_term = text
        
    # Search wikipedia
    try:
        desc, summary = search_wiki(text)
        # Update the tracker
        if trimmed:
            tracker['trimmed'] += 1
        else:
            tracker['original'] += 1
        # Segment setentences and use first one
        return original_term, desc, summary
    
    # If multiple potential options then try to pick one out
    except wikipedia.DisambiguationError as e:
        # Get the alternates
        alternates = str(e).split("\n")[1:]

        # placeholder
        new_term = None

        # Scan the alternates to find if any are medical
        for alt in alternates:
            # Is it a medical/medicine disambiguation?
            # For example - Syncope (medicine)
            if reg.findall(alt):
                new_term = alt
                
        # None of them have a mention of medice so if less than 10 alternates, 
        # take the first disambiguation suggestion
        if not new_term and len(alternates) < 10:
            new_term = alternates[0]
        
        # Picked an alternate so search
        if new_term:
            # Search the disamgiuation term
            try:
                desc, summary = search_wiki(new_term)
                # Update the tracker
                if trimmed:
                    tracker['trimmed_disambiguation'] += 1
                else:
                    tracker['disambiguation'] += 1
                return original_term, desc, summary
            except Exception as e:
                # This shouldn't happen but this takes a while so I don't
                # want a failure at term 10000
                print(f"Disambiguation failed!!! Search term: {alt}")
                print(f'\tError: {e}')
                tracker['failed'] += 1
                return original_term, original_term, original_term
                
        # Too ambiguous so stick with just returning the label
        else:
            tracker['failed'] += 1
            return original_term, original_term, original_term
    
    # No page found so trim
    except wikipedia.PageError:
        # don't search again if this is a trimmed search
        if not trimmed and text.find(' ') > -1:
            # Save the original label
            original_term = text
            # Trim the last word off the text
            text = ' '.join(text.split(' ')[:-1])
            # Recursive search
            return get_wiki_desc(text, original_term)
            
        else:
            # Text was already trimmed or nothing to trim so search has failed
            tracker['failed'] += 1
            return original_term, original_term, original_term
    except Exception as e:
        print(f'Unexpected failure on term {original_term}')
        print(f'\tError: {e}')
        tracker['failed'] += 1
        return original_term, original_term, original_term

def build_supplemental_data(label_list, sent_detector=None):
    ''' Iterate a list of labels and assemble the 
        supplemental dataset.
        Returns pandas.DataFrame
    '''
    
    # Initialize a dictionary to hold observations
    data_dict = {'label': [], 'desc':[], 'summary': []}
    
    # Counter
    i = 0
    
    # Loop the input list
    for label in tqdm(label_list):
        # Search the term
        label, desc, summary = get_wiki_desc(label)
        # Add to the dictionary
        data_dict['label'] += [label]
        data_dict['desc'] += [desc]
        data_dict['summary'] += [summary]
        
        # Increment counter
        i += 1
        
        # Mark progress
        # print(f'Completed {i} searches', end="\r")
        
    # Return as a data frame
    return pd.DataFrame(data_dict)

if __name__ == "__main__":
    # Open the label set and store as list
    with open(f'{LABEL_DIR}labels.csv', 'r') as f:
        labels = [x.replace('\n', '') for x in f.readlines()]
    
    start_time = dt.datetime.now()

    print(f'Found {len(labels)} labels to search.')
    print(f'Starting search at {start_time.strftime("%T")}...')
    # Build the supplemental dataset
    sup_data = build_supplemental_data(labels, tokenizer)

    # Write out to tab delemitted text
    sup_data.drop(['summary'], axis=1)\
        .to_csv(f'{DATA_DIR}supplement_data.txt', sep="\t",
                index=False)

    # Pre-process the supplemental data summaries
    tokenize = CustAnalyzer(mask_dates=True, max_length=None)
    sup_data['summary'] = sup_data.summary\
        .apply(lambda x: ' '.join(tokenize(x)))
    # Write out to tab delemitted text
    sup_data.drop(['desc'], axis=1)\
        .to_csv(f'{DATA_DIR}wiki_data.csv',
                index=False)

    # Calculate end time
    total_time = dt.datetime.now() - start_time
    # Report the tracker
    print(' ' * 100)
    print('Data assembly complete')
    print(f"Total runtime: {str(total_time).split('.', 2)[0]}")
    print('Summary of search results:')
    print(tracker)

