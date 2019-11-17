'''
buildVocab.py
    Builds vocabulary off of the provided data
    following CAML tokenization and preprocessing instructions
'''
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer

from constants import *

# Create a custom analyzer for the the vectorizer object
class CustAnalyzer():
    
    def __init__(self, mask_dates, max_length=MAX_LENGTH):
        # Steal the defaul preprocessor and tokenizer from sklearn
        v = CountVectorizer()
        self.max_length = max_length
        self.dat = re.compile(r'\b\d{1,2}\-?[a-z]{3}\-?\d{2,4}\b')
        if mask_dates:
            self.preprocess = lambda x: self.dat.sub('<DATE>', str(x).lower())
        else:
            self.preprocess = v.build_preprocessor()
        self.tokenize = v.build_tokenizer()
        self.is_num = re.compile(r'\b\d+\b') # isolated numbers
        
        
    def __call__(self, doc):
        # default clean and tokenize
        doc_clean = self.preprocess(doc)
        tokens = self.tokenize(doc_clean)
        
        # Return all tokens that aren't isolated numbers
        filtered = [t for t in tokens if not self.is_num.match(t)]
        if self.max_length:
            return filtered[:self.max_length]
        else:
            return filtered[:self.max_length]

def build_vocab(filedict=None, outfile=None, mask_dates=False, max_length=MAX_LENGTH):
    ''' Build up the vocab file from potentially multiple
        sources 
    '''

    # initialize a series
    corpus = pd.Series()

    # Loop the dictionary
    for file, column in filedict.items():
        print(f'Reading data {file}')
           # Read and pull off selected column
        tmp = pd.read_csv(f'{file}').iloc[:,column]
        # Stack back to the main dataframe
        corpus = corpus.append(tmp)

    # Get the corpus
    corpus = corpus.dropna()

    # Create the vectorizer
    vectorizer = CountVectorizer(
        min_df = 3, 
        analyzer=CustAnalyzer(mask_dates, max_length=max_length), 
        binary=True
    )

    print('Fitting vectorizer...')
    # Fit to the corpus
    vectorizer.fit(corpus)

    if outfile:
        print(f'Found {len(vectorizer.vocabulary_.keys())}) tokens')
        print(f'Writing out to {VOCAB_DIR}{outfile}...')
        with open(f'{VOCAB_DIR}{outfile}', 'w') as vocab_file:
            for word in vectorizer.vocabulary_.keys():
                vocab_file.write(word + "\n")
    
    return vectorizer.vocabulary_, vectorizer