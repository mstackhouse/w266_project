'''
buildVocab.py
    Builds vocabulary off of the provided data
    following CAML tokenization and preprocessing instructions
'''
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer

from constants import *

# Import the dataset
data = pd.read_csv(f'{DATA_DIR}{TRAINING_DATA}')

# Get the corpus
corpus = data.SYMPTOM_TEXT.dropna()

# Create a custom analyzer for the the vectorizer object
class CustAnalyzer():
    
    def __init__(self):
        # Steal the defaul preprocessor and tokenizer from sklearn
        v = CountVectorizer()
        self.preprocess = v.build_preprocessor()
        self.tokenize = v.build_tokenizer()
        self.is_num = re.compile(r'\b\d+\b') # isolated numbers
        
    def __call__(self, doc):
        # default clean and tokenize
        doc_clean = self.preprocess(doc)
        tokens = self.tokenize(doc)
        
        # Return all tokens that aren't isolated numbers
        return [t for t in tokens if not is_num.match(t)]

# Create the vectorizer
vectorizer = CountVectorizer(
    min_df = 3, 
    analyzer=CustAnalyzer(), 
    binary=True
)

# Fit to the corpus
vectorizer.fit(corpus)

print(f'Found {len(vectorizer.vocabulary_.keys()}) tokens')
print(f'Writing out to {DATA_DIR}...')
with open(f'{DATA_DIR}vocab.csv', 'w') as vocab_file:
    for word in vectorizer.keys():
        vocab_file.write(word + "\n")