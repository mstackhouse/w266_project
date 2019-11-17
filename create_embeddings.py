from collections import defaultdict
import csv
import gensim
import pandas as pd
import extract_wvs
import word_embeddings
import buildVocab
from constants import *


#Create file dictionary - Configure and update this to add additional files to embeddings
#Key - full file path, value - index of text column
files = {
        f'{DATA_DIR}train_raw.csv': 1, #Training data
        f'{DATA_DIR}device_data.csv': 1,
        f'{DATA_DIR}wiki_data.csv': 1
        }
embedding_name = 'train_device_wiki'

#Train w2v embeddings (note: can be skipped if model already trained)
print('Begin training embeddings...')
w2v_file = word_embeddings.word_embeddings(embedding_name, files, 100, 3, 5)
model = gensim.models.Word2Vec.load(w2v_file)

#Extract word vectors
wv = model.wv
del(model)

#Create Vocabulary
print('Create vocabulary..')
ind2w = defaultdict(str)

vocab, vz = buildVocab.build_vocab(
    filedict=files,
    outfile='vocab.csv', 
    mask_dates=True)

# Index to word dictionary
ind2w = {i+1:w for i,w in enumerate(sorted(vocab))}
# Flip it - word to index
w2ind = {w:i for i,w in ind2w.items()}

# Build the embedding lookup matrix
print('Build embedding matrix lookup...')
W, words = extract_wvs.build_matrix(ind2w, wv)

#Write out embeddings
print('Write out embedding matrix...')
with open(f'{DATA_DIR}{embedding_name}'+'.embed', 'w') as f:
    for i in range(len(words)):
        line = [words[i]]
        line.extend([str(d) for d in W[i]])
        f.write(" ".join(line) + "\n")

# Move on to the description vectors
print('Write description vectors with vocab')

# Read in the supplemental data
sup_data = pd.read_csv(f'{DATA_DIR}{SUPPLEMENTAL_DATA}', sep="\t")

# Tokenize
tokenizer = buildVocab.CustAnalyzer(mask_dates=True)
sup_data['tokens'] = sup_data.desc.apply(lambda x: tokenizer(x))
# Get index of each token
sup_data['inds'] = sup_data.tokens.apply(
    lambda tok: [w2ind[t] if t in w2ind.keys() else len(w2ind) + 1 for t in tok]
)
# Build the line to be used in the output file
sup_data['out_line'] = sup_data.apply(
    lambda x: [x.label] + x.inds,
    axis=1
)
# Subset
sup_data = sup_data[['out_line']]

# Write out the description vectors
with open(f'{DATA_DIR}description_vectors.vocab', 'w+') as f:
    w = csv.writer(f, delimiter=' ')
    w.writerow(['CODE', 'VECTOR'])
    sup_data.out_line.apply(lambda x: w.writerow(x))
