import gensim
import word_embeddings
from collections import defaultdict
from constants import *


#Create file dictionary - Configure and update this to add additional files to embeddings
#Key - full file path, value - index of text column
files = {
        f'{DATA_DIR}{train_split}': 1 #Training data
        }
embedding_name = 'train_only'

#Train w2v embeddings (note: can be skipped if model already trained)
print('Begin training embeddings...')
w2v_file = word_embeddings.word_embeddings(embedding_name, files, 100, 3, 5)
model = gensim.models.Word2Vec.load(w2v_file)

#Extract word vectors
wv = model.wv
del(model)

#Use vocabulary to create word vector matrix
print('Create word vector matrix...')
ind2w = defaultdict(str)
vocab = set()
with open(f'{VOCAB_DIR}'+'vocab.csv', 'r') as f:
    for i, line in enumerate(f):
        vocab.add(line.rstrip())
ind2w = {i+1:w for i,w in enumerate(sorted(vocab))}

W, words = extract_wvs.build_matrix(ind2w, wv)

#Write out embeddings
print('Write out embedding matrix...')
with open(f'{DATA_DIR}{embedding_name}'+'.embed', 'w') as f:
    for i in range(len(words)):
        line = [words[i]]
        line.extend([str(d) for d in W[i]])
        f.write(" ".join(line) + "\n")

