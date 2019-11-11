"""
    Pre-train embeddings using gensim w2v implementation (CBOW by default)
"""
import gensim.models.word2vec as w2v
import csv

from constants import *

class ProcessedIter(object):

    def __init__(self, embed_name, filedict):
        self.filedict = filedict

    def __iter__(self):
        for key, value in self.filedict.items():
            with open(key, 'r') as f:
                r = csv.reader(f)
                next(r)
                for row in r:
                    yield (row[value].split())

def word_embeddings(embed_name, file_dict, embedding_size, min_count, n_iter):
    """
    embed_name: the embedding name you want the file saved as.  File willbe saved as "processed_{embed_name}.w2v"
    file_dict: a dictionary of files and the index of the text column.  To configure:
        ####
        from constants import *
        files = {f'{DATA_DIR}{train_split}': 1}
        ####
    embedding_size: default in paper is 100
    min_count: default in paper is 3
    n_iter: default in paper is 5
    
    
    Sample execution code:
    ####
    w2v_file = word_embeddings.word_embeddings('train_only', files, 100, 3, 5)
    model = gensim.models.Word2Vec.load(w2v_file)
    ####
    
    """
    modelname = "processed_%s.w2v" % (embed_name)
    sentences = ProcessedIter(embed_name, file_dict)

    model = w2v.Word2Vec(size=embedding_size, min_count=min_count, workers=4, iter=n_iter)
    print("building word2vec vocab on %s..." % (list(file_dict.keys())))
    
    model.build_vocab(sentences)
    print("training...")
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
    out_file = f'{DATA_DIR}{modelname}'
    print("writing embeddings to %s" % (out_file))
    model.save(out_file)
    return out_file