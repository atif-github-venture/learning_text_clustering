import gensim as gensim
from nltk.tokenize import word_tokenize
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
import numpy as np

file_docs = []

with open('data/locators.txt') as fp:
    for line in fp:
        tokens = word_tokenize(line)
        file_docs.append(line)

print("Number of documents:", len(file_docs))

gen_docs = [[w.lower() for w in word_tokenize(text)]
            for text in file_docs]

dictionary = gensim.corpora.Dictionary(gen_docs)
print(dictionary.token2id)

corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
tf_idf = gensim.models.TfidfModel(corpus)

for doc in tf_idf[corpus]:
    print([[dictionary[id], np.around(freq, decimals=2)] for id, freq in doc])

sims = gensim.similarities.Similarity('workdir/', tf_idf[corpus],
                                      num_features=len(dictionary))

file2_docs = []

with open('data/query.txt') as f:
    for line in f:
        query_doc = [w.lower() for w in word_tokenize(line)]
        query_doc_bow = dictionary.doc2bow(query_doc)
        query_doc_tf_idf = tf_idf[query_doc_bow]
        print('Comparing Result:', sims[query_doc_tf_idf])

# https://dev.to/coderasha/compare-documents-similarity-using-python-nlp-4odp
