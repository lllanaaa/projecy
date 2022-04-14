import string
from nltk.corpus import brown
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
import pandas as pd
import pickle


# df_train = pd.read_csv('test_train_data.tsv', sep='\t', header=0)
# df_train_query = df_train[['qid', 'queries']]
#
#
# model = Word2Vec(
#     sentences=data,
#     size=50,
#     window=10,
#     iter=20,
# )

