import pandas as pd
import gensim
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from collections import Counter
import contractions
import re
import string
import unidecode
import numpy as np



lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def processDataRemoveStopword(data):
    stop_words = set(stopwords.words('english'))
    processedData = []
    for line in data:

        # whitespace: replace more than one space with a single space
        line = re.sub(' +', ' ', line)

        # convert to lowercase
        line = line.lower()

        # expand contractions
        line = contractions.fix(line)

        # remove punctuation
        punctuation_table = str.maketrans('', '', string.punctuation)
        line = line.translate(punctuation_table)

        # remove unicode characters
        line = unidecode.unidecode(line)

        # tokenization
        line = word_tokenize(line)

        # Lemmatisation, Stemming
        # remove stop words
        arr = []
        for i in range(len(line)):
            if line[i] not in stop_words:
                processed = lemmatizer.lemmatize(line[i])
                arr.append(processed)

        processedData.append(arr)
    return processedData


def load_dataset():
    df_train = pd.read_csv('dataset/train_data.tsv', sep='\t', header=0)
    df_validation = pd.read_csv('dataset/validation_data.tsv', sep='\t', header=0)
    return df_train, df_validation


def load_word2vec_model():
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    # print(word2vec_model['hello'].shape)
    # print(word2vec_model['hello'])
    # print(word2vec_model.similar_by_word('hello', topn=5))
    return word2vec_model


def sentence_embedding(df_train, df_validation):
    # word2vec model
    word2vec_model = load_word2vec_model()

    # train data
    df_train_query = df_train[['qid', 'queries']]
    df_train_query.df_train(subset=['queries'], keep='first', inplace=True)
    df_train_passage = df_train[['pid', 'passage']]
    df_train_passage.df_train(subset=['passage'], keep='first', inplace=True)

    # data dict
    dict = {}

    # query
    for index, row in df_train_query.iterrows():
        qid = row['qid']
        query = row['queries']
        terms_in_query = processDataRemoveStopword([query])[0]

        arr_term_in_query = np.zeros([1, 300])
        flag = 0
        for term in terms_in_query:
            if term in word2vec_model:
                term_embedding = word2vec_model[term]
            else:
                continue
            term_embedding = term_embedding.reshape(1, 300)
            if flag == 0:
                arr_term_in_query += term_embedding
                flag += 1
                continue
            arr_term_in_query = np.concatenate((arr_term_in_query, term_embedding))
            flag += 1

        arr_term_in_query = np.mean(arr_term_in_query, axis=0)
        dict[int(qid)] = arr_term_in_query


    # passage
    for index, row in df_train_passage.iterrows():
        pid = row['pid']
        passage = row['passage']
        terms_in_passage = processDataRemoveStopword([passage])[0]

        arr_term_in_passage = np.zeros([1, 300])
        flag = 0
        for term in terms_in_passage:
            if term in word2vec_model:
                term_embedding = word2vec_model[term]
            else:
                continue
            term_embedding = term_embedding.reshape(1, 300)
            if flag == 0:
                arr_term_in_passage += term_embedding
                flag += 1
                continue
            arr_term_in_passage = np.concatenate((arr_term_in_passage, term_embedding))
            flag += 1

        arr_term_in_passage = np.mean(arr_term_in_passage, axis=0)
        dict[int(pid)] = arr_term_in_passage

    return dict


def word_embedding(df_train, df_validation):

    # get query/passage embedding
    dict = sentence_embedding(df_train, df_validation)

    # logistic regression model
    # input: query/passage embeddings





def main():
    # 1. load dataset: train_data.tsv validation_data.tsv
    df_train,  df_validation = load_dataset()

    # 2. represent query - passage using word embedding (average embedding)
    word_embedding(df_train, df_validation)

    # 3. implement a logistic regression model, query and passage embedding as input

    # 4. assess relevance of query - passage  mAP and NDCG


pd.set_option('display.width', None)
main()
