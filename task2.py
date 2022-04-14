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
import time
import pickle


lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


class LogisticRegression1():
    def __init__(self, learning_rate, n_iterations):
        # 初始化学习率和迭代次数
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def initialize_weights(self, x, y):
        # 初始化参数
        self.m, self.n = x.shape
        self.w = np.zeros(self.n)
        self.b = 0
        self.x = x
        self.y = y

    def fit(self, x, y):
        self.initialize_weights(x, y)

        # gradient descent
        for i in range(self.n_iterations):
            self.update_weight()

    def update_weight(self):
        # gradient descent
        h = self.sigmoid(self.x.dot(self.w) + self.b)
        tmp = np.reshape(h - self.y.T, self.m)
        dw = np.dot(self.x.T, tmp) / self.m
        db = np.sum(tmp) / self.m
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db

    def predict(self, x):
        res = self.sigmoid(x.dot(self.w) + self.b)
        return res


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


def load_train_validation():
    df_train = pd.read_csv('dataset/train_data.tsv', sep='\t', header=0)
    df_validation = pd.read_csv('dataset/validation_data.tsv', sep='\t', header=0)

    # df_train = pd.read_csv('test_train_data.tsv', sep='\t', header=0)
    # df_validation = pd.read_csv('test_validation_data.tsv', sep='\t', header=0)
    return df_train, df_validation


def load_word2vec_model():
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', limit=10000, binary=True)
    # print(word2vec_model['hello'].shape)
    # print(word2vec_model['hello'])
    # print(word2vec_model.similar_by_word('hello', topn=5))
    return word2vec_model


def sentence_embedding(sentence, word2vec_model):

    terms_in_sentence = processDataRemoveStopword([sentence])[0]

    vec = np.zeros([1, 300])
    flag = 0
    for term in terms_in_sentence:
        if term in word2vec_model:
            term_embedding = word2vec_model[term]
        else:
            continue
        term_embedding = term_embedding.reshape(1, 300)
        if flag == 0:
            vec += term_embedding
            flag += 1
            continue
        vec = np.concatenate((vec, term_embedding))
        flag += 1

    vec = np.mean(vec, axis=0)

    return vec



# train
def data_preprocess_train(df_train_query, df_train_passage):
    # load word2vec model
    word2vec_model = load_word2vec_model()

    # train data
    # df_train_query = df_train[['qid', 'queries']]
    # df_train_query.drop_duplicates(subset=['queries'], keep='first', inplace=True)
    # df_train_passage = df_train[['pid', 'passage']]
    # df_train_passage.drop_duplicates(subset=['passage'], keep='first', inplace=True)

    # df_train_query.to_csv('datasets_processed/train_query.csv')
    # df_train_passage.to_csv('datasets_processed/train_passage.csv')

    # train data
    # df_query = df_train_query
    df_passage = df_train_passage
    dict_q = {}
    dict_p = {}
    # for index, row in df_query.iterrows():
    #     qid = row['qid']
    #     query = row['queries']
    #     vector_term_in_query = sentence_embedding(query, word2vec_model)
    #     dict_q[qid] = vector_term_in_query
    # with open('datasets_processed/train_query_vector.pickle', 'wb') as file:
    #     pickle.dump(dict_q, file)
    for index, row in df_passage.iterrows():
        pid = row['pid']
        passage = row['passage']
        vector_term_in_passage = sentence_embedding(passage, word2vec_model)
        dict_p[pid] = vector_term_in_passage
    with open('datasets_processed/train_passage_vector.pickle', 'wb') as file:
        pickle.dump(dict_p, file)


# validation
def data_preprocess_validation(df_validation_query, df_validation_passage):
    # load word2vec model
    word2vec_model = load_word2vec_model()

    # validation data
    # df_validation_query = df_validation[['qid', 'queries']]
    # df_validation_query.drop_duplicates(subset=['queries'], keep='first', inplace=True)
    # df_validation_passage = df_validation[['pid', 'passage']]
    # df_validation_passage.drop_duplicates(subset=['passage'], keep='first', inplace=True)

    # df_validation_query.to_csv('datasets_processed/validation_query.csv')
    # df_validation_passage.to_csv('datasets_processed/validation_passage.csv')

    # validation data
    df_query = df_validation_query
    df_passage = df_validation_passage
    dict_q = {}
    dict_p = {}
    for index, row in df_query.iterrows():
        qid = row['qid']
        query = row['queries']
        vector_term_in_query = sentence_embedding(query, word2vec_model)
        dict_q[qid] = vector_term_in_query
    with open('datasets_processed/validation_query_vector.pickle', 'wb') as file:
        pickle.dump(dict_q, file)
    for index, row in df_passage.iterrows():
        pid = row['pid']
        passage = row['passage']
        vector_term_in_passage = sentence_embedding(passage, word2vec_model)
        dict_p[pid] = vector_term_in_passage
    with open('datasets_processed/validation_passage_vector.pickle', 'wb') as file:
        pickle.dump(dict_p, file)


def process_model_input(df_train, train_query, train_passage):
    x = np.zeros([1, 600])
    y = np.zeros([1, 1])
    flag = 0

    for index, row in df_train.iterrows():
        qid = row['qid']
        pid = row['pid']
        relevancy = row['relevancy']

        vec_query = train_query[int(qid)]
        vec_passage = train_passage[int(pid)]
        vec_query_passage = np.concatenate((vec_query, vec_passage))
        vec_query_passage = vec_query_passage.reshape(1, 600)

        if flag == 0:
            x += vec_query_passage
            y += np.array([int(relevancy)]).reshape(1, 1)
            flag += 1
            continue
        x = np.concatenate((x, vec_query_passage))
        y = np.concatenate((y, np.array([int(relevancy)]).reshape(1, 1)))
        flag += 1

    return x, y


def evaluation_model(y_pred, y_test):
    # calculate mAP NDCG
    correctly_classified = 0
    count = 0
    for count in range(np.size(y_pred)):
        if y_test[count] == y_pred[count]:
            correctly_classified = correctly_classified + 1
        count = count + 1

    print("Accuracy on test set :  ", (
            correctly_classified / count) * 100)


def load_train_pickle():
    with open('datasets_processed/train_passage_vector.pickle', 'rb') as f:
        dict_p = pickle.load(f)
    with open('datasets_processed/train_query_vector.pickle', 'rb') as f:
        dict_q = pickle.load(f)
    return dict_p, dict_q



def main():
    # 1. load dataset, save query/passage vector to pickle file
    # df_train, df_validation = load_train_validation()
    df_train_query = pd.read_csv('datasets_processed/train_query.csv', sep=',', header=0)
    df_train_passage = pd.read_csv('datasets_processed/train_passage.csv', sep=',', header=0)
    data_preprocess_train(df_train_query, df_train_passage)
    df_validation_query = pd.read_csv('datasets_processed/validation_query.csv', sep=',', header=0)
    df_validation_passage = pd.read_csv('datasets_processed/validation_passage.csv', sep=',', header=0)
    data_preprocess_validation(df_validation_query, df_validation_passage)
    print("已经预处理完模型")

    # # 2. load pickle file, process them to model input
    # train_passage_vector, train_query_vector = load_train_pickle()
    #
    # # df_train = df_train[['qid', 'pid', 'relevancy']]
    # # x_train, y_train = process_model_input(df_train, train_query, train_passage)
    # df_train_test = pd.read_csv('test_train_data.tsv', sep='\t', header=0)
    # df_train_test = df_train_test[['qid', 'pid', 'relevancy']]
    # x_train_test, y_train_test = process_model_input(df_train_test, train_query_vector, train_passage_vector)
    #
    # df_validation = df_validation[['qid', 'pid', 'relevancy']]
    # x_validation, y_validation = process_model_input(df_validation, train_query_vector, train_passage_vector)
    #
    # # 3. implement a logistic regression model, query and passage embedding as input
    # model = LogisticRegression1(learning_rate=0.01, n_iterations=100)
    # model.fit(x_train_test, y_train_test)
    # y_pred = model.predict(x_validation)
    #
    # # 4. evaluation mAP / NDCG
    # evaluation_model(y_pred, y_validation)



pd.set_option('display.width', None)
main()
