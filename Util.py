# -*- coding: utf-8 -*-
import csv
import string
import nltk
import random
from sklearn.feature_extraction.text import CountVectorizer
import cPickle
from nltk.corpus import stopwords
from nltk import stem, word_tokenize
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
from nltk import stem
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
import numpy as np

def read_file(path):
    """
        Método responsável por ler os arquivos csv da base de dados e retornar os reviews e labels
        para serem trabalhados
    :param path: endereço do arquivo a ser lido
    :return: database = base com todos os reviews lidos pelo método
             labels = os labels referentes a cada um dos reviews
    """
    database = []
    labels = []
    with open(path, "rb") as csv_file:
        next(csv_file)
        spamreader = csv.reader(csv_file, delimiter = ',')
        for row in spamreader:
            row = map(lambda x: x if x != '' else '0', row)
            database.append(row[0])
            labels.append(row[1:3])

    return database, labels

def merge_lists(lists):
    """
        Método responsável por unir 3 listas em uma só
    :param lists: listas a serem unidas
    :return: uma lista com as informações contidas nas listas prévias
    """
    return lists[0] + lists[1] + lists[2] + lists[3]

def find_ngrams(input_list, n):
    """
        Método que calcula o n_gram para uma lista de sentença/frase
    :param input_list: lista com as sentenças/frase a serem extraídos
    :param n: tamanho do n_gram
    :return: retorna uma lista com os n_grams das sentenças/frase extraídos
    """
    return zip(*[input_list[i:] for i in range(n)])

def generate_n_gram_for_a_sentence(sentence = str, range = int):
    """
        Método que calcula o n_gram para uma lista de sentença/frase
    :param sentence: sentença/frase a ter seu n gram extraído
    :param range: tamanho do n gram
    :return: uma lista com o n gram da sentença/frase
    """
    vectorizer = CountVectorizer(ngram_range=(range, range))
    analyzer = vectorizer.build_analyzer()
    return analyzer(sentence)

def generate_n_gram_database(database = [], n = int, file_name = str):
    """
        Método responsável por extrai as características do tipo n gram de uma base de texto.
    :param database: Lista com todas as sentenças a terem seu n gram extraído
    :param n: tamanho do n gram
    :param file_name: arquivo onde será salvo o resultado
    :return: a base com o n gram extraído
    """
    n_gram_database = []
    for sentence in database:
        n_gram_sentence = generate_n_gram_for_a_sentence(sentence=sentence, range=n)
        n_gram_database.append(n_gram_sentence)
    cPickle.dump(n_gram_database, open(file_name, 'wb'))
    return n_gram_database

def vectorize_database_tfidf(database):
    database = map(lambda x: x.lower(), database)
    database_ = [tokenize(sentence) for sentence in database]

    pt_stop_words = set(stopwords.words('portuguese'))
    pt_stop_words.add('pra')
    pt_stop_words.add('para')

    vectorizer = TfidfVectorizer(max_df=0.75, max_features=5000, lowercase=False, min_df=2, stop_words=pt_stop_words, ngram_range=(1, 4),
                                 use_idf=True)
    data = vectorizer.fit_transform(database_)

    return data.todense(), vectorizer

def vectorize_database_hash(database):
    pt_stop_words = set(stopwords.words('portuguese'))
    vectorizer = HashingVectorizer(n_features=2000,
                                   stop_words=pt_stop_words, ngram_range=(2, 6), lowercase=True,
                                   non_negative=False, norm='l2',
                                   binary=False)
    data = vectorizer.fit_transform(database)

    cPickle.dump(data, open('salvar_hash.test', 'wb'))

    return data

def split_database(database=[], labels =[]):

    database = np.array(database)
    labels = np.array(labels)
    return train_test_split(database, labels, test_size=0.5)

def load_database():
    import os
    database = []
    labels =[]
    database, labels = read_file('Data/database.csv')

    labels = np.array(labels)
    labels = labels
    replace_data(labels, 'Pro', 'Product')
    replace_data(labels, 'Prod', 'Product')
    replace_data(labels, ' ', '0')

    database, vectorizer = vectorize_database_tfidf(database)
    return database, labels, vectorizer

def replace_data(list_labels, itens_to_replace, replacement_value):
    indices_to_replace = [i for i,x in enumerate(list_labels) if x[0]==itens_to_replace]
    for i in indices_to_replace:
        list_labels[i][0] = replacement_value

def write_csv(database, labels):
    a = database + labels.tolist()
    with open('database.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for line in a:
            spamwriter.writerow(a)


def encoding_labels(labels, labels_to_encode):
    le = preprocessing.LabelEncoder()
    le.fit(labels_to_encode)
    return le.transform(labels)

def tokenize(text):
    steemming = stem.RSLPStemmer()
    tokens = text.decode('utf8').split()
    tokens = [i for i in tokens if i not in string.punctuation]
    stems = [steemming.stem(token) for token in tokens]
    stems = join_strings(stems)
    return stems

def join_strings(list_of_strings):
    return " ".join(list_of_strings)

def load_database2():
    import os
    database = []
    labels =[]
    database, labels = read_file('Data/database.csv')

    labels = np.array(labels)
    labels = labels
    replace_data(labels, 'Pro', 'Product')
    replace_data(labels, 'Prod', 'Product')
    replace_data(labels, ' ', '0')

    #database, vectorizer = vectorize_database_tfidf(database)
    return database, labels