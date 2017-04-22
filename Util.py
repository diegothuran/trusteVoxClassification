# -*- coding: utf-8 -*-
import os
import csv
from sklearn.feature_extraction.text import CountVectorizer
import cPickle

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
            labels.append(row[1:-1])

    return database, labels

def merge_lists(lists):
    """
        Método responsável por unir 3 listas em uma só
    :param lists: listas a serem unidas
    :return: uma lista com as informações contidas nas listas prévias
    """
    return lists[0] + lists[1] + lists[2]

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
