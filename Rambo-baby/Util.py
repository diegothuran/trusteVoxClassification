# -*- coding: utf-8 -*-
import csv
import time


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

def timing(f):
    """
        Método para cauluar o tempo de execução de um outro método
    :param f: Método
    :return: Tempo de execução
    """
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print '%s demorou  %0.3f ms para ser executado' % (f.func_name, (time2-time1)*1000.0)
        return ret
    return wrap