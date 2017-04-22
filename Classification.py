import unicodecsv as csv
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt


def gerar_arquivo(lista, n_samples, nome_do_arquivo):
    samples = random.sample(lista, n_samples)
    #lista.remove(samples)
    c = csv.writer(open(nome_do_arquivo, "wb"), delimiter =';', dialect='excel', encoding='utf-8')
    for line in samples:
        c.writerow(line)
    return lista

lista = []

with open("/home/diego/Documentos/trustvox/trustvox_production_opinion.csv", "rb") as file:
    spamreader = csv.reader(file, delimiter=';', lineterminator='/n')
    for row in spamreader:
        if row[1] is not u'':
            lista.append(row)

n_samples = 1000
for i in range(3):
    lista = gerar_arquivo(lista, n_samples=n_samples, nome_do_arquivo= "opinion " + str(i)+ ".csv")

#lista = np.array(lista)

#vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer='char',
#                             use_idf=False)

#som = SimpleSOMMapper((20, 30), 400, learning_rate=0.05)

#som.train(vectorizer)
#plt.imshow(som.K, origin='lower')
