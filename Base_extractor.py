import sklearn
import os
import numpy as np
import matplotlib.pyplot as plt
from Classifications import *
from Util import read_file, merge_lists, generate_n_gram_database, vectorize_database_tfidf, split_database
import cPickle
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import metrics

database = []
labels = []
test = []
test2 = []

for path_to_file in os.listdir("Data"):
    data, labe = read_file(os.path.join("Data",path_to_file))
    database.append(data)
    labels.append(labe)


database = merge_lists(database)
labels = merge_lists(labels)

for label in labels:
    if(label[0]=='Product'):
        test.append(0)
    else:
        test.append(1)

    if (label[1] == 'Store'):
        test2.append(0)
    else:
        test2.append(1)

databases_n_gram = []
'''
for i in range(5):
    databases_n_gram.append(generate_n_gram_database(database, n=i+1, file_name='n_gram_databases/dabase_with_n_gram_'+ str(i+1) +'.data'))

classification_with_kmeans(databases_n_gram[0], 2)
'''

data = vectorize_database_tfidf(database)

X_train, X_test, y_train, y_test = split_database(data, np.array(labels)[:, 2])

#kmeans_classification(database, test, test2, 2)
#score = som_classificarion(X_train, y_train, X_test, y_test)
print logistic_regression_classification(X_train, y_train, X_test, y_test)
print svm_classification(X_train, y_train, X_test, y_test)