import sklearn
import os
import numpy as np
import matplotlib.pyplot as plt
from Classifications import *
from Util import read_file, merge_lists, generate_n_gram_database, vectorize_database_tfidf, split_database, encoding_labels
import cPickle
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import metrics

database = []
labels = []
test = []
test2 = []

for path_to_file in os.listdir("Data"):
    data, labe = read_file(os.path.join("Data", path_to_file))
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
labels = np.array(labels)[:, 1]
labels = labels.tolist()
data = vectorize_database_tfidf(database)
labels = map(lambda x:x.lower(), labels)

X_train, X_test, y_train, y_test = split_database(data, labels)

resuts = []
for i in range(30):
    resuts.append(logistic_regression_classification(X_train, y_train, X_test, y_test))

print np.mean(np.array(resuts))
print np.std(np.array(resuts))
'''
labels = encoding_labels(labels,['0', 'bad', 'dangerous', 'excellent', 'good', 'junk', 'neutral', 'neutro', 'offensive', 'store'])
best_result ={'random_base': 0, 'result': 0 , 'kernel':'a'}
for i in range(100):
    for kernel in ['linear', 'rbf', 'poly']:
        X_train, X_test, y_train, y_test = split_database(data, labels, i)
        result = svm_classification(X_train, y_train, X_test, y_test, kernel)
        print result
        if result > best_result['result']:
            best_result['result'] = result
            best_result['random_base'] = i
            best_result['kernel'] = kernel

print best_result
'''
