import sklearn
import os
import numpy as np
import matplotlib.pyplot as plt
from Classifications import *
from Util import read_file, merge_lists, generate_n_gram_database, vectorize_database_tfidf, split_database, encoding_labels, replace_data, load_database
import cPickle
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import metrics



database, labels, vectorizer = load_database()

replace_data(labels, 'Pro', 'Product')
replace_data(labels, 'Prod', 'Product')
replace_data(labels, ' ', '0')


X_train, X_test, y_train, y_test = split_database(database, labels)

resuts = []
for i in range(100):
    resuts.append(mlp_classification(database, labels, i))

print np.mean(np.array(resuts))
print np.std(np.array(resuts))
print np.array(resuts).max()
print np.argmax(np.array(resuts))
