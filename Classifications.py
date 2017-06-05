from Util import vectorize_database_tfidf, vectorize_database_hash
import cPickle, os
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import metrics
import random
import numpy as np
from minisom import MiniSom
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import preprocessing
from Util import encoding_labels, split_database

def kmeans_classification(database = [], labels = [], labels2 =[], n_clusters = int):
    data_tfidf = vectorize_database_tfidf(database)
    data_hash = vectorize_database_hash(database)
    km = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, n_init=1,
                verbose=False)
    km.fit_transform(data_tfidf)
    km2 = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, n_init=1,
                verbose=False)
    km2.fit_transform(data_hash)
    print("-"*15 + "TFIDF" + "-"*15)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
    print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
    print("Accuracy: %0.3f" % metrics.accuracy_score(labels, km.labels_))
    print("Adjusted Rand-Index: %.3f"
          % metrics.adjusted_rand_score(labels, km.labels_))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(data_tfidf, labels, sample_size=1000))
    print("-" * 15 + "Hash" + "-" * 15)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels2, km2.labels_))
    print("Completeness: %0.3f" % metrics.completeness_score(labels2, km2.labels_))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels2, km2.labels_))
    print("Accuracy: %0.3f" % metrics.accuracy_score(labels2, km2.labels_))
    print("Adjusted Rand-Index: %.3f"
          % metrics.adjusted_rand_score(labels2, km2.labels_))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(data_tfidf, labels2, sample_size=1000))

def mlp_classification(database, labels, randomState):
    X_train, X_test, y_train, y_test = split_database(database, labels)

    y_train_product = encoding_labels(y_train[:, 0], ['0', 'Product'])
    y_train_store = encoding_labels(y_train[:, 1], ['0', 'Store'])
    y_test_product = encoding_labels(y_train[:, 0], ['0', 'Product'])
    y_test_store = encoding_labels(y_train[:, 1], ['0', 'Store'])

    clf_product = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (50, 2), random_state = randomState)
    clf_store = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (50, 2), random_state = randomState)

    clf_product.fit(X_train, y_train_product)
    clf_store.fit(X_train, y_train_store)

    return test(clf_product, clf_store, X_test, y_test)
    return metrics.accuracy_score(y_test, Z)

def som_classificarion(X_train, y_train, X_test, y_test):
    som = MiniSom(6, 6, len(X_train[0]), learning_rate=0.1, sigma=0.3)
    som.train_random(X_train, 100)  # trains the SOM with 100 iterations
    from pylab import plot,axis,show,pcolor,colorbar,bone
    bone()
    pcolor(som.distance_map().T) # plotting the distance map as background
    colorbar()
    target = y_train
    t = np.zeros(len(target),dtype=int)
    t[target == 'Product'] = 0
    t[target == '0'] = 1
    # use different colors and markers for each label
    #markers = ['o','s','D', 'd', '|', '<', '>']
    #colors = ['r','g','b','c', 'm', 'y', 'w']
    markers = ['o', 's']
    colors = ['r', 'g']
    for cnt,xx in enumerate(X_train):
        w = som.winner(xx) # getting the winner
        # palce a marker on the winning position for the sample xx
        plot(w[0]+.5,w[1]+.5,markers[t[cnt]],markerfacecolor='None',
            markeredgecolor=colors[t[cnt]],markersize=12,markeredgewidth=2)
    axis([0,som.weights.shape[0],0,som.weights.shape[1]])
    show() # show the figure
    print som.winner(X_test[0])
    print y_test[0]

def logistic_regression_classification(X_train, y_train, X_test, y_test):
    clf = LogisticRegression(C=1e5)
    clf.fit(X_train, y_train)
    Z = clf.predict(X_test)
    return metrics.accuracy_score(y_test, Z)

def logistic_regression_classification(database, labels):
    X_train, X_test, y_train, y_test = split_database(database, labels)

    y_train_product = encoding_labels(y_train[:, 0], ['0', 'Product'])
    y_train_store = encoding_labels(y_train[:, 1], ['0', 'Store'])
    y_test_product = encoding_labels(y_train[:, 0], ['0', 'Product'])
    y_test_store = encoding_labels(y_train[:, 1], ['0', 'Store'])

    clf_product = LogisticRegression(C=1e5)
    clf_store = LogisticRegression(C=1e5)

    clf_product.fit(X_train, y_train_product)
    clf_store.fit(X_train, y_train_store)

    return test(clf_product, clf_store, X_test, y_test)



def test(clf1, clf2, patters, labels):
    labels_1 = encoding_labels(labels[:, 0], ['0', 'Product'])
    labels_2 = encoding_labels(labels[:, 1], ['0', 'Store'])
    acertos = 0
    for i in range(len(patters)):
        a = [labels_1[i], labels_2[i]]
        b = predict(clf1, clf2, patters[i])
        if np.array(b).all() == np.array(a).all():
            acertos += 1
    return float(float(acertos)/float(len(patters)))

def predict(clf1, clf2, patter):
    is_product = clf1.predict(patter)
    is_loja = clf2.predict(patter)

    return [is_product[0], is_loja[0]]

def svm_classification(X_train, y_train, X_test, y_test, kernel = 'linear'):
    clf = SVC(kernel=kernel)
    clf.fit(X_train, y_train)
    Z = clf.predict(X_test)
    return metrics.accuracy_score(y_test, Z)

