from Util import vectorize_database_tfidf, vectorize_database_hash
import cPickle
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import metrics
import random
import numpy as np
from minisom import MiniSom
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

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

def mlp_classification(X_train, y_train, X_test, y_test):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (5, 2), random_state = 1)

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

def svm_classification(X_train, y_train, X_test, y_test):
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    Z = clf.predict(X_test)
    return metrics.accuracy_score(y_test, Z)