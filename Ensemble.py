# -*- coding: utf-8 -*-
from Util import *
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from sklearn.svm import SVC
import ignore_warnings
from Util import timing


class Ensemble:

    def __init__(self):

        print "Starting Classifiers..."
        self.vectorizer = self.load_vec()
       
        self.clf = LogisticRegression(penalty='l2', C=1.0, max_iter=100, solver='liblinear')
        #self.clf2 = LogisticRegression(C=1.0, max_iter=100, solver='sag', penalty='l2')
        self.clf2 = SVC(kernel='linear', C=1000.0, probability=True, decision_function_shape='ovo', tol=0.001, shrinking=True, gamma=1.0)
        # Rambo Bêbado
        #self.clf = SGDClassifier(average=True, penalty='l2', loss='hinge', alpha=0.0001, epsilon=0.5, l1_ratio=0.01)
        #self.clf2 = SGDClassifier(average=True, penalty='l2', loss='log', alpha=0.0001, epsilon=0.43, l1_ratio=0.01)

    def training_models(self, train_dataset, labels_for_train_dataset_1, labels_for_train_dataset_2):
        print "Training Classifiers..."
        labels_1 = encoding_labels(labels_for_train_dataset_1, ['0', 'Product'])
        labels_2 = encoding_labels(labels_for_train_dataset_2, ['0', 'Store'])

        #print "Training MLP..."
        #self.mlp.fit(train_dataset, labels[:,0])
        self.clf2.fit(train_dataset, labels_2)
        self.clf.fit(train_dataset, labels_1)
        #print "Training MLP..."
        #self.clf.fit(train_dataset, labels_2)

    @timing
    def predict(self, patter):
        is_product = self.clf.predict(patter)
        is_loja = self.clf2.predict(patter)

        return [is_product[0], is_loja[0]]

    def test(self, patters, labels):
        labels_1 = encoding_labels(labels[:, 0], ['0', 'Product'])
        labels_2 = encoding_labels(labels[:, 1], ['0', 'Store'])
        acertos = 0
        for i in range(len(patters)):
            a = [labels_1[i], labels_2[i]]
            b = self.predict(patters[i])
            if np.array(b).all() == np.array(a).all():
                acertos += 1
        return float(float(acertos)/float(len(patters)))

    def load_vec(self):
        return joblib.load('Data/vectorizer.pkl')

    def save_ensemble(self):
        joblib.dump(self.clf, 'ClassifierWeigths/svm-02.pkl')
        joblib.dump(self.clf2, 'ClassifierWeigths/svm2-02.pkl')

    def testinho(self, patterns, labels, verbose=0):
        
        self.clf = joblib.load('ClassifierWeigths/svm-02.pkl')
        self.clf2 = joblib.load('ClassifierWeigths/svm2-02.pkl')

        errors = 0

        for i in range(len(patterns)):
            a = self.vectorizer.transform([tokenize(patterns[i])])
            a = a.todense()
            classification = ensemble.predict(a)
            if labels[i][1] == 'Store' and classification[1] != 1:
                errors += 1
                if verbose ==1:
                    print "{0} ------------- foi classificado como: {1} mas deveria ser: {2}".format(patterns[i], classification, labels[i])

        print "Stability = {0}".format(1 - float(errors)/len(patterns))
if __name__ == "__main__":
    #database, labels, vectorizer = load_database()
    #results = []
    #melhor = 0
    ensemble = Ensemble()
    #ensemble.training_models(np.array(database), labels[:, 0], labels[:, 1])
    #ensemble.save_ensemble()

    #joblib.dump(vectorizer, 'Data/vectorizer.pkl')

    patterns, labels = read_file("Data/database.csv")

    ensemble.testinho(patterns, labels, verbose=1)