from Util import *
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
import pickle


class Ensemble:

    def __init__(self):
        # inicializando Classificadores
        # inicializando MLP
        '''
            print "Inicializando Regressao Logistica"
            self.regression = LogisticRegression(C=1e5)
            print "Inicializando Regressao Logistica"
            self.regression2 = LogisticRegression(C=1e5)
            print "Inicializando SVM"
            self.svm = SVC(kernel='linear')
        '''


        print "Inicializando MLP..."
        self.svm = SVC(kernel='linear')
        self.svm2 = SVC(kernel='linear')
        # inicializando SVM


    def training_models(self, train_dataset, labels_for_train_dataset_1, labels_for_train_dataset_2):
        labels_1 = encoding_labels(labels_for_train_dataset_1, ['0', 'Product', 'Pro', 'Prod'])
        labels_2 = encoding_labels(labels_for_train_dataset_2, ['0', 'Store'])

        #print "Training MLP..."
        #self.mlp.fit(train_dataset, labels[:,0])
        self.svm2.fit(train_dataset, labels_2)
        self.svm.fit(train_dataset, labels_1)
        #print "Training MLP..."
        #self.svm.fit(train_dataset, labels_2)

    def predict(self, patter):
        is_product = self.svm.predict(patter)
        is_loja = self.svm2.predict(patter)
        #polarity = self.mlp.predict(patter)

        return [is_product[0], is_loja[0]]

    def test(self, patters, labels):
        labels_1 = encoding_labels(labels[:, 0], ['0', 'Product', 'Pro', 'Prod'])
        labels_2 = encoding_labels(labels[:, 1], ['0', 'Store'])
        acertos = 0
        for i in range(len(patters)):
            a = [labels_1[i], labels_2[i]]
            b = self.predict(patters[i])
            temp = cmp(b, a)
            if temp == 0:
                acertos += 1
        return float(float(acertos)/float(len(patters)))

    def save_ensemble(self):
        joblib.dump(self.svm, 'ClassifierWeigths/svm-02.pkl')
        joblib.dump(self.svm2, 'ClassifierWeigths/svm2-02.pkl')



if __name__ == "__main__":
    database, labels = load_database()
    results = []
    #print np.logspace(-5, 3, 5)
    for i in range(100):
        X_train, X_test, y_train, y_test = split_database(database, labels)

        ensemble = Ensemble()
        ensemble.training_models(X_train, y_train[:, 0], y_train[:, 1])
        result = ensemble.test(X_test, y_test)
        print result
        results.append(result)
        if i > 0:
            if results[i] > results[i-1]:
                ensemble.save_ensemble()
        else:
            ensemble.save_ensemble()

    print np.mean(np.array(results))
    print np.std(np.array(results))


