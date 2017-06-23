# -*- coding: utf-8 -*-
from Util import *
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
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
        self.svm = LogisticRegression()
        self.svm2 = LogisticRegression()
        # inicializando SVM


    def training_models(self, train_dataset, labels_for_train_dataset_1, labels_for_train_dataset_2):
        labels_1 = encoding_labels(labels_for_train_dataset_1, ['0', 'Product'])
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
        labels_1 = encoding_labels(labels[:, 0], ['0', 'Product'])
        labels_2 = encoding_labels(labels[:, 1], ['0', 'Store'])
        acertos = 0
        for i in range(len(patters)):
            a = [labels_1[i], labels_2[i]]
            b = self.predict(patters[i])
            if np.array(b).all() == np.array(a).all():
                acertos += 1
        return float(float(acertos)/float(len(patters)))


    def save_ensemble(self):
        joblib.dump(self.svm, 'ClassifierWeigths/svm-02.pkl')
        joblib.dump(self.svm2, 'ClassifierWeigths/svm2-02.pkl')



if __name__ == "__main__":
    database, labels, vectorizer = load_database()
    results = []
    melhor = 0
    #print np.logspace(-5, 3, 5)
    for i in range(30):
        X_train, X_test, y_train, y_test = split_database(database, labels)

        ensemble = Ensemble()
        ensemble.training_models(X_train, y_train[:, 0], y_train[:, 1])
        result = ensemble.test(X_test, y_test)
        results.append(result)


        if len(results) == 0:
            ensemble.save_ensemble()
            melhor = result
        elif len(results) > 0 and result > melhor:
            ensemble.save_ensemble()
            melhor = result


    print np.mean(np.array(results))
    print np.std(np.array(results))
    print melhor
    joblib.dump(vectorizer, 'Data/vectorizer.pkl')

    a = vectorizer.transform([tokenize("Ótimo produto. Boa durabilidade e chegou rapidamente.")])
    a = a.todense()
    print str(ensemble.predict(a)) + " " + "[1, 1]"
    a = vectorizer.transform([tokenize("Ótimo produto. Boa durabilidade e a entrega me surpreendeu positivamente.")])
    a = a.todense()
    print str(ensemble.predict(a)) + " " + "[1, 1]"
    a = vectorizer.transform([tokenize("A bisnaga que eu recebi não veio totalmente cheia.")])
    a = a.todense()
    print str(ensemble.predict(a)) + " " + "[1, 0]"
    a = vectorizer.transform([tokenize("Não usei ainda a amostra que recebi")])
    a = a.todense()
    print str(ensemble.predict(a)) + " " + "[0, 0]"
    a = vectorizer.transform([tokenize("Não tive nenhum problema com a compra pela internet")])
    a = a.todense()
    print str(ensemble.predict(a)) + " " + "[0, 1]"
    a = vectorizer.transform([tokenize("A chuteira é boa, mas o preço é salgado.")])
    a = a.todense()
    print str(ensemble.predict(a)) + " " + "[1, 1]"
    a = vectorizer.transform([tokenize("A chuteira é boa mas a cor é horrorosa. Fui muito mal atendido.")])
    a = a.todense()
    print str(ensemble.predict(a)) + " " + "[1, 1]"
    a = vectorizer.transform([tokenize("site muito complicado, não consegui realizar minha compra")])
    a = a.todense()
    print str(ensemble.predict(a)) + " " + "[0, 1]"
    a = vectorizer.transform([tokenize("atrasou")])
    a = a.todense()
    print str(ensemble.predict(a)) + " " + "[0, 1]"
    a = vectorizer.transform([tokenize("Ótima essa chapinha, mas o atendimento foi péssimo")]) # [1, 1]
    a = a.todense()
    print str(ensemble.predict(a)) + " " + "[1, 1]"
    a = vectorizer.transform([tokenize("gostei um pouco mas depois de dois dias essa coisa parou de funcionar")])  # [1, 0]
    a = a.todense()
    print str(ensemble.predict(a)) + " " + "[1, 0]"
    a = vectorizer.transform(
        [tokenize("curti muito meu leptop novo!")])  # [1, 0]
    a = a.todense()
    print str(ensemble.predict(a)) + " " + "[1, 0]"
    a = vectorizer.transform(
        [tokenize("Curti muito essa carteira, mas fui mal atendido.")])  # [1, 0]
    a = a.todense()
    print str(ensemble.predict(a)) + " " + "[1, 1]"
    a = vectorizer.transform(
        [tokenize("Sempre quis ter um desses e adorei. Só não gostei de ter de esperar mais de 20 dias.")])  # [1, 0]
    a = a.todense()
    print str(ensemble.predict(a)) + " " + "[1, 1]"
    a = vectorizer.transform(
        [tokenize("Sei que a culpa é dos Correios, mas passou a festa e não consegui usar a roupa. Chateada.")])  # [1, 0]
    a = a.todense()
    print str(ensemble.predict(a)) + " " + "[1, 1]"
    a = vectorizer.transform(
        [tokenize(
            "A tiara é linda, mas não gostei do atendimento.")])  # [1, 0]
    a = a.todense()
    print str(ensemble.predict(a)) + " " + "[1, 1]"
    a = vectorizer.transform(
        [tokenize(
            "Essa furadeira é ótima, mas não compro mais com vocês.")])  # [1, 0]
    a = a.todense()
    print str(ensemble.predict(a)) + " " + "[1, 1]"
    a = vectorizer.transform(
        [tokenize(
            "Realmente muito difícil pra achar os produtos. Não curti.")])  # [1, 0]
    a = a.todense()
    print str(ensemble.predict(a)) + " " + "[1, 1]"



