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
        #self.svm = LogisticRegression(C=1e5)
        #self.svm2 = LogisticRegression(C=1e5)
        self.svm = LogisticRegression(penalty='l2', C=1.0, max_iter=100, solver='newton-cg')
        self.svm2 = LogisticRegression(C=1.0, max_iter=100, solver='liblinear', penalty='l2')
        # inicializando SVM
        #self.svm = SGDClassifier(average=True, penalty='l2', loss='hinge', alpha=0.0001, epsilon=0.5, l1_ratio=0.01)
        #self.svm2 = SGDClassifier(average=True, penalty='l2', loss='log', alpha=0.0001, epsilon=0.43, l1_ratio=0.01)

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

    def testinho(self, patterns=[], labels=[]):
        for i in range(len(patterns)):
            a = vectorizer.transform([tokenize(patterns[i])])
            a = a.todense()
            classification = ensemble.predict(a)
            if labels[i] == classification:
                print patterns[i] + " " + str(classification) + " " + str(labels[i]) + " ="
            else:
                print patterns[i] + " " +str(classification) + " " + str(labels[i]) + " !="




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
    patterns = ["Ótimo produto. Boa durabilidade e chegou rapidamente.", "Ótimo produto. Boa durabilidade e a entrega me surpreendeu positivamente.",
                "A bisnaga que eu recebi não veio totalmente cheia.", "Não usei ainda a amostra que recebi", "Não tive nenhum problema com a compra pela internet",
                "A chuteira é boa, mas o preço é salgado.", "A chuteira é boa mas a cor é horrorosa. Fui muito mal atendido.", "site muito complicado, não consegui realizar minha compra",
                "Ótima essa chapinha, mas o atendimento foi péssimo", "gostei um pouco mas depois de dois dias essa coisa parou de funcionar", "curti muito meu leptop novo!",
                "Curti muito essa carteira, mas fui mal atendido.", "Sempre quis ter um desses e adorei. Só não gostei de ter de esperar mais de 20 dias.",
                "Sei que a culpa é dos Correios, mas passou a festa e não consegui usar a roupa. Chateada.", "A tiara é linda, mas não gostei do atendimento.",
                "Essa furadeira é ótima, mas não compro mais com vocês.", "Realmente muito difícil pra achar os produtos. Não curti.",
                "Não achei legal ter de esperar quase 1 mês.", "Não achei legal ter de esperar quase 1 mês.", "esperar um mês para receber foi a picada",
                "Produto ótimo, entrega tudo que promete!", "Quero meu dinheiro de volta", "O produto é ótimo. E adorei o preço",
                "Chegou direitinho, mas a pessoa que está usando não gostou! disse que o housekeeper não funciona.", "show de bola", "vale muito a pena",
                "A confiabilidade dessa placa é uma coisa fora de sério", "até o momento não tenho nada do que reclamar", "momento", "até agora", "Produto não funcionou corretamente", "Demora muito tempo o preparo.",
                "Não suportou o peso, e entortou as varetas.", "O material do produto é uma porcaria, e a embalagem uma merd@.",
                "Um produto comum e para a compra desse item não vejo necessidade de muita informação. A forma como está exposto já resolve."]

    labels = [[1, 1], [1, 1], [1, 0], [0, 0], [0, 1], [1, 1], [1, 1], [0, 1], [0, 1], [1, 1], [1, 0], [1, 0],
              [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [0, 1], [0, 1], [0, 1], [0, 1], [1, 1], [1, 1], [1, 1], [1, 0], [1, 0],
              [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]]
    ensemble.testinho(patterns, labels)