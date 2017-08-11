from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.externals import joblib
import Util

class OneClass:

    def __init__(self):
        print ("Inicializando Classificadores")
        self.product_classifier = LogisticRegression(penalty='l2', C=1.0, max_iter=100, solver='liblinear')
        self.one_class_classification = svm.OneClassSVM(nu=0.1, kernel="linear", gamma=0.1)

    def training_models(self):
        print "Training Classifiers..."
        product_data, labels, product_vec = Util.load_database()
        store_data, store_vec = Util.load_store_database()
        labels = Util.encoding_labels(labels[:, 0], ['0', 'Product'])

        self.save_vectorizers(product_vec, store_vec)

        self.product_classifier.fit(product_data, labels)
        self.one_class_classification.fit(store_data)

    def save_vectorizers(self, product_vectorizer, store_vectorizer):
        joblib.dump(product_vectorizer, 'Quixada/SaveVec/product_vectorizer.pkl')
        joblib.dump(store_vectorizer, 'Quixada/SaveVec/store_vectorizer.pkl')

    def predict(self, patter):
        is_product = self.product_classifier.predict(patter)
        is_loja = self.one_class_classification.predict(patter)

        return [is_product[0], is_loja[0]]

    def save_ensemble(self):
        joblib.dump(self.product_classifier, 'ClassifierWeigths/svm-02.pkl')
        joblib.dump(self.one_class_classification, 'ClassifierWeigths/svm2-02.pkl')


if __name__ == '__main__':
    classifier = OneClass()
    classifier.training_models()
    classifier.save_ensemble()






