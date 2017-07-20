from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.svm import SVC

import Util
import ignore_warnings

def main():
	results = {}
	training_samples, training_classes, vectorizer = Util.load_database()
	training_classes = training_classes[:, 1]
	# ==================================================
	# Support vector machine classifier
	# ==================================================

	# SVM parameters
	params = {
	'gamma': np.array([0.001, 0.01, 0.1, 1, 10, 100, 1000]),
	'C': np.array([0.001, 0.01, 0.1, 1, 10, 100, 1000]),
	'decision_function_shape': ['ovo', 'ovr', None],
	'tol' : np.array([1e-3, 1e-5, 1e-2]),
	'shrinking': [True, False],
	'probability': [True, False],
	'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}

	for param in params.keys():
		print("========================================")
		print("Testing values for '"+param+"'")
		print("========================================")
		classifier = SVC()
		grid = GridSearchCV(estimator=classifier, #verbose=10,
			param_grid={param:params[param]})
		grid.fit(training_samples, training_classes)
		print("> Best score: "+str(grid.best_score_))
		print("> Best param: "+str(getattr(grid.best_estimator_,param)))
		results[param] = str(getattr(grid.best_estimator_,param))

	print("\n\n")
	for arg in results.keys():
		print("Best value for '"+arg+"': "+results[arg])

main()