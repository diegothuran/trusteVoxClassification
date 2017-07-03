from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix
import numpy as np

import Util
import ignore_warnings

def main():
	results = {}
	training_samples, training_classes, vectorizer = Util.load_database()
	training_classes = training_classes[:, 0]
	# ==================================================
	# Support vector machine classifier
	# ==================================================

	# SVM parameters
	params = {
	'penalty': ['l1', 'l2'],
	'C': np.array([0.001, 0.01, 0.1, 1, 10, 100, 1000]),
	'max_iter': np.array([100, 200, 300]),
	'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag']}

	for param in params.keys():
		print("========================================")
		print("Testing values for '"+param+"'")
		print("========================================")
		classifier = LogisticRegression()
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