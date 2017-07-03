from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix
import numpy as np

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
	'penalty': ['l1', 'l2'],
	'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
    'alpha': np.array([0.0001, 0.001, 0.01, 0.1, 0.5]),
	'l1_ratio': np.array([0.15, 0.10, 0.01, 0.5, 0.6, 0.65, 0.7, 0.71, 0.72]),
	'epsilon': np.array([0.1, 0.2, 0.3, 0.35, 0.37, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.5])}

	for param in params.keys():
		print("========================================")
		print("Testing values for '"+param+"'")
		print("========================================")
		classifier = SGDClassifier(average=True)
		grid = GridSearchCV(estimator=classifier, cv=10,
			param_grid={param:params[param]})
		grid.fit(training_samples, training_classes)
		print("> Best score: "+str(grid.best_score_))
		print("> Best param: "+str(getattr(grid.best_estimator_,param)))
		results[param] = str(getattr(grid.best_estimator_,param))

	print("\n\n")
	for arg in results.keys():
		print("Best value for '"+arg+"': "+results[arg])

main()