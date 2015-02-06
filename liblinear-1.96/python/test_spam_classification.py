

import liblinearutil as ll
import os
import sys

base_path = "/Users/crankshaw/code/amplab/model-serving/data/spam-urls/url_svmlight"
cost = 100
model = 0

# -s type : set type of solver (default 1)
#   for multi-class classification
# 	 0 -- L2-regularized logistic regression (primal)
# 	 1 -- L2-regularized L2-loss support vector classification (dual)
# 	 2 -- L2-regularized L2-loss support vector classification (primal)
# 	 3 -- L2-regularized L1-loss support vector classification (dual)
# 	 4 -- support vector classification by Crammer and Singer
# 	 5 -- L1-regularized L2-loss support vector classification
# 	 6 -- L1-regularized logistic regression
# 	 7 -- L2-regularized logistic regression (dual)
#   for regression
# 	11 -- L2-regularized L2-loss support vector regression (primal)
# 	12 -- L2-regularized L2-loss support vector regression (dual)
# 	13 -- L2-regularized L1-loss support vector regression (dual)

def read_multiple_days(start_day, end_day):
	all_y = []
	all_x = []
	if start_day > end_day:
		return
	if start_day < 0:
		return
	if end_day > 120:
		return
	for day in range(start_day, end_day):
		path = "%s/Day%d.svm" % (base_path, day)
		(y, x) = ll.svm_read_problem(path)
		all_y.extend(y)
		all_x.extend(x)
	print "loaded data from days %d to %d" % (start_day, end_day)
	return (all_y, all_x)

def train_svm(start_day, end_day):
	all_y, all_x = read_multiple_days(start_day, end_day)
	model = ll.train(all_y, all_x, "-c %d -s 0 -1" % cost)
	return model

def predict_with_svm(model, start_day, end_day):
	all_y, all_x = read_multiple_days(start_day, end_day)
	labels, acc, values = ll.predict(all_y, all_x, model)
	print "ACC: %f, MSE: %f, SCC: %f" % acc


if __name__ == "__main__":
	model = train_svm(0, 50)
	print "Model trained"
	predict_with_svm(model, 50, 60)




	
