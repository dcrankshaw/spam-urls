

import os
import sys
sys.path.append(os.path.abspath("/home/ubuntu/liblinear-1.96/python"))
import liblinearutil as ll
import pprint
import csv

# base_path = "/Users/crankshaw/code/amplab/model-serving/data/spam-urls/url_svmlight"
base_path = "/home/ubuntu/url_svmlight"
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
	# print "loaded data from days %d to %d" % (start_day, end_day)
	return (all_y, all_x)

def train_svm(start_day, end_day):
	all_y, all_x = read_multiple_days(start_day, end_day)
	model = ll.train(all_y, all_x, "-c %d -s 0 -q" % cost)
	return model

def predict_with_svm(model, start_day, end_day):
	all_y, all_x = read_multiple_days(start_day, end_day)
	labels, acc, values = ll.predict(all_y, all_x, model)
	num_false_pos = 0
	num_false_neg = 0
	total = len(all_y)
	for gt, pred in zip(all_y, labels):
		diff = gt - pred
		if diff == -2:
			num_false_pos += 1
		if diff == 2:
			num_false_neg += 1
	return (total, num_false_pos, num_false_neg)

	# print "ACC: %f, MSE: %f, SCC: %f" % acc


if __name__ == "__main__":
	results = []
	error_rates = []
	cum_total = 0
	cum_false_pos = 0
	cum_false_neg = 0
	for end in range(20, 50):
		model = train_svm(0, end)
		t, fp, fn = predict_with_svm(model, end + 1, end + 2)
		cum_total += t
		cum_false_pos += fp
		cum_false_neg += fn
		results.append((end, cum_total, cum_false_pos, cum_false_neg))
		error_rates.append((end,
				    100.0*(cum_false_pos + cum_false_neg)/float(cum_total),
				    100.0*cum_false_pos/float(cum_total),
				    100.0*cum_false_neg/float(cum_total)))

		print "%d: %f%%, %f%%, %f%%" % (end,
					  100.0*(cum_false_pos + cum_false_neg)/float(cum_total),
					  100.0*cum_false_pos/float(cum_total), 
					  100.0*cum_false_neg/float(cum_total))
	pp = pprint.PrettyPrinter(indent=4)
	pp.pprint(results)
	with open("retrain_daily_results.csv", "wb") as out:
		file_writer = csv.writer(out)
		file_writer.writerow(['day', 'total_err', 'false_pos', 'false_neg'])
		for row in error_rates:
			file_writer.writerow(row)


	# model = train_svm(0, 50)
	# print "Model trained"
	# predict_with_svm(model, 50, 100)




	
