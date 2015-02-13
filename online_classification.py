
import numpy as np
import math
import sparse_linalg as la
# from numpy.linalg import inv
from scipy.stats import norm
# from test_spam_classification import read_multiple_days
# from sklearn import datasets
# import scipy.sparse as sp
import warnings
import pprint
import csv
# warnings.filterwarnings("error")

aws = True
if aws:
	# sys.path.append(os.path.abspath("/home/ubuntu/liblinear-1.96/python"))
	base_path = "/home/ubuntu/url_svmlight"
else:
	# sys.path.append(os.path.abspath("/Users/crankshaw/code/amplab/model-serving/spam-urls/liblinear-1.96/python"))
	base_path = "/Users/crankshaw/code/amplab/model-serving/data/spam-urls/url_svmlight"
# base_path = "/Users/crankshaw/code/amplab/model-serving/data/spam-urls/url_svmlight"
a_const = 1.0
eta_const = 0.9
phi_const = norm.ppf(eta_const)
psi_const = 1 + phi_const**2 / 2.0
xi_const = 1 + phi_const**2


# algorithm from "Exact Convex Confidence-Weighted Learning"
# http://nips.cc/Conferences/2008/Program/event.php?ID=1222

def update(x_i, true_y_i, mu_i, sigma_i, i):
    v_i = compute_v_i(x_i, sigma_i)

    # print v_i
    if v_i < 10**(-20):
	print "uh oh"
	return (mu_i, sigma_i)
    if math.isnan(v_i) or v_i == 0:
	print "iter %d: bad v_i % f" % (i, v_i)
    m_i = compute_m_i(x_i, true_y_i, mu_i)
    alpha_i = compute_alpha_i(v_i, xi_const, m_i, psi_const, phi_const)
    u_i = compute_u_i(alpha_i, v_i, phi_const)
    beta_i = compute_beta_i(alpha_i, phi_const, u_i, v_i)

    mu_prime = update_mu(mu_i, alpha_i, true_y_i, sigma_i, x_i)
    sigma_prime = diag_sigma_update(sigma_i, alpha_i, phi_const, u_i, x_i)
    return (mu_prime, sigma_prime)

def update_mu(mu_i, alpha_i, true_y_i, sigma_i, x_i):

    prod = la.sd_multiply(x_i, sigma_i)
    mu_prime = la.sd_sum(la.s_scalar_mult(prod, alpha_i * true_y_i), mu_i)
    # mu_prime = mu_i + alpha_i * true_y_i * la.sd_multiply(x_i, sigma_i)
    return mu_prime

def make_prediction(x_i, mu_i):
    return np.sign(la.sd_dot(x_i, mu_i))

def compute_v_i(x_i, sigma_i):
    prod = la.sd_multiply(x_i, sigma_i)
    v_i = la.ss_dot(prod, x_i)
    return v_i

def compute_m_i(x_i, true_y_i, mu_i):
    return true_y_i * la.sd_dot(x_i, mu_i)

def compute_alpha_i(v_i, xi_const, m_i, psi_const, phi_const):
    potential_alpha = 1.0 / (v_i*xi_const) * (-m_i*psi_const + np.sqrt(m_i**2 * phi_const**4 /4 + v_i*phi_const**2*xi_const))
    return max(0, potential_alpha)

def compute_u_i(alpha_i, v_i, phi_const):
    return (-alpha_i*v_i*phi_const + np.sqrt((alpha_i * v_i * phi_const)**2 + 4*v_i))**2 / 4

def compute_beta_i(alpha_i, phi_const, u_i, v_i):
    return alpha_i * phi_const / (np.sqrt(u_i) + v_i*alpha_i*phi_const)

def diag_sigma_update(sigma_i, alpha_i, phi_const, u_i, x_i):
    x_squared = la.sparse_unary_op(x_i, lambda x: x**2)
    sigma_inv = la.dense_unary_op(sigma_i, lambda x: 1.0 / x)
    x_squared = la.s_scalar_mult(x_squared, alpha_i * phi_const / np.sqrt(u_i))
    new_sigma_inv = la.sd_sum(x_squared, sigma_inv)
    return la.dense_unary_op(new_sigma_inv, lambda x: 1.0 / x)

def load_days(start, end):
    paths = []
    for day in range(start, end):
	path = "%s/Day%d.svm" % (base_path, day)
	paths.append(path)
    return la.read_svmlight(paths)

def load_day(day):
    path = "%s/Day%d.svm" % (base_path, day)
    return la.read_svmlight([path])


def init_model():
    xs, ys, max_dim = load_days(0, 1)
    mu = np.zeros(max_dim)
    sigma = np.ones(max_dim)
    for i in range(len(ys)):
	mu, sigma = update(xs[i], ys[i], mu, sigma, 0)
    print "Model initialized"
    return mu, sigma

def process_day(day, mu, sigma):
    num_false_pos = 0
    num_false_neg = 0
    xs, ys, max_dim = load_day(day)
    total = len(ys)
    # extend arrays to account for new features
    if max_dim > len(mu):
	app0 = np.zeros(max_dim - len(mu))
	app1 = np.ones(max_dim - len(sigma))
	mu = np.append(mu, app0)
	sigma = np.append(sigma, app1)
    
    for i in range(len(ys)):
	x = xs[i]
	y = ys[i]
	pred = make_prediction(x, mu)
	diff = y - pred
	total += 1
	if diff == -2:
		num_false_pos += 1
	if diff == 2:
		num_false_neg += 1
	mu, sigma = update(x, y, mu, sigma, i)
	if i % 1000 == 0:
	    print "Processed %d on day %d" % (i, day)
    return (mu, sigma, num_false_pos, num_false_neg, total)




def main():
    results = []
    error_rates = []
    cum_total = 0
    cum_false_pos = 0
    cum_false_neg = 0
    mu, sigma = init_model()

    for day in range(1, 30):
	mu, sigma, fp, fn, total = process_day(day, mu, sigma)
	cum_total += total
	cum_false_pos += fp
	cum_false_neg += fn
	results.append((day, cum_total, cum_false_pos, cum_false_neg))
	error_rates.append((day,
			    100.0*(cum_false_pos + cum_false_neg)/float(cum_total),
			    100.0*cum_false_pos/float(cum_total),
			    100.0*cum_false_neg/float(cum_total)))

	print "%d: %f%%, %f%%, %f%%" % (day,
				  100.0*(cum_false_pos + cum_false_neg)/float(cum_total),
				  100.0*cum_false_pos/float(cum_total), 
				  100.0*cum_false_neg/float(cum_total))

	
	
    with open("results/online_retrain_daily_results.csv", "wb") as out:
	file_writer = csv.writer(out)
	file_writer.writerow(['day', 'total_err', 'false_pos', 'false_neg'])
	for row in error_rates:
	    file_writer.writerow(row)

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(results)




if __name__=="__main__":
    main()




