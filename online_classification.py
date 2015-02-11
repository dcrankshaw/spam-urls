
import numpy as np
import math
import sparse_linalg as la
# from numpy.linalg import inv
from scipy.stats import norm
# from test_spam_classification import read_multiple_days
# from sklearn import datasets
# import scipy.sparse as sp
import warnings
# warnings.filterwarnings("error")


base_path = "/Users/crankshaw/code/amplab/model-serving/data/spam-urls/url_svmlight"
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

    # x_nonzeros = sp.find(x_i)
    # x_squared = sp.coo_matrix((x_nonzeros[2]**2, (x_nonzeros[0], x_nonzeros[0])), shape=sigma_i.shape).todia()
    # sig_inv = diag_inverse(sigma_i)
    # new_sig_inv = sig_inv + alpha_i * phi_const*(1.0/np.sqrt(u_i))*x_squared
    # return diag_inverse(new_sig_inv)

# def diag_inverse(m):
#     m.data = 1.0 / m.data
#     return m
#     # d = sp.find(m)
#     # new_vals = 1.0 / d[2]
#     # return sp.coo_matrix((new_vals, (d[0], d[1])), shape=m.shape).todia()


def load_days(start, end):
    paths = []
    for day in range(start, end):
	path = "%s/Day%d.svm" % (base_path, day)
	paths.append(path)
    return la.read_svmlight(paths)


def main():
    xs, ys, max_dim = load_days(0, 1)
    mu = np.zeros(max_dim)
    sigma = np.ones(max_dim)
    num_false_pos = 0
    num_false_neg = 0
    num_wrong = 0
    total = 0
    num_examples = len(ys)
    mu, sigma = update(xs[0], ys[0], mu, sigma, 0)
    for i in range(1, num_examples):
	x = xs[i]
	y = ys[i]
	pred = make_prediction(x, mu)
	diff = y - pred
	total += 1
	if diff != 0:
	    num_wrong += 1
	if diff == -2:
		num_false_pos += 1
	if diff == 2:
		num_false_neg += 1

	mu, sigma = update(x, y, mu, sigma, i)
	if i % 500 == 0:
	    print "Processed %d" % i


    # for d in all_days:
	# for i in range(len(d[1])):
	#     # x = d[0][i].toarray()[0] # have to extract 1st row because it makes a 2d array
	#     x = d[0][i].transpose()
	#     y = d[1][i]
	#     pred = make_prediction(x, mu)
	#     total += 1
	#     if y != pred:
	# 	num_wrong += 1
	#     mu, current_sigma = update_rule(x, y, mu, current_sigma, i)
	#     if total % 10 == 0:
	# 	# print "Processed %d examples" % total
	# 	print "Total: %d, wrong: %d" % (total, num_wrong)
	# 	break
	#
	# # print mu

    print "Num wrong: %d" % num_wrong
    print "FP: %d, FN: %d, combined: %d" % (num_false_pos, num_false_neg, num_false_pos + num_false_neg)
    print "Total: %d" % total






if __name__=="__main__":
    main()




