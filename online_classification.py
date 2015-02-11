
import numpy as np
import math
# from numpy.linalg import inv
from scipy.stats import norm
# from test_spam_classification import read_multiple_days
from sklearn import datasets
from sklearn.linear_model import LogisticRegression as LR
import scipy.sparse as sp
import warnings
warnings.filterwarnings("error")


base_path = "/Users/crankshaw/code/amplab/model-serving/data/spam-urls/url_svmlight"
dims = 20
a_const = 1.0
eta_const = 0.9
# mu_0 = np.zeros(dims)
mu_0 = np.zeros(dims)
sigma_0 = a_const*np.ones(dims)
phi_const = norm.ppf(eta_const)
psi_const = 1 + phi_const**2 / 2.0
xi_const = 1 + phi_const**2


# algorithm from "Exact Convex Confidence-Weighted Learning"
# http://nips.cc/Conferences/2008/Program/event.php?ID=1222

def update_rule(x_i, true_y_i, mu_i, sigma_i, i):
    bad = False
    v_i = compute_v_i(x_i, sigma_i)

    # print v_i
    if v_i < 10**(-20):
	return (mu_i, sigma_i)
    if math.isnan(v_i) or v_i == 0:
	print "iter %d: bad v_i % f" % (i, v_i)
	bad = True
    m_i = compute_m_i(x_i, true_y_i, mu_i)
    alpha_i = compute_alpha_i(v_i, xi_const, m_i, psi_const, phi_const)
    u_i = compute_u_i(alpha_i, v_i, phi_const)
    if math.isnan(u_i) or u_i == 0:
	print "iter %d: bad u_i: %f" % (i, u_i)
	bad = True
    beta_i = compute_beta_i(alpha_i, phi_const, u_i, v_i)

    mu_prime = mu_i + alpha_i * true_y_i * sigma_i.dot(x_i).transpose()
    sigma_prime = diag_sigma_update(sigma_i, alpha_i, phi_const, u_i, x_i)
    return (mu_prime, sigma_prime)


def make_prediction(x_i, mu_i):
    res = sp.find(mu_i.dot(x_i))[2]
    if len(res) == 0:
	return 0
    else:
	return np.sign(res[0])


def compute_v_i(x_i, sigma_i):
    v_i = x_i.transpose().dot(sigma_i).dot(x_i)
    return sp.find(v_i)[2][0]

def compute_m_i(x_i, true_y_i, mu_i):
    m = sp.find(true_y_i*(mu_i.dot(x_i)))[2]
    if len(m) == 0:
	return 0.0
    else:
	return m[0]

def compute_alpha_i(v_i, xi_const, m_i, psi_const, phi_const):
    potential_alpha = 1.0 / (v_i*xi_const) * (-m_i*psi_const + np.sqrt(m_i**2 * phi_const**4 /4 + v_i*phi_const**2*xi_const))
    return max(0, potential_alpha)

def compute_u_i(alpha_i, v_i, phi_const):
    return (-alpha_i*v_i*phi_const + np.sqrt((alpha_i * v_i * phi_const)**2 + 4*v_i))**2 / 4

def compute_beta_i(alpha_i, phi_const, u_i, v_i):
    return alpha_i * phi_const / (np.sqrt(u_i) + v_i*alpha_i*phi_const)

def diag_sigma_update(sigma_i, alpha_i, phi_const, u_i, x_i):
    x_nonzeros = sp.find(x_i)
    x_squared = sp.dia_matrix(
	    sp.coo_matrix((x_nonzeros[2]**2, (x_nonzeros[0], x_nonzeros[0])), shape=sigma_i.shape))
    sig_inv = diag_inverse(sigma_i)
    return diag_inverse(sig_inv + alpha_i * phi_const*(1.0/np.sqrt(u_i))*x_squared)

def diag_inverse(m):
    # d = np.diagonal(m).copy()
    d = sp.find(m)
    new_vals = [1.0 / j for j in d[2]]
    return sp.dia_matrix(sp.coo_matrix((new_vals, (d[0], d[1])), shape=m.shape))
    # d = m.copy()
    # for i in range(len(d)):
	# d[i] = 1.0/d[i]	
    # return d


def load_days(start, end):
    max_dim = 0
    days = []
    for day in range(start, end):
	path = "%s/Day%d.svm" % (base_path, day)
	x, y = datasets.load_svmlight_file(path)
	max_dim = max(max_dim, x.shape[1])
	days.append((x, y))
    return (days, max_dim)


def main():
    # samples = 500
    # all_x, all_y = datasets.make_classification(n_samples=samples, n_features=dims)
    # all_x, all_y = load_days(0, 2)
    # for i in range(len(all_y)):
	# if all_y[i] == 0:
	#     all_y[i] = -1

    all_days, max_dim = load_days(0, 1)
    current_mu = sp.csr_matrix((1, max_dim))
    current_sigma = sp.eye(max_dim)
    num_wrong = 0
    total = 0
    for d in all_days:
	for i in range(len(d[1])):
	    # x = d[0][i].toarray()[0] # have to extract 1st row because it makes a 2d array
	    x = d[0][i].transpose()
	    y = d[1][i]
	    pred = make_prediction(x, current_mu)
	    total += 1
	    if y != pred:
		num_wrong += 1
	    current_mu, current_sigma = update_rule(x, y, current_mu, current_sigma, i)
	    if total % 10 == 0:
		# print "Processed %d examples" % total
		print "Total: %d, wrong: %d" % (total, num_wrong)
		# break
	
	# print current_mu

    print "Num wrong: %d" % num_wrong
    print "Total: %d" % total
    print "initial mu", mu_0
    print "final mu", current_mu





if __name__=="__main__":
    main()



