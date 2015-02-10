
import numpy as np
import math
# from numpy.linalg import inv
from scipy.stats import norm
from test_spam_classification import read_multiple_days
from sklearn import datasets

dims = 20
a_const = 1.0
eta_const = 0.9
# mu_0 = np.zeros(dims)
mu_0 = np.ones(dims)
sigma_0 = a_const*np.identity(dims)
phi_const = norm.ppf(eta_const)
psi_const = 1 + phi_const**2 / 2.0
xi_const = 1 + phi_const**2


# algorithm from "Exact Convex Confidence-Weighted Learning"
# http://nips.cc/Conferences/2008/Program/event.php?ID=1222

def update_rule(x_i, true_y_i, mu_i, sigma_i, i):
    bad = False
    pred_y_i = make_prediction(x_i, mu_i)
    v_i = compute_v_i(x_i, sigma_i)

    # print v_i
    if v_i < 10**(-12):
	return (mu_i, sigma_i)
    if math.isnan(v_i) or v_i == 0:
	print "iter %d: bad v_i % f" % (i, v_i)
	bad = True
    m_i = compute_m_i(x_i, true_y_i, mu_i)
    if math.isnan(m_i) or m_i == 0:
	print "iter %d: bad m_i" % i
	bad = True
    alpha_i = compute_alpha_i(v_i, xi_const, m_i, psi_const, phi_const)
    if alpha_i == 0:
	return (mu_i, sigma_i)
    if math.isnan(alpha_i):
	print "iter %d: bad alpha_i" % i
	bad = True
    u_i = compute_u_i(alpha_i, v_i, phi_const)
    if math.isnan(u_i) or u_i == 0:
	print "iter %d: bad u_i: %f" % (i, u_i)
	bad = True
    beta_i = compute_beta_i(alpha_i, phi_const, u_i, v_i)
    if math.isnan(beta_i):
	print "iter %d: bad beta_i" % i
	bad = True

    mu_prime = mu_i + (alpha_i * true_y_i * sigma_i.dot(x_i))
    sigma_prime = diag_sigma_update(sigma_i, alpha_i, phi_const, u_i, x_i)
    # print np.diagonal(sigma_prime)
    # if bad:
	# print mu_prime

    if u_i == 0:
	print v_i, m_i, alpha_i, u_i, beta_i
	print mu_prime
    return (mu_prime, sigma_prime)


def make_prediction(x_i, mu_i):
    return np.sign(mu_i.dot(x_i))


def compute_v_i(x_i, sigma_i):
    return (x_i.dot(sigma_i)).dot(x_i)
    

def compute_m_i(x_i, true_y_i, mu_i):
    return true_y_i*(mu_i.dot(x_i))

def compute_alpha_i(v_i, xi_const, m_i, psi_const, phi_const):
    potential_alpha = 1.0 / (v_i*xi_const) * (-m_i*psi_const + np.sqrt(m_i**2 * phi_const**4 /4 + v_i*phi_const**2*xi_const))
    return max(0, potential_alpha)

def compute_u_i(alpha_i, v_i, phi_const):
    return (-alpha_i*v_i*phi_const + np.sqrt((alpha_i * v_i * phi_const)**2 + 4*v_i))**2 / 4

def compute_beta_i(alpha_i, phi_const, u_i, v_i):
    return alpha_i * phi_const / (np.sqrt(u_i) + v_i*alpha_i*phi_const)

def diag_sigma_update(sigma_i, alpha_i, phi_const, u_i, x_i):
    return diag_inverse(diag_inverse(sigma_i) + alpha_i * phi_const*(1.0/np.sqrt(u_i))*(x_i**2*np.identity(len(x_i))))

def diag_inverse(m):
    d = np.diagonal(m).copy()
    for i in range(len(d)):
	d[i] = 1.0/d[i]	
    return d*np.identity(len(d))

def main():
    samples = 5000
    # all_y, all_x = read_multiple_days(0, 1)
    all_x, all_y = datasets.make_classification(n_samples=samples, n_features=dims)

    for i in range(len(all_x)):
	x = all_x[i]
	if x.dot(x) == 0:
	    print i, x
    


    for i in range(len(all_y)):
	if all_y[i] == 0:
	    all_y[i] = -1

    current_mu = mu_0
    current_sigma = sigma_0
    num_wrong = 0
    for i in range(samples):
	pred = make_prediction(all_x[i], current_mu)
	diff = all_y[i] - pred
	if diff != 0:
	    num_wrong += 1
	
	current_mu, current_sigma = update_rule(all_x[i], all_y[i], current_mu, current_sigma, i)
	# print current_mu

    print "Num wrong: %d" % num_wrong
    print "initial mu", mu_0
    print "final mu", current_mu





if __name__=="__main__":
    main()



