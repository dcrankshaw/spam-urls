
import numpy as np
import numpy.linalg

dims = 10
a_const = 1.0
eta_const = 0.9
mu_0 = np.zeros(dims)
sigma_0 = a*np.identity(dim)
phi_const = scipy.stats.norm.ppf(eta)
psi_const = 1 + phi**2 / 2.0
xi_const = 1 + phi**2


# algorithm from "Exact Convex Confidence-Weighted Learning"
# http://nips.cc/Conferences/2008/Program/event.php?ID=1222

def update_rule(x_i, true_y_i, mu_i, sigma_i):
    pred_y_i = make_prediction(x_i, mu_i)
    v_i = compute_v_i(x_i, sigma_i)
    m_i = compute_m_i(x_i, true_y_i, mu_i)
    alpha_i = compute_alpha_i(v_i, xi_const, m_i, psi_const, phi_const)
    u_i = compute_u_i(alpha_i, v_i, phi_const)
    beta_i = compute_beta_i(alpha_i, phi_const, u_i, v_i)

    mu_prime = mu_i + (alpha_i * true_y_i * sigma_i.dot(x_i))
    sigma_prime = diag_sigma_update(sigma_i, alpha_i, phi_const, u_i, x_i)
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
    return (-alpha_i*v_i*phi_const + np.sqrt(alpha_i**2 * v_i**2 * phi_const**2 + 4*v_i))**2 / 4

def compute_beta_i(alpha_i, phi_const, u_i, v_i):
    return alpha_i * phi_const / (np.sqrt(u_i) + v_i*alpha_i*phi_const)

def diag_sigma_update(sigma_i, alpha_i, phi_const, u_i, x_i):
    return linalg.inv(linalg.inv(sigma_i) + alpha_i * phi_const*(u_i**(-0.5))*(x_i**2*np.identity(len(x_i))))
