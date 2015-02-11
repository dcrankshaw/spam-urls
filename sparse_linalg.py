
import numpy as np
import scipy.sparse as sp


class SparseVector:

    def __init__(self, nz, data):
	self.nz = nz
	self.data = data
	self.max_dim = max(nz)

# sparse-dense vector multiply
# returns a new sparse vector
def sd_multiply(sp, de):
    projected_de = de[sp.nz]
    prod = sp.data*projected_de
    return SparseVector(sp.nz, prod)

#### CAVEAT: Assumes same sparsity structure
def ss_dot(v1, v2):

    if not np.array_equal(v1.nz, v2.nz):
	raise Exception("Non matching sparsity")

    return np.sum(v1.data*v2.data)


    # # indexes in common
    # intersect_nz = np.intersect1d(v1.nz, v2.nz)
    #
    # #indexes in v1.data of intersecting values
    # v1_data_ind = np.in1d(v1.nz, v2.nz).nonzero()
    # v2_data_ind = np.in1d(v2.nz, v1.nz).nonzero()
    # prod = v1.data[v1_data_ind] * v2.data[v2_data_ind]
    # return SparseVector(intersect_nz, prod)

# sparse dense vector dot product
# returns a scalar
def sd_dot(sp, de):
    projected_de = de[sp.nz]
    dot = np.sum(sp.data*projected_de)
    return dot

# sparse-dense element-wise sum
# returns a dense vector
def sd_sum(sp, de):
    new_de = de.copy()
    for (i, v) in zip(sp.nz, sp.data):
	new_de[i] += v
    return new_de

# multiply scalar times sparse vector
def s_scalar_mult(sp, c):
    return SparseVector(sp.nz, c*sp.data)

# applies f to every element in de
def dense_unary_op(de, f):
    return f(de)

# applies f to every NON-ZERO element in sp
def sparse_unary_op(sp, f):
    new_data = f(sp.data)
    return SparseVector(sp.nz, new_data)

def read_svmlight(files):
    xs = []
    ys = []
    max_dim = 0
    
    for f in files:
	with open(f, 'r') as current:
	    for line in current:
		splits = line.split()
		ys.append(float(splits[0]))
		x_vals = splits[1:]
		nnz = len(x_vals)
		nz = np.zeros(nnz, dtype=np.int_)
		data = np.zeros(nnz)
		for i in range(nnz):
		    sp = x_vals[i].split(":")
		    nz[i] = int(sp[0])
		    data[i] = float(sp[1])
		xs.append(SparseVector(nz - 1, data))
		max_dim = max(max_dim, max(nz))
    return (xs, ys, max_dim)





















