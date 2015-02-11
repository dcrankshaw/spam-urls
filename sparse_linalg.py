
import numpy as np
import scipy.sparse as sp


class SparseVector:

    def __init__(self, nz, data):
	self.nz = nz
	self.data = data
	self.max_dim = max(nz)

# sparse-dense vector multiply
# returns a new sparse vector
def sparse_dense_multiply(sp, de):
    projected_de = de[sp.nz]
    prod = sp.data*projected_de
    return SparseVector(sp.nz, prod)

# sparse dense vector dot product
# returns a scalar
def sparse_dense_dot(sp, de):
    projected_de = de[sp.nz]
    dot = np.sum(sp.data*projected_de)
    return dot

# sparse-dense element-wise sum
# returns a dense vector
def sparse_dense_sum(sp, de):
    new_de = de.copy()
    for (i, v) in zip(sp.nz, sp.data):
	new_de[i] += v
    return new_de

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
    
    for f in files:
	with open(f, 'r') as current:
	    for line in current:
		splits = line.split()
		ys.append(float(splits[0]))
		x_vals = splits[1:]
		nnz = len(x_vals)
		nz = np.zeros(nnz)
		data = np.zeros(nnz)
		for i in range(nnz):
		    sp = x_vals[i].split(":")
		    nz[i] = int(sp[0])
		    data[i] = float(sp[1])
		xs.append(SparseVector(nz, data))
    return xs, ys





















