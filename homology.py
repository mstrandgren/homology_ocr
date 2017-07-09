import math
from functools import partial
from scipy import odr, spatial
import numpy as np
import barcode as bc
import matplotlib.pyplot as plt
from data import sparse_sample

def get_all(vertices, N_s, k, w, r):
	tangents = find_tangents(vertices, k)
	curve = find_curve(vertices, tangents, k, w)
	sparse = sparse_sample(vertices, N_s)
	v_s = vertices[sparse,:]
	t_s = tangents[sparse]
	c_s = curve[sparse]
	edges = rips_complex(v_s, t_s, w = w, r = r)
	return v_s, t_s, c_s, edges

def get_tangents(vertices, k): 
	N = vertices.shape[0]
	tangents = find_tangents(vertices, k)
	return np.concatenate([vertices, tangents.reshape(N,1)], axis=1)

def get_curve(vertices, k, w):
	tangents = find_tangents(vertices, k)
	curve = find_curve(vertices, tangents, k, w)
	return curve

def get_rips_complex(vertices, tangents = None, k = 20, w = .5, r = .5):
	if tangents is None: 
		tangents = find_tangents(vertices, k)
	# curve = find_curve(vertices, tangents, k, w)
	edges = rips_complex(vertices, tangents, w = w, r = r)
	return edges

# ------------------------------------------------------------------------





def get_tangent_space(vertices, k = 4, r = .6, w = .5, double = True): 
	"""
	Tangent space is (x, y, cos(v), sin(v))
	"""
	kd_tree_2d = spatial.KDTree(vertices)
	curve, tangents = find_curve(vertices, k, kd_tree_2d)
	if double: factor = 2
	else: factor = 1
	tangent_space = np.concatenate([vertices / r, np.array([(w / r) * np.cos(factor * tangents), (w / r) * np.sin(factor * tangents)]).T], axis=1)
	return tangent_space, tangents, curve

def test_edges(vertices, k=4, r=.6, w=.5):
	tangent_space, tangents, curve = get_tangent_space(vertices, k, r, w)
	edges = find_edges(tangent_space, vertices, r)
	return edges


def test_filtration(vertices, edges = None, k=4, r=.6, w=.5):
	tangent_space, tangents, curve = get_tangent_space(vertices, k, r, w)
	if edges is None:
		edges = find_edges(tangent_space, vertices, r)

	ordered_simplices, _ = get_ordered_simplices(vertices, curve, edges)
	return ordered_simplices, curve


def test_barcode(vertices, edges = None, k=4, r=.6, w=.5):
	tangent_space, tangents, curve = get_tangent_space(vertices, k, r, w)
	if edges is None:
		edges = find_edges(tangent_space, vertices, r)
	ordered_simplices, curve_lookup = get_ordered_simplices(vertices, curve, edges)
	barcode = bc.get_barcode(ordered_simplices, degree_values=curve[np.argsort(curve)])
	return barcode, ordered_simplices


def process_shape(vertices, k=4, r=.6, w=.5): 
	"""
	Takes a set of vertices and returns simplical complex and bar code for persistent homology
	"""

	tangent_space, tangents, curve = get_tangent_space(vertices, k, r, w)
	edges = find_edges(tangent_space, vertices, r)

	ordered_simplices, curve_lookup = get_ordered_simplices(vertices, curve, edges)
	barcode = bc.get_barcode(ordered_simplices, degree_values=curve[np.argsort(curve)])

	return ordered_simplices, barcode


def get_ordered_simplices(vertices, curve, edges): 
	"""
	Orders simplices based on curve (and puts into a useful structure)
	[id, boundary1, boundary2, degree, dimension]

	"""
	N_v = vertices.shape[0]
	N_e = edges.shape[0]

	degree = curve.argsort()
	c = np.zeros([N_v + N_e,5], dtype=int)
	c[0:N_v,0] = np.arange(curve.size) # Id
	# c[:,1:3] = 0 Boundary
	c[degree,3] = np.arange(curve.size) # Degree
	# c[0:N_v,4] = 0 Dimenison

	degree_idx = np.argmax([c[edges[:,0],3].T, c[edges[:,1],3].T], axis=0) 
	c[N_v:,0] = np.arange(N_e) + N_v # Id
	c[N_v:,1:3] = edges # Boundary
	c[N_v:,3] = np.amax([c[edges[:,0],3].T, c[edges[:,1],3].T], axis=0) # Degree
	c[N_v:,4] = np.ones(N_e) # edge_dim

	dual_sorter = c[:,3] + 1.0j * c[:,4]
	ordered_simplices = c[np.argsort(dual_sorter), :]

	curve_lookup = np.zeros([N_v + N_e], dtype=float)
	curve_lookup[:N_v] = curve
	curve_lookup[N_v:] = curve[degree_idx]

	return ordered_simplices, curve_lookup


if __name__ == "__main__": 
	vertices = np.array([[1,0], [2,0], [3,0], [4,0]])
	edges = np.array([[0,1], [1,2], [2,3]])
	curve = np.array([5,3,1,3])


	os = get_ordered_simplices(vertices, curve, edges)
	print(os)


# ---------------------------------------------------------------------------------

def rips_complex(vertices, tangents, r, w):
	"""
	Draw edges from all points point to all neighbors within radius r
	"""
	N = vertices.shape[0]
	tspace = get_tspace(vertices, tangents, w)
	kd_tree = spatial.KDTree(tspace)
	rips_complex_partial = partial(rips_complex_for_point, points=tspace, kd_tree=kd_tree, r=r)
	rips_complex_array = np.vectorize(rips_complex_partial, otypes=[np.ndarray])
	all_edges = np.concatenate(rips_complex_array(np.arange(0,N)), axis=1).T
	edges = remove_duplicate_edges(all_edges)
	return edges


def rips_complex_for_point(idx, points, kd_tree, r): 
	"""
	Draw edges from point idx to all points within radius r
	"""
	x = points[idx,:]
	neighbors_idx = np.array(kd_tree.query_ball_point(x, r))
	neighbors_idx = np.delete(neighbors_idx, np.argwhere(neighbors_idx==idx))
	N = neighbors_idx.shape[0]
	return np.array([np.ones(N).astype(int) * idx, neighbors_idx.T])



def find_edges(tangent_space, vertices, r):
	"""
	returns Nx2 matrix, array elements are indices of the vertices that bound the edge
	"""
	kd_tree_4d = spatial.KDTree(tangent_space)

	N = tangent_space.shape[0]
	find_edges_partial = partial(find_edges_for_point, points=tangent_space, kd_tree=kd_tree_4d)
	find_edges_array = np.vectorize(find_edges_partial, otypes=[np.ndarray])
	all_edges = np.concatenate(find_edges_array(np.arange(0,N)), axis=1).T
	edges = remove_duplicate_edges(all_edges)
	edges = remove_small_cycles(vertices, edges, r)
	return edges


def remove_duplicate_edges(edges):
	"""
	edges is a column array
	"""
	c = to_complex(np.sort(edges, axis=1))
	unique_idx = np.unique(c, return_index=True)[1]
	return edges[unique_idx, :]


def find_edges_for_point(idx, points, kd_tree):
	x = points[idx,:]
	neighbors_idx = np.array(kd_tree.query_ball_point(x, 1))
	neighbors_idx = np.delete(neighbors_idx, np.argwhere(neighbors_idx==idx))
	N = neighbors_idx.shape[0]
	return np.array([np.ones(N).astype(int) * idx, neighbors_idx.T])


def remove_small_cycles(vertices, edges, r): 
	N = vertices.shape[0]
	A = np.zeros((N,N))
	A[edges[:,0], edges[:,1]] = 1
	A[edges[:,1], edges[:,0]] = 1
	bad_verts = np.argwhere(np.diagonal(A.dot(A).dot(A))).flatten()
	D = spatial.distance_matrix(vertices[bad_verts,:], vertices[bad_verts,:])
	A_bad = A[bad_verts,:][:,bad_verts]
	bad_edges = bad_verts[np.argwhere((D * A_bad) > r / math.sqrt(2))]
	unique_bad_edges = bad_edges[bad_edges[:,0] < bad_edges[:,1]]
	unique_bad_edges_c = to_complex(unique_bad_edges)
	edges_c = to_complex(edges)
	new_edges_c = np.setdiff1d(edges_c, unique_bad_edges_c)
	return to_real(new_edges_c)

# ---------------------------------------------------------------------------------

def find_tangents(vertices, k): 
	"""
	Returns tangent angle [-pi, pi] for all vertices
	"""
	kd_tree = spatial.KDTree(vertices)
	find_tangent_partial = partial(find_tangent_for_point, k=k, vertices=vertices, kd_tree=kd_tree)
	return np.apply_along_axis(find_tangent_partial, arr=vertices, axis=1).T


def find_tangent_for_point(x, k, vertices, kd_tree):
	"""
	Returns tangent angle [-pi, pi] for the given point x
	"""
	(distances, neighbors_idx) = kd_tree.query(x, k)
	neighbors = vertices[neighbors_idx,:]
	x_0 = np.mean(neighbors, axis=0)
	M = neighbors - x_0
	eigen_values, eigen_vectors = np.linalg.eigh(np.dot(M.T, M))
	tangent_vector = eigen_vectors[:, np.argmax(eigen_values)]
	eigen_value_ratio = np.min(eigen_values) / np.max(eigen_values)
	angle = math.atan2(tangent_vector[1], tangent_vector[0])
	return np.mod(angle, math.pi) #, eigen_value_ratio # no reason to allow angles outside of pi, and eigenvalue ratio doesn't work


def find_curve(vertices, tangents, k, w):
	"""
	vertices: Nx2-array of vertices
	tangents: Nx1-array of angles
	k: number of neighbors
	w: weight to tangent distance
	"""
	N = vertices.shape[0]
	# print(tangents.shape)
	tspace = get_tspace(vertices, tangents, w)
	kd_tree = spatial.KDTree(tspace)
	find_curve_partial = partial(find_curve_for_point, k=k, tangents=tangents, tspace=tspace, kd_tree=kd_tree)
	curve = np.apply_along_axis(find_curve_partial, arr=np.arange(N).reshape(1,N), axis=0).T
	return curve



def find_curve_for_point(idx, tspace, tangents, k, kd_tree):
	# Find curve
	idx = idx[0]
	vertices = tspace[:,:2]
	x = tspace[idx, :]
	(distances, neighbors_idx) = kd_tree.query(x, k)
	neighbors = vertices[neighbors_idx.flatten(),:2]
	# plt.scatter(neighbors[:,0], neighbors[:,1], marker='+', c='blue')

	v = tangents[idx]
	x_0 = np.mean(neighbors, axis=0)
	rot_matrix = np.array([[np.cos(-v), -np.sin(-v)], [np.sin(-v), np.cos(-v)]])
	translated_neighbors = neighbors - x_0
	transformed_neighbors =  np.dot(rot_matrix, translated_neighbors.T).T

	# A Barcode Shape Descriptor... 3.4
	p = np.array([0,1,2])
	X = transformed_neighbors[:,0:1] #0:1 instead of 0 to make it a column vector
	A = np.power(X, p)
	Y = transformed_neighbors[:,1:2]
	C = np.linalg.pinv(A.T.dot(A)).dot(A.T).dot(Y)
	curve = np.abs(2*C[2,0])
	return curve

# ------------------------------------------------------------------------

def get_tspace(vertices, tangents, w): 
	N = vertices.shape[0]
	return np.concatenate([vertices, w * np.cos(tangents * 2).reshape(N,1), w * np.sin(tangents * 2).reshape(N,1)], axis = 1)

# ------------------------------------------------------------------------

def fitQuad(data):
	def f(B, x):
		return B[0]*x*x + B[1]*x + B[2]
	quad = odr.Model(f)
	data = odr.Data(data[:,0], data[:,1])
	o = odr.ODR(data, quad, beta0=[1,0,0])
	out = o.run()
	return out.beta	

def fitODR(data):
	def f(B, x):
		return B[0]*x + B[1]
	linear = odr.Model(f)
	data = odr.Data(data[:,0], data[:,1])
	o = odr.ODR(data, linear, beta0=[1,0])
	out = o.run()
	return out.beta

# ------------------------------------------------------------------------

def to_complex(v):
	# Expects column matrix Nx2
	x, y = v.T
	return x + y * 1.0j		

def to_real(c, dtype = int): 
	return np.array([np.real(c), np.imag(c)]).T.astype(dtype)
