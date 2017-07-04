import math
from functools import partial
from scipy import odr, spatial
import numpy as np
import bar_code as bc


def get_tangent_space(vertices, k = 4, r = .6, w = .5, double = True): 
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


def test_bar_code(vertices, edges = None, k=4, r=.6, w=.5):
	tangent_space, tangents, curve = get_tangent_space(vertices, k, r, w)
	if edges is None:
		edges = find_edges(tangent_space, vertices, r)
	ordered_simplices, curve_lookup = get_ordered_simplices(vertices, curve, edges)
	bar_code = bc.get_bar_code(ordered_simplices, degree_values=curve[np.argsort(curve)])
	return bar_code, ordered_simplices


def process_shape(vertices, k=4, r=.6, w=.5): 
	"""
	Takes a set of vertices and returns simplical complex and bar code for persistent homology
	"""

	tangent_space, tangents, curve = get_tangent_space(vertices, k, r, w)
	edges = find_edges(tangent_space, vertices, r)

	ordered_simplices, curve_lookup = get_ordered_simplices(vertices, curve, edges)
	bar_code = bc.get_bar_code(ordered_simplices, degree_values=curve[np.argsort(curve)])

	return ordered_simplices, bar_code


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
	c = to_complex(np.sort(edges, axis=1).T)
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

def find_curve(vertices, k, kd_tree):
	"""
	vertices: Nx2-array of vertices
	k: number of neighbors
	kd_tree: A KD-tree of the vertices (pre-made)
	"""
	find_tangent_partial = partial(find_tangent, k=k, vertices=vertices, kd_tree=kd_tree)
	(tangents, curve) = np.apply_along_axis(find_tangent_partial, arr=vertices, axis=1).T
	return curve, tangents


def find_tangent(x, k, vertices, kd_tree):
	"""
	Returns a Nx2 matrix with column vectors representing tangent angles (in radians [-PI, PI])
	and curvature (as the second derivative of the osculating parabola ]-Inf, Inf[)
	"""

	(distances, neighbors_idx) = kd_tree.query(x, k)
	# Alternative: neighbors_idx = kd_tree.query_ball_point(x, 5) 
	# k = neighbors.shape[0] # Number of neighbors
	neighbors = vertices[neighbors_idx,:]
	center = np.mean(neighbors, axis=0)


	# Find tangent
	beta = fitODR(neighbors)
	tangent = math.atan(beta[0])

	# Find curve
	rot_matrix = np.array([[np.cos(-tangent), -np.sin(-tangent)], [np.sin(-tangent), np.cos(-tangent)]])
	translated_neighbors = neighbors - np.repeat([center, center], k/2, axis=0)
	transformed_neighbors =  np.dot(rot_matrix, translated_neighbors.T).T

	# A Barcode Shape Descriptor... 3.4
	p = np.array([0,1,2])
	X = transformed_neighbors[:,0:1] #0:1 instead of 0 to make it a column vector
	A = np.power(X, p)
	Y = transformed_neighbors[:,1:2]
	C = np.linalg.pinv(A.T.dot(A)).dot(A.T).dot(Y)
	curve = np.abs(2*C[2,0])

	return tangent, curve

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
	if v.shape[0] == 2: x, y = v
	else: x, y = v.T
	return x + y * 1.0j		

def to_real(c, dtype = int): 
	return np.array([np.real(c), np.imag(c)]).T.astype(dtype)
