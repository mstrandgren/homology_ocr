import math
import numpy as np
import barcode as bc
from utils import sparse_sample
from tangents import find_tangents, find_curve
from triangulation import *


def get_all(vertices, N_s, k, w, r, triangulation, edges = None):
	if triangulation == 'witness4':
		v, t, c, e = get_all_witness_4d(vertices, N_s, k, w, r)
	elif triangulation == 'witness2':
		v, t, c, e = get_all_witness_2d(vertices, N_s, k, w, r)
	elif triangulation == 'rips4':
		v, t, c, e = get_all_rips_4d(vertices, N_s, k, w, r)
	elif triangulation == 'rips2':
		v, t, c, e = get_all_rips_2d(vertices, N_s, k, w, r)
	else:
		raise Error("Unknown triangulation: {0}".format(triangulation))

	if edges is not None: e = edges
	simplices = get_ordered_simplices(v, c, e)
	return v, t, c, e, simplices


def get_all_witness_4d(vertices, N_s, k, w, r):
	tangents = find_tangents(vertices, k)
	curve = find_curve(vertices, tangents, k, w)
	edges, v_s, sparse = witness_complex_4d(vertices, tangents, w = w, N_s = N_s)
	t_s = tangents[sparse]
	c_s = curve[sparse]
	return v_s, t_s, c_s, edges


def get_all_witness_2d(vertices, N_s, k, w, r):
	tangents = find_tangents(vertices, k)
	curve = find_curve(vertices, tangents, k, w)
	edges, v_s, sparse = witness_complex_2d(vertices, N_s)
	t_s = tangents[sparse]
	c_s = curve[sparse]
	return v_s, t_s, c_s, edges


def get_all_rips_4d(vertices, N_s, k, w, r):
	tangents = find_tangents(vertices, k)
	curve = find_curve(vertices, tangents, k, w)
	sparse = sparse_sample(vertices, N_s)
	v_s = vertices[sparse,:]
	t_s = tangents[sparse]
	c_s = curve[sparse]
	edges = rips_complex_4d(v_s, t_s, w = w, r = r)
	return v_s, t_s, c_s, edges


def get_all_rips_2d(vertices, N_s, k, w, r):
	tangents = find_tangents(vertices, k)
	curve = find_curve(vertices, tangents, k, w)
	sparse = sparse_sample(vertices, N_s)
	v_s = vertices[sparse,:]
	t_s = tangents[sparse]
	c_s = curve[sparse]
	edges = rips_complex_2d(v_s, r = r)
	return v_s, t_s, c_s, edges

# ------------------------------------------------------------------------

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

	return ordered_simplices

# ------------------------------------------------------------------------

if __name__ == "__main__": 
	vertices = np.array([[1,0], [2,0], [3,0], [4,0]])
	edges = np.array([[0,1], [1,2], [2,3]])
	curve = np.array([5,3,1,3])
	os = get_ordered_simplices(vertices, curve, edges)
	print(os)


