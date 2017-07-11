import math
from functools import partial
from scipy import spatial
import numpy as np
from utils import sparse_sample, get_tspace


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
	tspace = get_tspace(vertices, tangents, w)
	kd_tree = spatial.KDTree(tspace)
	find_curve_partial = partial(find_curve_for_point, k=k, tangents=tangents, tspace=tspace, kd_tree=kd_tree)
	curve = np.apply_along_axis(find_curve_partial, arr=np.arange(N).reshape(1,N), axis=0).T
	return curve


def find_curve_for_point(idx, tspace, tangents, k, kd_tree):
	idx = idx[0]
	vertices = tspace[:,:2]
	x = tspace[idx, :]
	(distances, neighbors_idx) = kd_tree.query(x, k)
	neighbors = vertices[neighbors_idx.flatten(),:2]

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
