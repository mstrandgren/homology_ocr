import math
from functools import partial
from scipy import spatial
import numpy as np
from utils import sparse_sample, get_tspace, to_complex


def delaunay_complex_2d(v_s):
	tri = spatial.Delaunay(v_s)
	edges = np.concatenate([tri.simplices[:,:2], tri.simplices[:,1:], tri.simplices[:,[2,0]]], axis=0)
	return edges


def delaunay_complex_4d(v_s):
	tri = spatial.Delaunay(v_s)
	edges = np.concatenate([
		tri.simplices[:,0:2], 
		tri.simplices[:,1:3], 
		tri.simplices[:,2:4], 
		tri.simplices[:,3:5], 
		tri.simplices[:,[4,0]]
		], axis=0)
	return edges


def alpha_complex_2d(vertices, r, N_s):
	sparse = sparse_sample(vertices, N_s)
	v_s = vertices[sparse,:]
	edges = delaunay_complex_2d(v_s)
	edge_lengths = np.linalg.norm(v_s[edges[:,0],:] - v_s[edges[:,1],:], axis=1)
	edges = edges[edge_lengths < r, :]
	return edges, v_s, sparse


def alpha_complex_4d(vertices, tangents, r, w, N_s):
	tspace = get_tspace(vertices, tangents, w)
	sparse = sparse_sample(tspace, N_s)
	landmarks = tspace[sparse,:]
	edges = delaunay_complex_4d(landmarks)

	edge_lengths = np.linalg.norm(tspace[edges[:,0],:] - tspace[edges[:,1],:], axis=1)
	edges = edges[edge_lengths < r, :]
	return edges, landmarks, sparse


def witness_complex_2d(vertices, N_s):
	sparse = sparse_sample(vertices, N_s)
	landmarks = vertices[sparse,:]
	D = spatial.distance.cdist(vertices, landmarks)
	closest = np.argsort(D, axis=1)
	edges = closest[:,:2]
	edges = remove_duplicate_edges(edges)
	return edges, landmarks, sparse


def witness_complex_4d(vertices, tangents, w, N_s):
	tspace = get_tspace(vertices, tangents, w)
	sparse = sparse_sample(tspace, N_s)
	landmarks = tspace[sparse,:]
	D = spatial.distance.cdist(tspace, landmarks)
	closest = np.argsort(D, axis=1)
	edges = closest[:,:2]
	edges = remove_duplicate_edges(edges)
	return edges, landmarks, sparse


def rips_complex_2d(vertices, r):
	"""
	Draw edges from all points point to all neighbors within radius r
	"""
	N = vertices.shape[0]
	kd_tree = spatial.KDTree(vertices)
	rips_complex_partial = partial(rips_complex_for_point, points=vertices, kd_tree=kd_tree, r=r)
	rips_complex_array = np.vectorize(rips_complex_partial, otypes=[np.ndarray])
	all_edges = np.concatenate(rips_complex_array(np.arange(0,N)), axis=1).T
	edges = remove_duplicate_edges(all_edges)
	return edges


def rips_complex_4d(vertices, tangents, r, w):
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


def remove_duplicate_edges(edges):
	"""
	edges is a column array
	"""
	c = to_complex(np.sort(edges, axis=1))
	unique_idx = np.unique(c, return_index=True)[1]
	return edges[unique_idx, :]


