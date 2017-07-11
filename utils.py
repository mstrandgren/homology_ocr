import numpy as np
from scipy.spatial import distance


def get_tspace(vertices, tangents, w): 
	N = vertices.shape[0]
	return np.concatenate([vertices, w * np.cos(tangents * 2).reshape(N,1), w * np.sin(tangents * 2).reshape(N,1)], axis = 1)


def to_complex(v):
	# Expects column matrix Nx2
	x, y = v.T
	return x + y * 1.0j		


def to_real(c, dtype = int): 
	return np.array([np.real(c), np.imag(c)]).T.astype(dtype)


def sparse_sample(point_cloud, N):
	if N == point_cloud.shape[0]:
		return np.arange(N)

	# Returns indices
	D = distance.squareform(distance.pdist(point_cloud))
	idx = 0
	v = set([idx])
	while len(v) < N:
		new_idx = np.argmax(np.min(D[:,list(v)], axis=1))
		v.add(new_idx)
		idx = new_idx

	return list(v)
