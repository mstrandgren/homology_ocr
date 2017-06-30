
# (virtualenv venv)
# source venv/bin/activate


# from Tkinter import Tk, Toplevel, Canvas
import math
from functools import partial
from scipy import misc, ndimage, spatial, odr
import numpy as np
import skimage.morphology as mp
import matplotlib.pyplot as plt
import bar_code as bc


# ------------------------------------------------------------------------



	


# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------



def run(): 

	k = 8
	r = 2
	sz = 10

	# Get point cloud
	image = misc.imread("b.png")
	image_resized = misc.imresize(image, [sz, sz], interp="nearest")
	bitmap_thick = np.invert(image_resized) # Image is black on white
	bitmap = mp.thin(bitmap_thick)
	vertices = np.flip(np.array(np.nonzero(bitmap)).T, axis=1)
	N = vertices.shape[0]

	# Find tangents
	kd_tree = spatial.KDTree(vertices)

	curve = find_curve(vertices, k, kd_tree)
	edges = find_edges(vertices, r, kd_tree)

	# print("{verts} vertices\n{edges} edges".format(verts=vertices.shape[0], edges=edges.shape[0]))

	ordered_simplices = get_ordered_simplices(vertices, curve, edges)

	# print(ordered_simplices)
	P = bc.get_bar_code(ordered_simplices)
	print("P is -----")
	print(sorted(P, key=lambda tup: tup[0]))
	# bc.plot_barcode_gant(P)
	# Plot result
	# print(vertices)
	# print(ordered_simplices)

	# plot_simplices(ordered_simplices, math.inf, vertices, plt)

	# D_max = np.max(ordered_simplices[:,3])

	# f, ax = plt.subplots(3,3, sharex=True, sharey=True)
	# axs = tuple([e for tupl in ax for e in tupl])
	# for idx, subp in enumerate(axs):
	# 	subp.set_title("t = {0}".format(idx * 1))
	# 	plot_simplices(ordered_simplices, idx * 1, vertices, subp)

	# plt.show()


# ---------------------------------------------------------------------------------

def plot_simplices(simplices, degree, vertices, plt):
	np.apply_along_axis(plot_simplex, arr=simplices[np.argwhere(simplices[:,3]<=degree).flatten(),:], axis=1, plt=plt, vertices=vertices)

def plot_simplex(simplex, plt, vertices):
	(i, b1, b2, deg, k) = simplex.flatten()
	if k == 0:
		plt.plot(vertices[i,0], vertices[i,1], marker=".", zorder=2, c='k')
		plt.annotate("{0}".format(i), (vertices[i,0], vertices[i,1]))
	if k == 1:
		plt.plot(vertices[[b1,b2],0], vertices[[b1,b2],1], lw=1, c='#aaaaaa', zorder=1)

# ---------------------------------------------------------------------------------

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
	c[N_v:,0] = np.arange(N_e) + N_v # Id
	c[N_v:,1:3] = edges # Boundary
	c[N_v:,3] = np.amax([c[edges[:,0],3].T, c[edges[:,1],3].T], axis=0) # Degree
	c[N_v:,4] = np.ones(N_e) # edge_dim

	dual_sorter = c[:,3] + 1.0j * c[:,4]
	ordered_simplices = c[np.argsort(dual_sorter), :]
	return ordered_simplices


# ---------------------------------------------------------------------------------

def find_edges(vertices, r, kd_tree):
	"""
	returns Nx2 matrix, array elements are indices of the vertices that bound the edge
	"""
	N = vertices.shape[0]
	find_edges_partial = partial(find_edges_for_point, r=r, vertices=vertices, kd_tree=kd_tree)
	find_edges_array = np.vectorize(find_edges_partial, otypes=[np.ndarray])
	all_edges = np.concatenate(find_edges_array(np.arange(0,N)), axis=1).T
	edges = remove_duplicate_edges(all_edges)
	return edges


def remove_duplicate_edges(edges):
	"""
	edges is a column array
	"""
	x,y = np.sort(edges, axis=1).T
	unique_idx = np.unique(x + y*1.0j, return_index=True)[1]
	return edges[unique_idx, :]


def find_edges_for_point(idx, r, vertices, kd_tree):
	x = vertices[idx,:]
	neighbors_idx = np.array(kd_tree.query_ball_point(x, r))
	neighbors_idx = np.delete(neighbors_idx, np.argwhere(neighbors_idx==idx))
	N = neighbors_idx.shape[0]
	return np.array([np.ones(N).astype(int) * idx, neighbors_idx.T])

# ---------------------------------------------------------------------------------

def find_curve(vertices, k, kd_tree):
	"""
	vertices: Nx2-array of vertices
	k: number of neighbors
	kd_tree: A KD-tree of the vertices (pre-made)
	"""
	find_tangent_partial = partial(find_tangent, k=k, vertices=vertices, kd_tree=kd_tree)
	(tangents, curve) = np.apply_along_axis(find_tangent_partial, arr=vertices, axis=1).T
	return curve


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
	data = np.concatenate(([x], neighbors))
	beta = fitODR(data)
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

def fitODR(data):
	def f(B, x):
		return B[0]*x + B[1]
	linear = odr.Model(f)
	data = odr.Data(data[:,0], data[:,1])
	o = odr.ODR(data, linear, beta0=[1,0])
	out = o.run()
	return out.beta

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

run()