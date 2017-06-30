
# (virtualenv venv)
# source venv/bin/activate


# from Tkinter import Tk, Toplevel, Canvas
import math
from functools import partial
from scipy import misc, ndimage, spatial, odr
import numpy as np
import skimage.morphology as mp
import matplotlib.pyplot as plt



# ------------------------------------------------------------------------



	


# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------




def run(): 

	k = 8
	r = 2

	# Get point cloud
	image = misc.imread("b.png")
	image_resized = misc.imresize(image, [10, 10], interp="nearest")
	bitmap_thick = np.invert(image_resized) # Image is black on white
	bitmap = mp.thin(bitmap_thick)
	vertices = np.flip(np.array(np.nonzero(bitmap)).T, axis=1)
	N = vertices.shape[0]

	# Find tangents
	kd_tree = spatial.KDTree(vertices)

	curve = find_curve(vertices, k, kd_tree)
	edges = find_edges(vertices, r, kd_tree)

	print("{verts} vertices\n{edges} edges".format(verts=vertices.shape[0], edges=edges.shape[0]))

	ordered_simplices = get_ordered_simplices(vertices, curve, edges)

	vidx = np.argwhere(ordered_simplices[:,4]==0)
	eidx = np.argwhere(ordered_simplices[:,4]==1)
	vertex_indices = ordered_simplices[vidx,0].flatten()
	edge_indices = ordered_simplices[eidx,0].flatten()
	# Plot result
	print(vertices)
	print(ordered_simplices)
	# plt.set_cmap('hot')
	(X,Y) = vertices.T
	plt.scatter(X, Y, marker=".")

	for idx, simplex in enumerate(ordered_simplices[vidx,:]):
		s = simplex.flatten()
		plt.annotate("{0} -> {1}".format(s[0], s[3]), (X[s[0]], Y[s[0]]))

	# v_d = np.zeros([X.size, 4], dtype=int)
	# v_d[:,:2] = vertices
	# v_d[:,2] = np.arange(X.size)[degree]
	# v_d[:,3] = curve * 1000
	


		# print(c[np.argsort(c[:,5]),:])

	# e[:,3] = c[edges[:,0],2]
	# e[:,4] = c[edges[:,1],2]

	# e_d = np.zeros([edges.shape[0], 4], dtype=int)
	# e_d[:,:2] = edges
	# e_d[:,2] = v_d[e_d[:,0],2]
	# e_d[:,3] = v_d[e_d[:,1],2]

	# print(vertices)
	# print(v_d[degree,:])
	# print(v_d)
	# print(edges)



	# plt.scatter(X[degree], Y[degree], marker=".", c=range(len(degree)))
	# plt.figure()
	# plt.plot(curve[degree])
	# plt.figure()
	# threshold = np.max(curve) * .5
	# (X,Y) = vertices[curve > threshold, :].T
	
	# plt.scatter(X, -Y, marker=".", c=curve)
	# rand_idx = np.random.randint(vertices.shape[0])
	# indices = [0,5]
	# for idx1,_ in enumerate(X):
	# 	for idx2,_ in enumerate(X):
	# 		plt.plot(X[[idx1,idx2]], Y[[idx1,idx2]], c="magenta", lw=1)


	def plot_edge(edge):
		plt.plot(X[edge], Y[edge], c="magenta", lw=1)

	np.apply_along_axis(plot_edge, arr=edges, axis=1)


	# plt.figure()
	# plt.scatter(vertices[:,1], vertices[:,0], marker=".", c=curve)
	# plt.figure()

	# plt.imshow(bitmap)
	# plt.scatter(x[1], x[0], marker=".", c=tangents)
	# plt.plot(np.array(range(0,100)) * beta[0] + beta[1])
	
	plt.show()


# ---------------------------------------------------------------------------------

def get_ordered_simplices(vertices, curve, edges): 
	"""
	Orders simplices based on curve (and puts into a useful structure)
	[id, boundary1, boundary2, degree, dimension]

	"""
	N_v = vertices.shape[0]
	N_e = edges.shape[0]

	degree = curve.argsort()
	c = np.zeros([N_v + N_e,6], dtype=int)
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