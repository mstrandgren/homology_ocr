
# (virtualenv venv)
# source venv/bin/activate


# from Tkinter import Tk, Toplevel, Canvas
import math
from functools import partial
from scipy import misc, ndimage, spatial, odr
import numpy as np
import skimage.morphology as mp
import matplotlib.pyplot as plt


def simplices_from_str(str):
	simplices_txt = str.split(',')

	def translate_simplex(readable_simplex):
		idx, simplex = readable_simplex
		if len(simplex) == 1:
			return (idx,)
		elif len(simplex) == 2:
			return (simplices_txt.index(simplex[0]), simplices_txt.index(simplex[1]))
		elif len(simplex) == 3:
			return (simplices_txt.index(simplex[:2]), simplices_txt.index(simplex[1:]), simplices_txt.index(simplex[0] + simplex[2]))

	return list(map(translate_simplex, enumerate(simplices_txt)))

def example():
	ordered_simplices = simplices_from_str("s,t,u,st,v,w,sw,tw,uv,sv,su,uw,tu,tuw,suw,stu,suv,stw")
	# ordered_simplices = simplices_from_str("a,b,c,d,ab,bc,cd,ad,ac,abc,acd")		
	# degrees = [0,0,1,1,1,1,2,2,3,4,5]
	print(ordered_simplices)
	print(get_bar_code(ordered_simplices))


# ------------------------------------------------------------------------

def get_bar_code(ordered_simplices):

	degrees = list(range(len(ordered_simplices)))

	T = [None] * len(ordered_simplices)
	marked = set()
	L = [[], [], []]

	def boundary(simplex):
		k = len(simplex) - 1
		if k == 0: return ()
		if k >= 1: return simplex

	def simplex_add(s1, s2):
		v1 = np.zeros(max(s1 + s2) + 1)
		v1[list(s1)] = 1
		v2 = np.zeros(max(s1 + s2) + 1)
		v2[list(s2)] = 1
		return tuple(np.argwhere(np.mod(v1 + v2, 2)).flatten())

	def remove_pivot_rows(simplex):
		print("Removing pivot rows for {0}".format(simplex))
		k = len(simplex) - 1
		d = boundary(simplex)
		print("Boundary is {0}".format(d))
		it = 0
		while len(d) > 0 and it < 100:
			it += 1
			i = max(d)
			if T[i] is None: 
				break
			q = T[i][0]
			d = simplex_add(d, T[i][1])
			print("Added {1} to simplex, d is now {0}".format(d, T[i][1]))
		return d

	def compute_intervals(s_complex):
		for j, simplex in enumerate(s_complex):
			print("j = {0}, simplex={1}".format(j ,simplex))
			d = remove_pivot_rows(simplex)
			if len(d) == 0:
				marked.add(simplex)
			else:
				i = max(d)
				k = len(s_complex[i]) - 1
				T[i] = (j, d)
				L[k].append((degrees[i], degrees[j]))

		for j, simplex in enumerate(s_complex):
			if simplex in marked and T[j] is None:
				k = len(simplex) - 1
				L[k].append((degrees[j], math.inf))

		return L
	
	return compute_intervals(ordered_simplices)


# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

def run(): 

	k = 20

	# Get point cloud
	bitmap_thick = np.invert(misc.imread("b.png")) # Image is black on white
	bitmap = mp.thin(bitmap_thick)
	# bitmap = bitmap_thick
	vertices = np.flip(np.array(np.nonzero(bitmap)).T, axis=1)
	N = vertices.shape[0]

	# Find tangents
	kd_tree = spatial.KDTree(vertices)
	find_tangent_partial = partial(find_tangent, k=k, vertices=vertices, kd_tree=kd_tree)
	(tangents, curve) = np.apply_along_axis(find_tangent_partial, arr=vertices, axis=1).T

	# vertices = [x, y], Nx2
	# tangents = Nx1, radians
	# curve = Nx1, float > 0



	# Plot result
	
	plt.set_cmap('hot')
	(X,Y) = vertices.T

	find_edges_partial = partial(find_edges, r=3, vertices=vertices, kd_tree=kd_tree)
	find_edges_array = np.vectorize(find_edges_partial, otypes=[np.ndarray])
	all_edges = np.concatenate(find_edges_array(np.arange(0,N)), axis=1).T
	edges = remove_duplicate_edges(all_edges)
	print("{verts} vertices\n{edges} edges".format(verts=vertices.shape[0], edges=edges.shape[0]))

	plt.scatter(X, Y, marker=".")

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


def remove_duplicate_edges(edges):
	"""
	edges is a column array
	"""
	x,y = np.sort(edges, axis=1).T
	unique_idx = np.unique(x + y*1.0j, return_index=True)[1]
	return edges[unique_idx, :]


def find_edges(idx, r, vertices, kd_tree):
	x = vertices[idx,:]
	neighbors_idx = np.array(kd_tree.query_ball_point(x, r))
	N = neighbors_idx.shape[0]
	return np.array([np.ones(N).astype(int) * idx, neighbors_idx.T])

# def find_edges(vertices, r, kd_tree):
# 	result = 

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



def fitODR(data):
	def f(B, x):
		return B[0]*x + B[1]
	linear = odr.Model(f)
	data = odr.Data(data[:,0], data[:,1])
	o = odr.ODR(data, linear, beta0=[1,0])
	out = o.run()
	return out.beta



# ------------------------------------------------------------------------

example()

# run()