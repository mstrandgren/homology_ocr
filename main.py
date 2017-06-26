
# (virtualenv venv)
# source venv/bin/activate


# from Tkinter import Tk, Toplevel, Canvas
import math
from functools import partial
from scipy import misc, ndimage, spatial, odr
import numpy as np
import skimage.morphology as mp
import matplotlib.pyplot as plt


plt.set_cmap('binary')

def run(): 

	k = 20

	# Get point cloud
	bitmap_thick = np.invert(misc.imread("b.png")) # Image is black on white
	bitmap = mp.thin(bitmap_thick)
	# bitmap = bitmap_thick
	point_cloud = np.flip(np.array(np.nonzero(bitmap)).T, axis=1)

	# Find tangents
	kd_tree = spatial.KDTree(point_cloud)
	find_tangent_partial = partial(find_tangent, k=k, point_cloud=point_cloud, kd_tree=kd_tree)
	(tangents, curve) = np.apply_along_axis(find_tangent_partial, axis=1, arr=point_cloud).T

	# point_cloud = [x, y]
	# tangents = [tangent, curve]



	# Plot result
	
	plt.set_cmap('hot')
	(X,Y) = point_cloud.T
	# plt.scatter(X, -Y, marker=".", c=curve)

	rand_idx = np.random.randint(point_cloud.shape[0])
	lines = edges(rand_idx, r=5, point_cloud=point_cloud, kd_tree=kd_tree)

	# plt.figure()
	# threshold = np.max(curve) * .5
	# (X,Y) = point_cloud[curve > threshold, :].T
	
	# indices = [0,5]
	# for idx1,_ in enumerate(X):
	# 	for idx2,_ in enumerate(X):
	# 		plt.plot(X[[idx1,idx2]], Y[[idx1,idx2]], c="magenta", lw=1)

	plt.scatter(X, Y, marker=".")

	def plot_edge(edge):
		plt.plot(X[edge], Y[edge], c="magenta", lw=1)

	np.apply_along_axis(plot_edge, axis=1, arr=lines).T		


	# plt.figure()
	# plt.scatter(point_cloud[:,1], point_cloud[:,0], marker=".", c=curve)
	# plt.figure()

	# plt.imshow(bitmap)
	# plt.scatter(x[1], x[0], marker=".", c=tangents)
	# plt.plot(np.array(range(0,100)) * beta[0] + beta[1])
	plt.show()


def edges(idx, r, point_cloud, kd_tree):
	x = point_cloud[idx,:]
	neighbors_idx = np.array(kd_tree.query_ball_point(x, r))
	N = neighbors_idx.shape[0]
	return np.array([np.ones(N).astype(int) * idx, neighbors_idx]).T


def find_tangent(x, k, point_cloud, kd_tree):
	"""
	Returns a Nx2 matrix with column vectors representing tangent angles (in radians [-PI, PI])
	and curvature (as the second derivative of the osculating parabola ]-Inf, Inf[)
	"""

	(distances, neighbors_idx) = kd_tree.query(x, k)
	# Alternative: neighbors_idx = kd_tree.query_ball_point(x, 5) 
	# k = neighbors.shape[0] # Number of neighbors
	neighbors = point_cloud[neighbors_idx,:]
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

run()