
# (virtualenv venv)
# source venv/bin/activate


# from Tkinter import Tk, Toplevel, Canvas
import math
from scipy import misc, ndimage, spatial, odr
import numpy as np
import skimage.morphology as mp
import matplotlib.pyplot as plt

plt.set_cmap('binary')

def run(): 

	a = np.invert(misc.imread("a.png")) # Image is black on white

	pc = point_cloud(a)
	kd_tree = spatial.KDTree(pc)
	x = pc[np.random.randint(pc.shape[0]), :]
	# neighbors_idx = kd_tree.query_ball_point(x, 5)
	(distances, neighbors_idx) = kd_tree.query(x, 50)
	neighbors = pc[neighbors_idx,:]
	# x_sort = np.argsort(pc[0, :])
	# y_sort = np.argsort(pc[1, :])

	data = np.concatenate(([x], neighbors))
	beta = fitODR(data)
	print(beta)

	plt.scatter(neighbors[:,1], neighbors[:, 0], marker=".")
	plt.scatter(x[1], x[0], marker="+")
	# plt.scatter(pc[:,1], pc[:,0], marker="+")

	plt.plot(np.array(range(0,100)) * beta[0] + beta[1])

	plt.imshow(a)
	plt.show()


# image = np.zeros((max_x, max_y))
# image[coordinates] = 1

def point_cloud(a):
	return np.array(np.nonzero(a)).T


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