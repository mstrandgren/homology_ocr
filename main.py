
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

	bitmap_thick = np.invert(misc.imread("a.png")) # Image is black on white
	bitmap = mp.thin(bitmap_thick)

	point_cloud = np.array(np.nonzero(bitmap)).T
	kd_tree = spatial.KDTree(point_cloud)

	# x = pc[np.random.randint(pc.shape[0]), :]
	
	def find_tangent(x):
		(distances, neighbors_idx) = kd_tree.query(x, 10)
		# Alternative: neighbors_idx = kd_tree.query_ball_point(x, 5) 
		neighbors = point_cloud[neighbors_idx,:]
		data = np.concatenate(([x], neighbors))
		beta = fitODR(data)
		return math.atan(beta[0])

	tangents = np.apply_along_axis(find_tangent, axis=1, arr=point_cloud)
	
	plt.set_cmap('hot')
	plt.scatter(point_cloud[:,1], point_cloud[:,0], marker=".", c=tangents)

	# plt.plot(np.array(range(0,100)) * beta[0] + beta[1])

	plt.imshow(bitmap)
	plt.show()


# image = np.zeros((max_x, max_y))
# image[coordinates] = 1

def point_cloud(a):
	return 


def fitODR(data):
	def f(B, x):
		return B[0]*x + B[1]
	linear = odr.Model(f)
	data = odr.Data(data[:,1], data[:,0])
	o = odr.ODR(data, linear, beta0=[1,0])
	out = o.run()
	return out.beta



# ------------------------------------------------------------------------

run()