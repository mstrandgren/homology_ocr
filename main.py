
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

	k = 20
	bitmap_thick = np.invert(misc.imread("o.png")) # Image is black on white
	bitmap = mp.thin(bitmap_thick)
	# bitmap = bitmap_thick

	point_cloud = np.flip(np.array(np.nonzero(bitmap)).T, axis=1)
	kd_tree = spatial.KDTree(point_cloud)

	
	def find_tangent(x):
		(distances, neighbors_idx) = kd_tree.query(x, k)
		# Alternative: neighbors_idx = kd_tree.query_ball_point(x, 5) 
		neighbors = point_cloud[neighbors_idx,:]
		center = np.mean(neighbors, axis=0)
		print(center)


		# Find tangent
		data = np.concatenate(([x], neighbors))
		beta = fitODR(data)
		tangent = math.atan(beta[0])

		print(beta)
		# Find curve
		rot_matrix = np.array([[np.cos(-tangent), -np.sin(-tangent)], [np.sin(-tangent), np.cos(-tangent)]])
		translated_neighbors = neighbors - np.repeat([center, center], k/2, axis=0)
		transformed_neighbors =  np.dot(rot_matrix, translated_neighbors.T).T
		# print(transformed_neighbors)

		plt.imshow(bitmap)
		plt.scatter(x[0], x[1], marker=".")
		plt.scatter(center[0], center[1], marker=".")
		plt.plot(np.array(range(0,50)) * beta[0] + beta[1])
		plt.figure()
		plt.scatter(translated_neighbors[:,0], translated_neighbors[:,1], marker=".")
		plt.plot( range(-5,5), (np.array(range(-5,5)) + center[0]) * beta[0] + beta[1] - center[1])
		plt.figure()
		plt.scatter(transformed_neighbors[:,0], transformed_neighbors[:,1], marker=".")
		plt.plot( range(-5,5), np.zeros(10))


		return tangent

	# This is not how you calculate a curve, it's in the notes somewhere
	def find_curve(x): 
		(distances, neighbors_idx) = kd_tree.query(x, k)
		neighbor_tangents = tangents[neighbors_idx]
		mean = math.atan2(np.sin(neighbor_tangents * 2).sum(),np.cos(neighbor_tangents * 2).sum()) / 2
		distances = np.mod((neighbor_tangents - mean) * 2 + math.pi, math.pi * 2) - math.pi
		var = (distances ** 2).sum()/distances.size
		return var

	# tangents = np.apply_along_axis(find_tangent, axis=1, arr=point_cloud)
	# curve = np.apply_along_axis(find_curve, axis=1, arr=point_cloud)

	x = point_cloud[np.random.randint(point_cloud.shape[0]), :]
	print(x)
	tangent = find_tangent(x)

	# plt.set_cmap('hsv')
	# plt.figure()
	# plt.scatter(point_cloud[:,1], point_cloud[:,0], marker=".", c=curve)
	# plt.figure()
	# plt.scatter(point_cloud[:,1], point_cloud[:,0], marker=".", c=tangents)

	# plt.imshow(bitmap)
	# plt.scatter(x[1], x[0], marker=".", c=tangents)
	# plt.plot(np.array(range(0,100)) * beta[0] + beta[1])
	plt.show()


# image = np.zeros((max_x, max_y))
# image[coordinates] = 1

def point_cloud(a):
	return 


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