
import math
from scipy import misc, ndimage
import numpy as np
import skimage.morphology as mp
import matplotlib.pyplot as plt
import homology as hm
import bar_code as bc
from copy import deepcopy
from test_plots import *
from scipy import odr, spatial


def run(): 

	# vertices = get_ellipse(N = 32)	
	vertices = get_image('A', 1, size = 30)[0]
	# vertices2 = get_image('O', 1, size = 10)[0]

	N = vertices.shape[0]
	k = int(N / 4)
	r = .3
	w = 3

	# print(vertices)
	kd_tree = spatial.KDTree(vertices)
	
	x = vertices[0,:]
	visited = np.array([0])
	marked = set([(0,0)])


	def distances(i, j):
		return np.linalg.norm(vertices[j,:] - vertices[i,:])


	for i in range(100000):
		# print("i={0}, x={1}".format(i, x))
		neighbors = np.array(kd_tree.query_ball_point(x, r))
		new_idx = np.logical_not(np.in1d(neighbors, visited))
		# print(new_idx)
		new_neighbors = neighbors[new_idx]
		# print(new_neighbors)

		if new_neighbors.size == 0:
			print("breaking")
			break

		new_distances = np.linalg.norm(vertices[new_neighbors,:] - x, axis=1)

		# print(new_distances)
		visited = np.append(visited, new_neighbors)
		next_idx = new_neighbors[np.argmax(new_distances)]
		# print("New idx = {0}".format(next_idx))
		marked.add((next_idx, i + 1))
		x = vertices[next_idx, :]


	plt.scatter(vertices[:,0], vertices[:,1], marker='.')

	for xi,i in marked:
		x = vertices[xi,:]
		plt.annotate("{0}".format(i), (x[0], x[1]))


	# plot_edges(vertices, k = k, r = r, w = w)
	# plt.figure()
	# plot_tangent_space(vertices, k = k, r = r, w = .5)

	# f, ax = plt.subplots(1,2)
	# plot_edges(vertices, k = k, r = r, w = w, plt = ax[0])
	# plot_bar_code(vertices, k = k, r = r, w = w, plt = ax[1])

	# plot_curve_color(vertices, k = k, r = r, w = w)
	# plot_filtration(vertices, k = k, r = r, w = w)

	# plt.tight_layout()
	plt.show()
	return

	tangent_space, tangents, _ = hm.get_tangent_space(vertices, w = 8)


	# (simplices, bar_code, curve, tangents, edges) = hm.process_shape(vertices, test=True)
	# print(bar_code)

	plt.set_cmap('gray')
	# plot_simplices(simplices, math.inf, vertices, plt, annotate=True)

	for idx, x in enumerate(tangent_space): 
		plt.plot([x[0], x[0] + x[2]], [x[1], x[1] + x[3]])
		plt.annotate("{0:1.2f}".format(tangents[idx]/math.pi), (x[0], x[1]))
	# ax = plt.gca()
	# ax.set_facecolor('#ffffaa')

	plt.scatter(vertices[:,0],vertices[:,1], marker='.')
	# verts_degree = simplices[simplices[:,4] == 0,0:3]

	# f, ax = plt.subplots(4,4, sharex=True, sharey=True)
	# for i in range(4):
	# 	for j in range(4):
	# 		idx = i*4 + j
	# 		plot_simplices(simplices, idx, vertices, ax[i][j], annotate=True)
	# 		ax[i][j].set_title("Curve={0:1.4f}".format(curve[verts_degree[idx,0]]))

	# plt.figure()
	# bc.plot_barcode_gant(bar_code, plt, annotate=True)

	plt.tight_layout()
	plt.show()

# ---------------------------------------------------------------------------------

def get_ellipse(N = 16, skew = 0.7):
	X = skew * np.cos(np.arange(N) * 2.0 * math.pi / N)
	Y = np.sin(np.arange(N) * 2.0 * math.pi / N)
	vertices = np.array([X,Y]).T
	return vertices

# ---------------------------------------------------------------------------------

def analyze_image(image): 
	vertices = get_vertices(image)
	simplices, bar_code = hm.process_shape(vertices)	


def get_image(letter, number, size=50):
	sample_idx = ord(letter) - ord('A') + 11
	image = misc.imread("./res/img/Sample{0:03d}/img{0:03d}-{1:03d}.png".format(sample_idx, number + 1))
	image = np.invert(image)[:,:,0] # Make background = 0 and letter > 0
	original = image
	mask = image > 0
	image = image[np.ix_(mask.any(1),mask.any(0))] # Crop
	image = misc.imresize(image, [size*2, size*2], interp="nearest")
	image = mp.thin(image) # Skeleton
	image = misc.imresize(image, [size, size], interp="bicubic")
	image[image>0] = 1 # Make binary
	image = mp.thin(image) # Skeleton
	vertices = np.flip(np.array(np.nonzero(image)).T, axis=1)
	vertices = vertices * 2.0 / np.max(vertices) - 1
	return vertices, image, original

run()