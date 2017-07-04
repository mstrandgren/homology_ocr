
import math
from scipy import misc, ndimage
from functools import partial
import numpy as np
import skimage.morphology as mp
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy import odr, spatial

import homology as hm
import bar_code as bc
from test_plots import *
from man_data import data as manual_data

from complex_creator import draw_complex


# AD
# BOPQR
# CEFGHIJKLMNSTUVXYZ


def run(): 

	im_size = 30
	letters = 'A'
	M = 5
	K = len(letters)

	vertices = [0] * (M * K)

	for k, l in enumerate(letters):
		for m in range(M): 
			idx = k * M + m
			vertices[idx] = get_image(l, m, im_size)[0]


	k = 20
	r = 2 * 1.01 * math.sqrt(2) / im_size
	w = 0

	plot_difference(vertices, k = k, r = r, w = w)

	f, ax = plt.subplots(3,M)
	for m in range(M):
		plot_curve_color(vertices[m], plt = ax[0][m], k = k, r = r, w = w)
		plot_edges(vertices[m], plt = ax[1][m], k = k, r = r, w = w)
		plot_bar_code(vertices[m], plt = ax[2][m], k = k, r = r, w = w)



	# A = np.zeros((5,5))
	# edges = np.array([[0,1],[1,2],[1,3],[2,3],[3,4]])
	# A[edges[:,0], edges[:,1]] = 1
	# A[edges[:,1], edges[:,0]] = 1
	# print(A)
	# print(np.argwhere(np.diagonal(A.dot(A).dot(A))))

	# print(np.argwhere(np.bincount(edges[:,0])>1).flatten())

	# plot_difference(vertices, plt=plt)

	# N_grid = 5
	# plt.imshow(im)
	# M,N = im.shape

	# grid = np.mgrid[0:N_grid, 0:N_grid]
	# x = grid[0] * N/N_grid + N/(2 * N_grid)
	# y = grid[1] * M/N_grid + M/(2 * N_grid)
	# plt.scatter(x, y, marker='+')


	plt.show()
	return


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
	image = misc.imresize(image, [size, size], interp="bicubic")
	image[image>0] = 1 # Make binary
	image = mp.skeletonize(image) # Skeleton
	vertices = np.flip(np.array(np.nonzero(image)).T, axis=1)
	vertices = vertices * 2.0 / size - 1
	return vertices, image, original



run()
