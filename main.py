
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



def run(): 
	# d1 = manual_data['D']
	# v1 = np.array(d1['vertices'])
	# e1 = np.array(d1['edges'])

	# d2 = manual_data['O']
	# v2 = np.array(d2['vertices'])
	# e2 = np.array(d2['edges'])

	N = 8
	M = 2
	vertices = [0] * M
	edges = [0] * M

	for m in range(M):
		vertices[m] = get_ellipse(N, .5 - .1 * m)
		edges[m] = np.array([np.arange(N), np.append(np.arange(N-1) + 1, 0)]).T

	# plot_filtration(vertices[0], edges[0])
	plot_difference(vertices, edges, plt)

	# plot_filtration(vertices, edges)
	# plt.figure()
	f, ax = plt.subplots(1,2)
	plot_bar_code(vertices[0], edges[0], plt=ax[0])
	plot_bar_code(vertices[1], edges[1], plt=ax[1])

	# N = vertices.shape[0]
	# k = int(N / 4)
	# r = .3
	# w = 3

	# print(vertices)

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
