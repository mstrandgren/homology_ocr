
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import homology as hm
import barcode as bc
from test_plots import *
from data import *
from man_data import data as manual_data

# AD
# BOPQR
# CEFGHIJKLMNSTUVXYZ


def test_triangulation():
	im_size = 30
	k = 20
	r = .2 # 2 * 1.01 * math.sqrt(2) / im_size
	w = .5
	vertices = get_image2('A', 0, im_size)[0]
	# plot_curve_color(vertices,k = k, r = r, w = w)
	plot_edges(vertices, k = k, r = r, w = w, annotate = False)
	plt.show()


def run_manual():
	letters = 'ABD'
	M = 5
	K = len(letters)
	vertices = [0] * (M * K)
	edges = [0] * (M * K)

	for k, l in enumerate(letters):
		for m in range(M): 
			idx = k * M + m
			vertices[idx] = np.array(manual_data[l][m]['vertices'])
			edges[idx] = np.array(manual_data[l][m]['edges'])

	k = 20
	r = 1
	w = 0

	plot_difference(vertices, edges, k = k, r = r, w = w)
	# f, ax = plt.subplots(K,M)
	# for i, l in enumerate(letters):
	# 	for m in range(M):
	# 		idx = i * M + m
	# 		plot_curve_color(vertices[idx], plt = ax[i][m],  k = k, r = r, w = w)
	plt.show()


def test_barcode(): 
	im_size = 30
	vertices = get_image2('P',0, size=im_size)[0]
	k = 20
	r = 2 * 1.01 * math.sqrt(2) / im_size
	w = 0
	plot_filtration(vertices, k = k, r = r, w = w, annotate = False)
	# plot_edges(vertices, k = k, r = r, w = w, annotate = False)
	plt.figure()
	plot_barcode(vertices, k = k, r = r, w = w, annotate = False)
	plt.show()


def test_tangent_space():
	vertices, all_points, img, _ = get_image('V', 0, size=100, sample_size=200)
	tspace = hm.get_tangents(vertices, k = 16, double = False)

	f, ax = plt.subplots(1, 3)
	ax[0].scatter(vertices[:,0], vertices[:,1], marker='.', c=tspace[:,3])
	ax[0].invert_yaxis()
	idx = np.argwhere(tspace[:,3] < 0.25).flatten()
	print(idx.shape)
	v = vertices[idx,:]
	print(v.shape)
	ax[1].scatter(v[:,0], v[:,1], marker='.')
	ax[1].invert_yaxis()
	

	ax = plt.gcf().add_subplot('133', projection='3d')
	ax.scatter(xs=tspace[:,0], ys=tspace[:,1], zs=tspace[:,2], c=tspace[:,3])
	ax.invert_yaxis()

	plt.show()
	return	

def run(): 
	test_tangent_space()
# ---------------------------------------------------------------------------------

run()
