
import math
import numpy as np
import matplotlib.pyplot as plt

import homology as hm
import bar_code as bc
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


def test_bar_code(): 
	im_size = 30
	vertices = get_image2('P',0, size=im_size)[0]
	k = 20
	r = 2 * 1.01 * math.sqrt(2) / im_size
	w = 0
	plot_filtration(vertices, k = k, r = r, w = w, annotate = False)
	# plot_edges(vertices, k = k, r = r, w = w, annotate = False)
	plt.figure()
	plot_bar_code(vertices, k = k, r = r, w = w, annotate = False)
	plt.show()

def run(): 
	test_triangulation()
	# run_manual()
	# test_bar_code()
	return

	im_size = 30
	letters = 'APB'
	M = 5
	K = len(letters)

	vertices = [0] * (M * K)

	for k, l in enumerate(letters):
		for m in range(M): 
			idx = k * M + m
			vertices[idx] = get_image2(l, m, im_size)[0]


	k = 20
	r = 2 * 1.01 * math.sqrt(2) / im_size
	w = 0

	plot_difference(vertices, k = k, r = r, w = w)

	# f, ax = plt.subplots(3,M)
	# for m in range(M):
	# 	plot_curve_color(vertices[m], plt = ax[0][m], k = k, r = r, w = w)
	# 	plot_edges(vertices[m], plt = ax[1][m], k = k, r = r, w = w)
	# 	plot_bar_code(vertices[m], plt = ax[2][m], k = k, r = r, w = w)

	# plot_difference(vertices, plt=plt)

	f, ax = plt.subplots(K,M)
	for i, l in enumerate(letters):
		for m in range(M):
			idx = i * M + m
			# ax[i][m].imshow(vertices[idx])
			# plot_curve_color(vertices[idx], plt = ax[i][m],  k = k, r = r, w = w)
			# plot_edges(vertices[idx], plt = ax[i][m],  k = k, r = r, w = w)
			plot_bar_code(vertices[idx], plt = ax[i][m],  k = k, r = r, w = w)
	# A = np.zeros((5,5))
	# edges = np.array([[0,1],[1,2],[1,3],[2,3],[3,4]])
	# A[edges[:,0], edges[:,1]] = 1
	# A[edges[:,1], edges[:,0]] = 1
	# print(A)
	# print(np.argwhere(np.diagonal(A.dot(A).dot(A))))

	# print(np.argwhere(np.bincount(edges[:,0])>1).flatten())


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

run()
