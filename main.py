
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy import odr, spatial

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


def plot_tangent_space_3d(tspace, subpl = '111'):
	ax = plt.gcf().add_subplot(subpl, projection='3d')
	ax.scatter(xs=tspace[:,0], ys=tspace[:,1], zs=tspace[:,2])
	ax.invert_yaxis()

def plot_tangent_space_2d(tspace, subpl = '111'):
	ax = plt.gca()
	for idx, x in enumerate(tspace): 
		# plt.plot([x[0], x[0] + x[2]], [x[1], x[1] + x[3]], lw = 1, c='gray')
		ax.arrow(x = x[0], y = x[1], dx = 0.1*math.cos(x[2]), dy = 0.1*math.sin(x[2]), lw = .3, head_width=0.02, head_length=0.04, fc='blue', ec='blue')
		# if annotate:
		# 	plt.annotate("{0:1.2f}".format(tangents[idx]/math.pi), (x[0], x[1]))

	ax.scatter(tspace[:,0], tspace[:,1], marker = '.', c='k', s=3)
	ax.invert_yaxis()


def test_tangent():
	vertices, all_points, img, _ = get_image('U', 0, size=200, sample_size=200)
	tspace = hm.get_tangents(vertices, k = 16, double = False)
	plot_tangent_space_2d(tspace)
	# f, ax = plt.subplots(1, 3)
	# ax[0].scatter(vertices[:,0], vertices[:,1], marker='.', c=tspace[:,3])
	# ax[0].invert_yaxis()
	# idx = np.argwhere(tspace[:,3] < 0.25).flatten()
	# print(idx.shape)
	# v = vertices[idx,:]
	# print(v.shape)
	# ax[1].scatter(v[:,0], v[:,1], marker='.')
	# ax[1].invert_yaxis()
	plt.show()
	return	

def test_curve_for_point():

	N = 200
	vertices = get_image('V', 0, size=100, sample_size=N)[0]

	plt.scatter(vertices[:,0], vertices[:,1], marker = '.', c='gray')
	point = vertices[78,:]
	
	k = int(N/5)
	w = 1

	tangents = hm.find_tangents(vertices, k)
	tspace = np.concatenate([vertices, w * np.cos(tangents * 2).reshape(N,1), w * np.sin(tangents * 2).reshape(N,1)], axis=1)
	ax = plt.gca()
	for idx, x in enumerate(tspace): 
		# plt.plot([x[0], x[0] + x[2]], [x[1], x[1] + x[3]], lw = 1, c='gray')
		ax.arrow(x = x[0], y = x[1], dx = x[2], dy = x[3], lw = .3, head_width=0.02, head_length=0.04, fc='#eeeeee', ec='#eeeeee')
	
	kd_tree = spatial.KDTree(tspace)
	hm.find_curve_for_point([78], tspace, tangents, k, kd_tree)
	plt.scatter(point[0], point[1], marker = '+', c = 'red')
	plt.show()


def test_curve():
	N = 200
	k = int(N/5)
	w = 1
	vertices = get_image('P', 0, size=100, sample_size=N)[0]
	curve = hm.get_curve(vertices, k)
	plt.scatter(vertices[:,0], vertices[:,1], marker = '.', c=curve, cmap='plasma', s=200)
	plt.gca().invert_yaxis()
	plt.show()

def run(): 
	# test_tangent()
	test_curve()
# ---------------------------------------------------------------------------------

run()
