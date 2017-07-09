
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
	N = 200
	k = int(N/5)
	vertices, all_points, img, _ = get_image('B', 0, size=200, sample_size=N)
	tspace = hm.get_tangents(vertices, k = k)
	sparse = sparse_sample(vertices, 50)
	plot_tangent_space_2d(tspace[sparse,:])
	# plot_tangent_space_3d(tspace[sparse,:])

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
	hm.find_curve_for_point([78], tspace, tangents, k = k, kd_tree = kd_tree)
	plt.scatter(point[0], point[1], marker = '+', c = 'red')
	plt.show()


def test_curve():
	N = 200
	k = int(N/10)
	w = .5
	vertices = get_image('B', 0, size=100, sample_size=N)[0]
	curve = hm.get_curve(vertices, k = k, w = w)
	plt.scatter(vertices[:,0], vertices[:,1], marker = '.', c=curve, cmap='plasma', s=200)
	plt.gca().invert_yaxis()
	plt.show()


def test_edges_for_point():
	N = 200
	k = int(N/5)
	w = .2
	r = 0.2
	vertices = get_image('P', 0, size=100, sample_size=N)[0]
	# edges = hm.get_rips_complex(vertices, k, w, r)

	tangents = hm.find_tangents(vertices, k)
	tspace = hm.get_tspace(vertices, tangents, w)
	kd_tree = spatial.KDTree(tspace)

	edges = hm.rips_complex_for_point(0, tspace, kd_tree, r = r).T
	print(edges)
	plt.scatter(vertices[:,0], vertices[:,1], marker = '.', c='gray', s=4)
	for edge in edges:
		plt.plot(vertices[edge, 0], vertices[edge, 1], lw = 1, c = 'blue')
		# if annotate:
		# 	plt.annotate("{0}".format(edge[0]), (vertices[edge[0],0], vertices[edge[0],1]))
		# 	plt.annotate("{0}".format(edge[1]), (vertices[edge[1],0], vertices[edge[1],1]))			

	plt.show()


def test_edges():
	N = 200
	N_s = 50
	k = int(N/5)
	w = .5
	r = .5
	vertices = get_image('C', 0, size=100, sample_size=N)[0]
	tangents = hm.find_tangents(vertices, k)
	sparse = sparse_sample(vertices, N_s)
	v_s = vertices[sparse,:]
	t_s = tangents[sparse]
	edges = hm.get_rips_complex(v_s, tangents=t_s, k = k, w = w, r = r)
	plt.scatter(v_s[:,0], v_s[:,1], marker = '.', c='gray', s=200)
	for edge in edges:
		plt.plot(v_s[edge, 0], v_s[edge, 1], lw = 1, c = 'blue')

	plt.gca().invert_yaxis()

	plt.show()


def test_image():
	N = 200
	sample, vertices, image, original = get_image('D', 0, size=100, sample_size=N)

	f, ax = plt.subplots(2,2)
	ax[0][0].imshow(original)
	ax[0][1].imshow(image)
	plt.show()


def test_sparse_sampling():
	N = 500
	vertices = get_image_skeleton('P', 0, size=200, sample_size=N)[0]
	plt.scatter(vertices[:,0], vertices[:,1], marker = '.', c='#eeeeee')

	sparse = sparse_sample(vertices, 50)
	plt.scatter(sparse[:,0], sparse[:,1], marker='+', c='blue')
	plt.show()

def test_barcode():
	N = 200
	N_s = 50
	k = int(N/5)
	w = .5
	r = .6
	vertices = get_image('C', 1, size=100, sample_size=N)[0]
	tangents = hm.find_tangents(vertices, k)
	curve = hm.get_curve(vertices, k = k, w = w)
	sparse = sparse_sample(vertices, N_s)
	v_s = vertices[sparse,:]
	t_s = tangents[sparse]
	c_s = curve[sparse]
	edges = hm.get_rips_complex(v_s, tangents=t_s, k = k, w = w, r = r)
	
	f, ax = plt.subplots(1,2)
	plt_edges, plt_barcode = ax
	plt_edges.scatter(v_s[:,0], v_s[:,1], marker = '.', c=c_s, s=200)
	for edge in edges:
		plt_edges.plot(v_s[edge, 0], v_s[edge, 1], lw = 1, c = 'blue')
	plt_edges.invert_yaxis()

	ordered_simplices, curve_lookup = hm.get_ordered_simplices(v_s, c_s, edges)
	barcode = bc.get_barcode(ordered_simplices, degree_values=c_s[np.argsort(c_s)])
	b0 = barcode[barcode[:,2] == 0, :]
	bc.plot_barcode_gant(b0, plt=plt_barcode)
	plt.show()

def test_distances():
	letters = 'ABC'
	M = 3
	N = 200
	N_s = 50
	k = int(N/5)
	w = .5
	r = .6

	barcodes = []

	for i, letter in enumerate(letters):
		print("Doing {0}".format(letter))
		for j in range(1, M+1):
			print("Doing {0}-{1}".format(letter, j))
			idx = i * M + j
			vertices = get_image(letter, 1, size=200, sample_size=N)[0]
			v, t, c, e = hm.get_all(vertices, N_s = N_s, k = k, r = r, w = w)
			ordered_simplices = hm.get_ordered_simplices(v, c, e)[0]
			b = bc.get_barcode(ordered_simplices, degree_values=c[np.argsort(c)])
			b = b[b[:, 2] == 0, :]
			barcodes.append(b)

	print("Doing diffs...")
	M_b = len(barcodes)
	diffs = np.zeros([M_b,M_b])
	inf = 1e14
	for i in range(M_b):
		for j in range(M_b):
			diffs[i,j] = bc.barcode_diff(barcodes[i], barcodes[j], inf=inf)

	dmax = np.max(diffs[diffs < inf/100]) * 1.1
	diffs[diffs > inf/100] = dmax
	plt.imshow(diffs, cmap='gray')
	plt.show()


def run(): 
	# test_image()
	# test_tangent()
	# test_curve()
	# test_edges_for_point()
	# test_edges()
	# test_sparse_sampling()
	# test_barcode()
	test_distances()



# Todo: 
#  - Why does size matter?
#  - Well behaved data set?
#  - Witness complex?
#  - Redo figures & settle for result



# ---------------------------------------------------------------------------------

run()
