import math
import numpy as np
import matplotlib.pyplot as plt
import homology as hm
import barcode as bc
from data import *
from utils import sparse_sample
from plot_utils import *


def run():
	# ellipse_filtration()
	# ellipse_barcode()
	# p_tangents()
	# puv_curve()
	# rips_test()
	# delaunay_test()
	# alpha_test()
	# witness_test()
	# delaunay_vs_alpha()
	# triangulation_test()
	distance_test()
	# image_preprocessing()

def ellipse_filtration(): 
	vertices, edges = get_ellipse(16, .5)

	k = 4
	r = 1
	w = 0
	# plot_edges(vertices, edges, plt, k = k, r = r, w = w)
	plot_filtration(vertices, edges, plt, N_s = vertices.shape[0], k = k, r = r, w = w)
	plt.tight_layout()
	plt.show()

def ellipse_barcode():
	vertices, edges = get_ellipse(16, .5)

	k = 4
	r = 1
	w = 0
	# plot_edges(vertices, edges, plt, k = k, r = r, w = w)
	plot_barcode(vertices, edges, plt, N_s = vertices.shape[0], k = k, r = r, w = w)
	plt.tight_layout()
	plt.show()


def p_tangents(): 
	N = 500
	N_s = 50
	k = 50
	r = .5
	w = .6
	vertices = get_image('P', 0, size=200, sample_size=N)[0]
	plot_tangents(vertices, plt, k = k)
	plt.show()

def puv_curve(): 
	N = 500
	N_s = 50
	k = 50
	r = .5
	w = .6
	P = get_image('P', 0, size=200, sample_size=N)[0]
	U = get_image('U', 0, size=200, sample_size=N)[0]
	V = get_image('V', 0, size=200, sample_size=N)[0]
	f, ax = plt.subplots(1,3)
	plot_curve(P, ax[0], k = k, w = w)
	plot_curve(U, ax[1], k = k, w = w)
	plot_curve(V, ax[2], k = k, w = w)

	plt.show()


def rips_test():
	N = 500
	N_s = 50
	k = 50
	w = .6
	P = get_image('P', 0, size=200, sample_size=N)[0]

	f, ax = plt.subplots(1, 2)
	plot_triangulation(P, plt=ax[0], N_s = N_s, k = k, r = 0.80, w = w, triangulation='rips4')
	ax[0].set_title("Rips Tangent Complex")
	plot_triangulation(P, plt=ax[1], N_s = N_s, k = k, r = 0.4, w = w, triangulation='rips2')
	ax[1].set_title("Rips Vertex Complex")
	plt.show()


def delaunay_test():
	N = 500
	N_s = 20
	k = 50
	r = .2
	w = .6
	P = get_image('P', 0, size=200, sample_size=N)[0]

	f, ax = plt.subplots(1, 2)
	plot_triangulation(P, plt=ax[0], N_s = N_s, k = k, r = r, w = w, triangulation='delaunay4')
	ax[0].set_title("Delaunay Triangulation of Tangent Space")
	plot_triangulation(P, plt=ax[1], N_s = N_s, k = k, r = r, w = w, triangulation='delaunay2')
	ax[1].set_title("Delaunay Triangulation of Vertex Space")
	plt.show()


def alpha_test():
	N = 500
	N_s = 50
	k = 50
	w = .7
	P = get_image('P', 0, size=200, sample_size=N)[0]

	f, ax = plt.subplots(1, 2)
	plot_triangulation(P, plt=ax[0], N_s = N_s, k = k, r = .6, w = w, triangulation='alpha4')
	ax[0].set_title("α Tangent Complex")
	plot_triangulation(P, plt=ax[1], N_s = N_s, k = k, r = .2, w = w, triangulation='alpha2')
	ax[1].set_title("α Vertex Complex")
	plt.show()


def delaunay_vs_alpha():
	N = 500
	N_s = 20
	k = 50
	r = .4
	w = .6
	P = get_image('P', 0, size=200, sample_size=N)[0]

	f, ax = plt.subplots(1, 2)
	plot_triangulation(P, plt=ax[0], N_s = N_s, k = k, r = r, w = w, triangulation='delaunay2')
	ax[0].set_title("Delaunay Vertex Complex")
	plot_triangulation(P, plt=ax[1], N_s = N_s, k = k, r = r, w = w, triangulation='alpha2')
	ax[1].set_title("α Vertex Complex")
	plt.show()

def witness_test(): 
	N = 500
	N_s = 50
	k = 50
	w = .7
	P = get_image('P', 0, size=200, sample_size=N)[0]

	f, ax = plt.subplots(1, 2)
	plot_triangulation(P, plt=ax[0], N_s = N_s, k = k, r = 1, w = w, triangulation='witness4')
	ax[0].set_title("Witness Tangent Complex")
	plot_triangulation(P, plt=ax[1], N_s = N_s, k = k, r = 1, w = w, triangulation='witness2')
	ax[1].set_title("Witness Vertex Complex")
	plt.show()

def triangulation_test():
	N = 500
	N_s = 30
	k = 50
	r = .4
	w = .6
	M = 3
	letter = 'T'

	f, ax = plt.subplots(3, 3)
	for m in range(M):
		vs,_,_,original = get_image(letter, m, size=200, sample_size=N)
		ax[0][m].set_title("Original image")
		ax[0][m].imshow(original, cmap='binary')
		ax[0][m].set_xticks([])
		ax[0][m].set_yticks([])

		plot_triangulation(vs, plt=ax[1][m], N_s = N_s, k = k, r = .3, w = .7, triangulation='alpha2')
		ax[1][m].set_title("α Complex (2D)")
		std_plot(ax[1][m])

		plot_triangulation(vs, plt=ax[2][m], N_s = N_s, k = k, r = .5, w = .7, triangulation='witness2')
		ax[2][m].set_title("Witness Complex (2D)")
		std_plot(ax[2][m])

	plt.tight_layout()
	plt.show()


def distance_test():
	letters = 'ABCDE'
	L = len(letters)
	M = 5
	N = 500
	N_s = 50
	k = int(N/5)
	w = .7
	r = .5
	triangulation = 'alpha4'

	barcodes = []

	f, bax = plt.subplots(len(letters),M)
	f, iax = plt.subplots(len(letters),M)


	for i, letter in enumerate(letters):
		for j in range(0, M):
			print("Doing {0}-{1}".format(letter, j))
			idx = i * M + j
			vertices = get_image(letter, j, size=100, sample_size=N)[0]
			v, t, c, e, simplices = hm.get_all(vertices, N_s = N_s, k = k, r = r, w = w, triangulation=triangulation)
			b = bc.get_barcode(simplices, degree_values=c[np.argsort(c)])
			b = b[b[:, 2] == 0, :] # Only 1-bars
			barcodes.append(b)
			plot_barcode_gant(b, plt=bax[i][j])
			bax[i][j]
			plot_edges(v, e, iax[i][j])
			std_plot(iax[i][j])

	print("Doing diffs...")
	M_b = len(barcodes)
	diffs = np.zeros([M_b,M_b])
	inf = 1e14
	for i in range(M_b):
		for j in range(M_b):
			diffs[i,j] = bc.barcode_diff(barcodes[i], barcodes[j], inf=inf)
			# print("Diff ({0},{1}) = {2:1.3f}".format(i, j, diffs[i,j]))

	plt.figure()
	plot_diffs(diffs, letters, plt)
	plt.show()



def image_preprocessing():
	N = 500
	N_s = 30
	sample, vertices, image, original = get_image('A', 1, size=200, sample_size=N)
	f, ax = plt.subplots(2,2, gridspec_kw = {'width_ratios':[1,1]})
 
	ax[0][0].imshow(original, cmap='binary')
	ax[0][0].set_xticks([])
	ax[0][0].set_yticks([])
	ax[0][0].set_title("Original Image")

	ax[0][1].imshow(image, cmap='binary')
	ax[0][1].set_xticks([])
	ax[0][1].set_yticks([])
	ax[0][1].set_title("Scaled, Cropped & Thinned")


	ax[1][0].scatter(sample[:,0], sample[:,1], marker='.')
	ax[1][0].invert_yaxis()
	ax[1][0].set_xticks(np.arange(-1, 1.1))
	ax[1][0].set_yticks(np.arange(-1, 1.1))
	ax[1][0].set_title('Random Sampling')

	sparse_idx = sparse_sample(sample, N_s)
	sparse = sample[sparse_idx, :]
	ax[1][1].scatter(sparse[:,0], sparse[:,1], marker='.')
	ax[1][1].invert_yaxis()
	ax[1][1].set_xticks(np.arange(-1, 1.1))
	ax[1][1].set_yticks(np.arange(-1, 1.1))
	ax[1][1].set_title('Downsampled')
	plt.tight_layout()
	plt.show()



if __name__ == "__main__":
	run()