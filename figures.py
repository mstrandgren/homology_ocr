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
	image_preprocessing()

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
	k = 20
	r = .5
	w = .6
	vertices = get_image('P', 0, size=200, sample_size=N)[0]
	plot_tangents(vertices, plt, k = k)
	plt.show()

def puv_curve(): 
	N = 500
	N_s = 50
	k = 100
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
	N_s = 100
	k = 20
	r = .1
	w = .6
	P = get_image('P', 1, size=200, sample_size=N)[0]

	f, ax = plt.subplots(1, 2)
	plot_triangulation(P, plt=ax[0], N_s = N_s, k = k, r = 0.7, w = w, triangulation='rips4')
	ax[0].set_title("Rips Tangent Complex")
	plot_triangulation(P, plt=ax[1], N_s = N_s, k = k, r = 0.15, w = w, triangulation='rips2')
	ax[1].set_title("Rips Vertex Complex")
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