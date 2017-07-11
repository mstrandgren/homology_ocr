import math
import numpy as np
import matplotlib.pyplot as plt
import homology as hm
import barcode as bc
from data import *
from utils import sparse_sample
from plot_utils import *


def run():
	# image_preprocessing()
	ellipse_filtration()
	# ellipse_barcode()
	# p_tangents()
	# p_curve()

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
	plot_barcode(vertices, edges, plt, k = k, r = r, w = w)
	plt.tight_layout()
	plt.show()


def p_tangents(): 
	vertices = get_image('P', 0)[0]
	k = 16
	r = 1
	w = .1
	plot_tangent_space(vertices, plt, k = k, r = r, w = w)
	plt.show()

def p_curve(): 
	# vertices, edges = get_ellipse(16, .5)
	P = get_image('P', 0)[0]
	U = get_image('U', 0)[0]
	V = get_image('V', 0)[0]
	k = 16
	r = 1
	w = 0
	f, ax = plt.subplots(1,3)
	plot_curve_color(P, ax[0], k = k, r = r, w = w)
	plot_curve_color(U, ax[1], k = k, r = r, w = w)
	plot_curve_color(V, ax[2], k = k, r = r, w = w)

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