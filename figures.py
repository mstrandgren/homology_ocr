import math
import numpy as np
import matplotlib.pyplot as plt

import homology as hm
import barcode as bc
from test_plots import *
from data import *
from man_data import data as manual_data

def ellipse_filtration(): 
	vertices, edges = get_ellipse(16, .5)

	k = 4
	r = 1
	w = 0
	# plot_edges(vertices, edges, plt, k = k, r = r, w = w)
	plot_filtration(vertices, edges, plt, k = k, r = r, w = w)
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


def run():
	# ellipse_filtration()
	# ellipse_barcode()
	p_tangents()
	# p_curve()

if __name__ == "__main__":
	run()