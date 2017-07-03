import math
import numpy as np
import matplotlib.pyplot as plt
import homology as hm
import bar_code as bc


def plot_vertices(vertices, plt = plt):
	plt.scatter(vertices[:,0], vertices[:,1], marker='.')

def plot_tangent_space(vertices, plt = plt, k=4, r=.6, w=.5, annotate = False):
	tangent_space, tangents, _ = hm.get_tangent_space(vertices, k = k, r = r, w = w, double=True)

	for idx, x in enumerate(tangent_space): 
		plt.plot([x[0], x[0] + x[2]], [x[1], x[1] + x[3]], lw = 1, c='gray')
		if annotate:
			plt.annotate("{0:1.2f}".format(tangents[idx]/math.pi), (x[0], x[1]))

	plt.scatter(1 / r * vertices[:,0], 1 / r * vertices[:,1], marker = '.')


def plot_edges(vertices, plt = plt, k=4, r=.6, w=.5):
	"""
	Fun settings for w,r: (0,.5), (0,.6), (0,3), (100, 55)
	"""
	edges = hm.test_edges(vertices, k = k, r = r, w = w)
	plt.scatter(vertices[:,0],vertices[:,1], marker='.')
	for edge in edges:
		plt.plot(vertices[edge, 0], vertices[edge, 1], lw = 1, c = 'gray')
	
	plt.scatter(vertices[:,0],vertices[:,1], marker='.')
	set_limits(1.1, plt)


def plot_curve(vertices, plt = plt, k = 4, r = .6, w = .5):
	tangent_space, tangents, curve = hm.get_tangent_space(vertices, k = k, r = r, w = w, double=False)
	for idx, x in enumerate(tangent_space): 
		normal = curve[idx] *curve[idx] * np.array([ -x[3], x[2] ])
		plt.plot([x[0], x[0] + normal[0]], [x[1], x[1] + normal[1]], lw = 1, c='gray')

	plt.scatter(1 / r * vertices[:,0], 1 / r * vertices[:,1], marker = '.')
	set_limits(1 / r + 5, plt)


def plot_curve_color(vertices, plt = plt, k = 4, r = .6, w = .5):
	tangent_space, tangents, curve = hm.get_tangent_space(vertices, k = k, r = r, w = w, double=False)
	plt.set_cmap('plasma')
	plt.scatter(vertices[:,0], vertices[:,1], marker = '.', c=curve, s=200)
	set_limits(1.1, plt)


def plot_filtration(vertices, plt = plt, k=4, r=.6, w=.5): 
	simplices, curve = hm.test_filtration(vertices, k = k, r = r, w = w)
	f, ax = plt.subplots(4,4, sharex=True, sharey=True)
	verts_degree = simplices[simplices[:,4] == 0,0:3]
	for i in range(4):
		for j in range(4):
			idx = i*4 + j
			plot_simplices(simplices, idx, vertices, ax[i][j], annotate=True)
			ax[i][j].set_title("Curve={0:1.4f}".format(curve[verts_degree[idx,0]]))

	set_limits(1.1, plt)


def plot_bar_code(vertices, plt = plt, k=4, r=.6, w=.5):
	bar_code, _ = hm.test_bar_code(vertices, k = k, r = r, w = w)
	bc.plot_barcode_gant(bar_code, plt = plt, annotate = True)

# ------------------------------------------------------------------------

def plot_simplices(simplices, degree, vertices, plt, annotate=False):
	np.apply_along_axis(plot_simplex, 
		arr=simplices[np.argwhere(simplices[:,3]<=degree).flatten(),:], 
		axis=1, 
		plt=plt, 
		vertices=vertices,
		annotate=annotate)


def plot_simplex(simplex, plt, vertices, annotate):
	(i, b1, b2, deg, k) = simplex.flatten()
	if k == 0:
		plt.plot(vertices[i,0], vertices[i,1], marker=".", zorder=2, c='k', markersize=3)
		if annotate:
			plt.annotate("{0}".format(i), (vertices[i,0], vertices[i,1]))
	if k == 1:
		plt.plot(vertices[[b1,b2],0], vertices[[b1,b2],1], lw=1, c='#aaaaaa', zorder=1)


def set_limits(l, plt):
	try:
		plt.xlim((-l ,l))
		plt.ylim((-l ,l))
	except: 
		plt.set_xlim(-l ,l)
		plt.set_ylim(-l ,l)
