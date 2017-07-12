import math
import numpy as np
import matplotlib.pyplot as plt
import homology as hm
import barcode as bc
from utils import get_tspace
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter

def plot_filtration(vertices, edges = None, plt = plt, N_s = 50, k=4, r=.6, w=.5, annotate = False, triangulation = 'rips4'): 
	vertices, tangents, curve, edges, simplices = hm.get_all(vertices, N_s, k, w, r, triangulation=triangulation, edges=edges)

	f, ax = plt.subplots(4,4, sharex=True, sharey=True)
	max_degree = np.max(simplices[:,3])
	degree_step = math.ceil(max_degree/16.0)
	verts_degree = simplices[simplices[:,4] == 0,0:3]
	for i in range(4):
		for j in range(4):
			idx = i*5 + j
			deg = min(idx * degree_step, max_degree)
			plot_simplices(simplices, deg, vertices, ax[i][j], annotate = annotate)

			ax[i][j].set_title("κ={0:1.4f}".format(curve[verts_degree[deg,0]]), fontsize=8)
			ax[i][j].set_xticks([])
			ax[i][j].set_yticks([])

	set_limits(1.1, plt)


def plot_barcode(vertices, edges = None, plt = plt, N_s = 50, k=4, r=.6, w=.5, annotate = False, triangulation='rips4'):
	vertices, tangents, curve, edges, simplices = hm.get_all(vertices, N_s, k, w, r, triangulation=triangulation, edges=edges)
	if edges is None: edges = _edges
	simplices = hm.get_ordered_simplices(vertices, curve, edges)
	barcode = bc.get_barcode(simplices, degree_values=curve[np.argsort(curve)])
	plot_barcode_gant(barcode, plt = plt, annotate = annotate)
	try: 
		plt.set_yticks([])
	except:
		plt.gca().set_yticks([])


def plot_barcode_gant(barcode, plt, annotate=False):
	bars = barcode[:,:3]
	inf = np.max(bars[bars != math.inf]) + 1
	markers = ('s', '.', 'x')
	marker_size = (4,10,3)
	lengths = barcode[:,1] - barcode[:,0]
	barcode = barcode[lengths > 0, :]

	for idx, row in enumerate(barcode):
		start,end = row[:2]
		if row[1] == math.inf: end = inf
		plt.plot([start,end], [idx, idx], marker=markers[row[2].astype(int)], c='k', lw=1, ms=marker_size[row[2].astype(int)])
		if annotate:
			plt.annotate("{0:4.0f}".format(barcode[idx,3]), (start,idx), horizontalalignment='right')
			plt.annotate("{0:4.0f}".format(barcode[idx,4]), (end,idx), horizontalalignment='left')

	ax = get_axis(plt)
	ax.set_xlim([-0.1 * inf, inf - 0.5])
	ax.set_xticks([])
	ax.set_yticks([])


def plot_tangents(vertices, plt = plt, k=4, annotate = False):
	vertices, tangents, curve, edges, simplices = hm.get_all(vertices, N_s = 50, k = k, w = 1, r = 1, triangulation='rips4')
	for idx, x in enumerate(vertices): 
		plt.arrow(x = x[0], y = x[1], dx = .1*np.cos(tangents[idx]), dy = .1*np.sin(tangents[idx]), lw = .3, head_width=0.02, head_length=0.04, fc='blue', ec='blue')
		if annotate:
			plt.annotate("{0:1.2f}".format(tangents[idx]/math.pi), (x[0], x[1]))

	plt.scatter(vertices[:,0], vertices[:,1], marker = '.', c='k', s=3)
	std_plot(plt)


def plot_curve(vertices, plt = plt, k = 4, w = .5):
	vertices, tangents, curve, edges, simplices = hm.get_all(vertices, N_s = vertices.shape[0], k = k, w = w, r = 1, triangulation='rips4')
	plt.scatter(vertices[:,0], vertices[:,1], marker = '.', c=curve, cmap='plasma', s=200)
	std_plot(plt)


def plot_triangulation(vertices, edges = None, plt = plt, N_s=50, k=4, r=.6, w=.5, annotate = False, triangulation = 'rips4d'):
	"""
	Fun settings for w,r: (0,.5), (0,.6), (0,3), (100, 55)
	"""
	vertices, tangents, curve, edges, simplices = hm.get_all(vertices, N_s = N_s, k = k, w = w, r = r, triangulation=triangulation)

	for edge in edges:
		plt.plot(vertices[edge, 0], vertices[edge, 1], lw = 1, c = 'gray', zorder=1)
		# if annotate:
		# 	plt.annotate("{0}".format(edge[0]), (vertices[edge[0],0], vertices[edge[0],1]))
		# 	plt.annotate("{0}".format(edge[1]), (vertices[edge[1],0], vertices[edge[1],1]))			
	
	plt.scatter(vertices[:,0],vertices[:,1], marker='.', s=5, c='k', zorder=2)
	std_plot(plt)


def plot_edges(vertices, edges, plt):
	for edge in edges:
		plt.plot(vertices[edge, 0], vertices[edge, 1], lw = 1, c = 'gray', zorder=1)
		# if annotate:
		# 	plt.annotate("{0}".format(edge[0]), (vertices[edge[0],0], vertices[edge[0],1]))
		# 	plt.annotate("{0}".format(edge[1]), (vertices[edge[1],0], vertices[edge[1],1]))			
	
	plt.scatter(vertices[:,0],vertices[:,1], marker='.', s=5, c='k', zorder=2)


def plot_diffs(diffs, letters, plt, inf=1e14):
	L = len(letters)
	M = diffs.shape[0]/L

	dmax = np.max(diffs[diffs < inf/100]) * 1.1 + 1
	diffs[diffs > inf/100] = dmax
	plt.imshow(diffs, cmap='gray')

	ax = get_axis(plt)
	major_ticks = np.arange(-1, L)*M+(M-.5)
	minor_ticks = np.arange(L)*M + M/2.0 - 0.5
	for axis in (ax.xaxis, ax.yaxis):
		axis.set_minor_locator(FixedLocator(minor_ticks))
		axis.set_minor_formatter(FixedFormatter(letters))
		axis.set_major_formatter(NullFormatter())
		axis.set_tick_params(which='major', width=1)
		axis.set_tick_params(which='minor', width=0)

	ax.set_xticks(major_ticks)
	ax.set_yticks(major_ticks)


# --------------------------------------------------------------------------------

def plot_vertices(vertices, plt = plt):
	plt.scatter(vertices[:,0], vertices[:,1], marker='.')


def plot_difference(vertices, edges = None, plt = plt, k=4, r=.6, w=.5, inf=1e14):
	M = len(vertices)
	diffs = np.zeros([M,M])
	barcodes = [0] * len(vertices)


	for idx, v in enumerate(vertices):
		print(v.shape)
		if edges is not None:
			barcodes[idx] = hm.test_barcode(v, edges[idx], k = k, r = r, w = w)[0]
		else: 
			barcodes[idx] = hm.test_barcode(v, k = k, r = r, w = w)[0]
		print("Calculated bar code {0}".format(idx))

	for i in range(M):
		for j in range(M):
			diffs[i,j] = bc.barcode_diff(barcodes[i], barcodes[j], inf = inf)

	dmax = np.max(diffs[diffs < inf/100]) * 1.1
	diffs[diffs > inf/100] = dmax
	plt.imshow(diffs, cmap='gray')

# ------------------------------------------------------------------------

def plot_simplices(simplices, degree, vertices, plt, annotate = False):
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

def get_axis(plt):
	try:
		return plt.gca()
	except:
		return plt

def std_plot(plt):
	ax = get_axis(plt)
	ax.set_xlim(-1.1, 1.1)
	ax.set_ylim(-1.1, 1.1)
	ax.invert_yaxis()
	ax.set_xticks([])
	ax.set_yticks([])


