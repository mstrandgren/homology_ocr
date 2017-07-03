
import math
from scipy import misc, ndimage
from functools import partial
import numpy as np
import skimage.morphology as mp
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy import odr, spatial

import homology as hm
import bar_code as bc
from test_plots import *
import man_data as md

from time import time



def run(): 

	_,_,im = get_image('B', 1)

	fig = plt.figure()
	# plt.imshow(im)

	points = []


	dragging = False
	start = 0
	end = 0
	lastptime = 0
	p = None

	def dragged(event):
		nonlocal dragging, lastptime, p
		if dragging:
			# newp = [event.xdata, event.ydata]
			# dist = np.linalg.norm(np.array(p) - np.array(newp))
			dt = time() - lastptime
			if p is not None:
				points.append(p)
				p = None
			if dt > .1:
				p = [event.xdata, event.ydata]
				points.append(p)
				lastptime = time()


	def clicked(event):
		nonlocal dragging, lastptime, points, start, p
		if event.button == 3:
			plt.cla()
			plt.xlim(-1,1)
			plt.ylim(-1,1)
			plt.draw()
			points = []
			return

		p = [event.xdata, event.ydata]
		dragging = True
		start = time()
		lastptime = time()


	def released(event):
		nonlocal dragging, lastptime, start
		dragging = False
		
		dt = time() - start

		if dt < .2:
			# TODO: add edge
			return
		else:
			p = np.array(points)
			if p.size == 0: return
			N = p.shape[0]
			edges = np.array([np.arange(N), np.append(np.arange(N-1)+1,0)])
			plt.scatter(p[:,0], p[:,1], marker='+')
			for edge in edges:
				plt.plot(p[edge,0], p[edge,1])

		plt.draw()


	fig.canvas.mpl_connect('button_press_event', clicked)
	fig.canvas.mpl_connect('button_release_event', released)
	fig.canvas.mpl_connect('motion_notify_event', dragged)	

	plt.xlim(-1,1)
	plt.ylim(-1,1)
	# mark_vertices('B', 3, 4)
	# mark_edges('B', 3, 4, md.indices)
	plt.show()
	return
	# vertices = get_ellipse(N = 32)	
	# vertices2 = get_image('O', 1, size = 10)[0]

	N = vertices.shape[0]
	k = int(N / 4)
	r = .3
	w = 3

	# print(vertices)

	# plot_edges(vertices, k = k, r = r, w = w)
	# plt.figure()
	# plot_tangent_space(vertices, k = k, r = r, w = .5)

	# f, ax = plt.subplots(1,2)
	# plot_edges(vertices, k = k, r = r, w = w, plt = ax[0])
	# plot_bar_code(vertices, k = k, r = r, w = w, plt = ax[1])

	# plot_curve_color(vertices, k = k, r = r, w = w)
	# plot_filtration(vertices, k = k, r = r, w = w)

	# plt.tight_layout()
	plt.show()
	return


# ---------------------------------------------------------------------------------

def get_ellipse(N = 16, skew = 0.7):
	X = skew * np.cos(np.arange(N) * 2.0 * math.pi / N)
	Y = np.sin(np.arange(N) * 2.0 * math.pi / N)
	vertices = np.array([X,Y]).T
	return vertices

# ---------------------------------------------------------------------------------

def analyze_image(image): 
	vertices = get_vertices(image)
	simplices, bar_code = hm.process_shape(vertices)	


def get_image(letter, number, size=50):
	sample_idx = ord(letter) - ord('A') + 11
	image = misc.imread("./res/img/Sample{0:03d}/img{0:03d}-{1:03d}.png".format(sample_idx, number + 1))
	image = np.invert(image)[:,:,0] # Make background = 0 and letter > 0
	original = image
	mask = image > 0
	image = image[np.ix_(mask.any(1),mask.any(0))] # Crop
	image = misc.imresize(image, [size*2, size*2], interp="nearest")
	image = mp.thin(image) # Skeleton
	image = misc.imresize(image, [size, size], interp="bicubic")
	image[image>0] = 1 # Make binary
	image = mp.thin(image) # Skeleton
	vertices = np.flip(np.array(np.nonzero(image)).T, axis=1)
	vertices = vertices * 2.0 / np.max(vertices) - 1
	return vertices, image, original



run()
