
import math
from scipy import misc, ndimage
import numpy as np
import skimage.morphology as mp
import matplotlib.pyplot as plt
import homology as hm
import bar_code as bc
from copy import deepcopy


def test_curve():
	im = get_image('B', 1, size=30)[1]
	vertices = get_vertices(im)
	(simplices, bar_code, curve, tangents, edges) = hm.process_shape(vertices, test=True)
		
	f, ax = plt.subplots(1,3)

	ax[0].imshow(im)
	plt.set_cmap('hot')
	ax[1].scatter(vertices[:,0], vertices[:,1], marker='.', c=curve)
	ax[1].set_facecolor("#dddddd")
	ax[2].plot(curve)
	plt.show()


def do_letter(im, plt = None):
	vertices = get_vertices(im)
	(simplices, bar_code) = hm.process_shape(vertices)
	return bar_code

def run(): 
	# Get point cloud

	N = 16
	X = .7*np.cos(np.arange(N)*2.0*math.pi/N)
	Y = np.sin(np.arange(N)*2.0*math.pi/N)
	vertices = np.array([X,Y]).T
	(simplices, bar_code, curve, tangents, edges) = hm.process_shape(vertices, test=True)
	# print(bar_code)

	plt.set_cmap('gray')
	# ax = plt.gca()
	# ax.set_facecolor('#ffffaa')

	# plt.scatter(X,Y, marker='.', c=curve)
	verts_degree = simplices[simplices[:,4] == 0,0:3]

	f, ax = plt.subplots(4,4, sharex=True, sharey=True)
	for i in range(4):
		for j in range(4):
			idx = i*4 + j
			plot_simplices(simplices, idx, vertices, ax[i][j], annotate=True)
			ax[i][j].set_title("Curve={0:1.4f}".format(curve[verts_degree[idx,0]]))

	plt.figure()
	bc.plot_barcode_gant(bar_code, plt, annotate=True)

	plt.tight_layout()
	plt.show()



# ---------------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------------

def analyze_image(image): 
	vertices = get_vertices(image)
	simplices, bar_code = hm.process_shape(vertices)


def get_vertices(image): 
	return np.flip(np.array(np.nonzero(image)).T, axis=1)


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
	return original, image

run()