
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

	# sample = 26 # np.random.randint(50)
	# plt.set_cmap('binary')
	# f, ax = plt.subplots(5,3)

	# for sample in range(5):
	# 	orig, im = get_image('A', sample, size=30)
	# 	ax[sample][0].imshow(orig)
	# 	vertices = get_vertices(im)
	# 	ax[sample][1].scatter(vertices[:,0], vertices[:,1], marker='.')
	# 	ax[sample][1].invert_yaxis()
	# 	simplices, bar_code = hm.process_shape(vertices)
	# 	plot_simplices(simplices, math.inf, vertices, ax[sample][2])
	# 	ax[sample][2].invert_yaxis()

	barcodes = []
	complexes = []
	images = []
	for ltr in 'ABCDE':
		for nmbr in range(5):
			im = get_image(ltr, nmbr, size=30)[1]
			vertices = get_vertices(im)
			compl, bar_code = hm.process_shape(vertices)
			barcodes.append(bar_code)
			complexes.append(compl)
			images.append(compl)

	diff = np.zeros([len(barcodes),len(barcodes)], dtype=float)

	for i, bc1 in enumerate(barcodes):
		for j, bc2 in enumerate(barcodes):
			diff[i,j] = bc.barcode_diff(bc1, bc2)

	# for i, bc1 in enumerate(images):
	# 	for j, bc2 in enumerate(images):
	# 		diff[i,j] = np.sum(np.abs(bc1 - bc2))

	# print(diff)
	# print(barcodes[1])
	# print(barcodes[2])
	# print(bc.barcode_diff(barcodes[0], barcodes[1]))
	# return
	plt.set_cmap('gray')
	plt.imshow(diff)
	# plt.figure()
	# plt.imshow(images[1])
	plt.show()

	# bc.plot_barcode_gant(P)
	
	# Plot result

	def plot_complex():
		"""
		Plots the entire complex. Should work for large sets, but it's slow.
		"""
		plot_simplices(simplices, math.inf, vertices, plt)


	def plot_sequence():
		"""
		Plots the filtration in a number of subplots. Works for really small sets
		"""	
		f, ax = plt.subplots(3,3, sharex=True, sharey=True)
		axs = tuple([e for tupl in ax for e in tupl])
		for idx, subp in enumerate(axs):
			subp.set_title("t = {0}".format(idx * 1))
			plot_simplices(simplices, idx * 1, vertices, subp)




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