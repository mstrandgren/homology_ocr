import math
import numpy as np
from scipy.misc import imread, imresize
import skimage.morphology as mp



def get_ellipse(N = 16, skew = 0.7):
	X = skew * np.cos(np.arange(N) * 2.0 * math.pi / N)
	Y = np.sin(np.arange(N) * 2.0 * math.pi / N)
	vertices = np.array([X,Y]).T
	edges =  np.array([np.arange(N), np.append(np.arange(N-1) + 1, 0)]).T
	return vertices, edges

# ---------------------------------------------------------------------------------

def analyze_image(image): 
	vertices = get_vertices(image)
	simplices, barcode = hm.process_shape(vertices)	


def get_image(letter, number, size=50):
	sample_idx = ord(letter) - ord('A') + 11
	image = imread("./res/img/Sample{0:03d}/img{0:03d}-{1:03d}.png".format(sample_idx, number + 1))
	image = np.invert(image)[:,:,0] # Make background = 0 and letter > 0
	original = image
	mask = image > 0
	image = image[np.ix_(mask.any(1),mask.any(0))] # Crop
	image = imresize(image, [size, size], interp="bicubic")
	image[image>0] = 1 # Make binary
	image = mp.skeletonize(image) # Skeleton
	vertices = np.flip(np.array(np.nonzero(image)).T, axis=1)
	vertices = vertices * 2.0 / size - 1
	return vertices, image, original

def get_image2(letter, number, size=50):
	image = imread("./res/{0}{1}.png".format(letter.lower(), number + 1))
	image = image[:,:,3] # Make background = 0 and letter > 0
	original = image
	mask = image > 0
	image = image[np.ix_(mask.any(1),mask.any(0))] # Crop
	image = imresize(image, [size, size], interp="bicubic")
	image[image>0] = 1 # Make binary
	image = mp.skeletonize(image) # Skeleton
	vertices = np.flip(np.array(np.nonzero(image)).T, axis=1)
	vertices = vertices * 2.0 / size - 1
	return vertices, image, original