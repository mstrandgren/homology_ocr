import math
import numpy as np
from scipy.misc import imread, imresize
from scipy import spatial
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


def get_thick_image(letter, number, size=100, sample_size=200):
	sample_idx = ord(letter) - ord('A') + 11
	image = imread("./res/img/Sample{0:03d}/img{0:03d}-{1:03d}.png".format(sample_idx, img_idx))
	image = np.invert(image)[:,:,0] # Make background = 0 and letter > 0
	original = image
	mask = image > 0
	image = image[np.ix_(mask.any(1),mask.any(0))] # Crop
	image = imresize(image, [size, size], interp="nearest")
	image[image>0] = 1 # Make binary

	for i in range(5):
		image = mp.binary_erosion(image)

	vertices = np.flip(np.array(np.nonzero(image)).T, axis=1)
	vertices = vertices * 2.0 / size - 1
	N = vertices.shape[0]
	sample = vertices[np.random.choice(np.arange(N), size=sample_size),:]
	return sample, vertices, image, original

def get_image(letter, number, size, sample_size = 200):
	sample_idx = ord(letter) - ord('A') + 11
	if well_behaved_letters.get(letter) is not None:
		img_idx = well_behaved_letters[letter][number]
	else:
		img_idx = number + 1
	image = imread("./res/img/Sample{0:03d}/img{0:03d}-{1:03d}.png".format(sample_idx, img_idx))
	image = np.invert(image)[:,:,0] # Make background = 0 and letter > 0
	original = image
	mask = image > 0
	image = image[np.ix_(mask.any(1),mask.any(0))] # Crop
	image = imresize(image, [size, size], interp="nearest")
	image[image>0] = 1 # Make binary
	image = mp.skeletonize(image)
	vertices = np.flip(np.array(np.nonzero(image)).T, axis=1)
	vertices = vertices * 2.0 / size - 1
	N = vertices.shape[0]
	sample = vertices[np.random.choice(np.arange(N), size=sample_size),:]
	return sample, vertices, image, original


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


def sparse_sample(point_cloud, N):
	if N == point_cloud.shape[0]:
		return point_cloud

	# Returns indices
	D = spatial.distance.squareform(spatial.distance.pdist(point_cloud))
	idx = 0
	v = set([idx])
	while len(v) < N:
		new_idx = np.argmax(np.min(D[:,list(v)], axis=1))
		# d = D[idx, new_idx]
		v.add(new_idx)
		idx = new_idx

	return list(v)


well_behaved_letters = {
'A': [1,2,3,10,12,15,16,20,24,25],
'B': [2,3,11,12,14,23,26,28,38,39],
'C': [2,3,5,6,7,9,10,11,12,20],
'D': [1,2,3,6,9,11,12,16,19,23],
'E': [2,3,9,10,11,12,13,14,15,16],
'F': [2,3,9,10,11,12,13,15,16,18],
'O': [1,3,4,5,9,12,15,17,20,22],
'U': [1,4,9,12,14,18,23,25,29,30],
'V': [1,2,3,4,5,6,7,8,9,10]
}
