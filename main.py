
# (virtualenv venv)
# source venv/bin/activate


# from Tkinter import Tk, Toplevel, Canvas
import math
from scipy import misc, ndimage
import numpy as np
import skimage.morphology as mp
import matplotlib.pyplot as plt

plt.set_cmap('binary')

def run(): 

	a = np.invert(misc.imread("a.png")) # Image is black on white

	# pc = point_cloud(a)
	a_eroded = mp.thin(a)

	plt.title('Original')
	plt.imshow(a)
	plt.figure()
	plt.title('Thin')
	plt.imshow(a_eroded)

	plt.show()


# image = np.zeros((max_x, max_y))
# image[coordinates] = 1

def point_cloud(a):
	np.array(np.nonzero(a)).T




# ------------------------------------------------------------------------

run()