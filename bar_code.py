import math
import operator
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

def dim(simplex):
	# Returns scalar 
	return np.nonzero(simplex)[0].size - 1

def no_zeros(arr):
	return arr[np.nonzero(arr)[0]]

def is_empty(t_i):
	# Returns true if t_i is all zeros
	return np.nonzero(t_i)[0].size == 0

def boundary(simplex):
	# Returns ndarray
	k = dim(simplex)
	if k == 0: return np.array([], dtype=int)
	return no_zeros(simplex)

def simplex_add(s1, s2):
	# Returns ndarray
	return no_zeros(np.setxor1d(s1, s2))

def remove_pivot_rows(simplex, T, marked):
	# Returns ndarray
	k = dim(simplex)
	d = np.intersect1d(boundary(simplex), list(marked))

	while d.size > 0:
		i = np.max(d) - 1 # 1-index
		if T[i] is None:
			break
		q = T[i][0]
		d = simplex_add(d, T[i][1])
	return d

def get_bar_code(ordered_simplices, degrees = None):
	m_max, k_max = ordered_simplices.shape

	if degrees is None:
		degrees = np.arange(len(ordered_simplices))

	# T is list of (int, ndarray), or None
	T = [None] * m_max 
	
	# Set of indices (int) of marked simplices
	marked = set()

	# Output, list of 3d vectors [start, stop, k]
	L = []

	for j in range(m_max):
		simplex = ordered_simplices[j,:]
		d = remove_pivot_rows(simplex, T, marked)
		if d.size == 0:
			marked.add(j + 1)
		else:
			i = np.max(d) - 1 # 1-index
			k = dim(ordered_simplices[i,:])
			T[i] = (j,d)
			L.append([degrees[i], degrees[j], k])

	for j in marked:
		simplex = ordered_simplices[j - 1,:]
		if T[j - 1] is None:
			k = dim(simplex)
			L.append([degrees[j - 1], m_max + 1, k])

	Ln = np.array(L)
	return Ln[Ln[:,0].argsort()]


# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

def test():
	# From Edelsbrunner, Letscher & Zomorodian
	ordered_simplices = simplices_from_str("s,t,u,st,v,w,sw,tw,uv,sv,su,uw,tu,tuw,suw,stu,suv,stw")
	# From Zomorodian & Carlsson
	# ordered_simplices = simplices_from_str("a,b,c,d,ab,bc,cd,ad,ac,abc,acd")		
	# degrees = [0,0,1,1,1,1,2,2,3,4,5]
	print(ordered_simplices)
	barcode = get_bar_code(ordered_simplices)
	print(barcode)
	plot_barcode_gant(barcode)


def simplices_from_str(str):
	simplices_txt = str.split(',')

	def translate_simplex(readable_simplex):
		idx, simplex = readable_simplex
		out = np.zeros(3, dtype=int)
		if len(simplex) == 1:
			out[0] = idx + 1
		elif len(simplex) == 2:
			out[0] = simplices_txt.index(simplex[0]) + 1
			out[1] = simplices_txt.index(simplex[1]) + 1
		elif len(simplex) == 3:
			out[0] = simplices_txt.index(simplex[:2]) + 1
			out[1] = simplices_txt.index(simplex[1:]) + 1
			out[2] = simplices_txt.index(simplex[0] + simplex[2]) + 1
		return out.reshape(1,3)

	return np.concatenate(list(map(translate_simplex, enumerate(simplices_txt))), axis=0)

def plot_barcode_scatter(barcode):
	inf = np.max(barcode[:,1]) + 1
	axes = plt.gca()
	axes.set_xlim([0,inf])
	axes.set_ylim([0,inf])
	axes.set_xticks(range(0,inf,2))
	axes.set_yticks(range(0,inf,2))
	axes.minorticks_on()
	axes.grid(True, which='minor', color='#eeeeee', zorder=1)
	axes.grid(True, which='major', color='#999999', zorder=0)
	plt.scatter(barcode[:,0], barcode[:,1], marker=".", c=barcode[:,2]/2.0, zorder=2)
	plt.plot([0,inf],[0,inf], lw=1, color="lightgray")
	plt.show()

def plot_barcode_gant(barcode):
	inf = np.max(barcode[:,1]) + 1
	markers = ('s', '*', 'x')
	for i in range(barcode.shape[0]):
		plt.plot(barcode[i, 0:2], [i, i], marker=markers[barcode[i, 2]], c='k', lw=1, ms=3)

	axes = plt.gca()
	axes.set_xlim([-0.5, inf - 1.5])
	axes.set_xticks(range(0,inf,2))
	plt.show()	



if __name__ == "__main__":
	test()

