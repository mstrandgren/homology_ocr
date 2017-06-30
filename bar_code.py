import math
import operator
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

def dim(simplex):
	# Returns scalar 
	return simplex[4]

# def no_zeros(arr):
# 	return arr[np.nonzero(arr)[0]]

def boundary(simplex):
	# Returns set
	k = dim(simplex)
	if k == 0: return set(())
	elif k == 1: return set(simplex[1:3])
	else: 
		assert(false)
		# return set((simplex[:2], simplex[0] + simplex[2], simplex[1:]))

def simplex_add(s1, s2):
	# Returns set
	return s1.symmetric_difference(s2)
	# set(tuple(s1)).symmetric_difference(set(tuple(s2)))

def remove_pivot_rows(simplex, T, marked, youngest_simplex):
	# Returns ndarray
	k = dim(simplex)
	d = boundary(simplex).intersection(marked)

	while len(d) > 0:
		i = youngest_simplex(d)
		# print(sigma_i)
		if T.get(i) is None:
			break
		d = simplex_add(d, T[i])
	return d

def get_bar_code(ordered_simplices, degrees = None):
	"""
	ordered_simplices is a 5 column matrix (i, b1, b2, deg, k)
	degrees is a map from simplex to degree (birth time)

	a simplex is a flat 5-array

	"""
	m_max = ordered_simplices.shape[0]
	k_max = 2
	lookup = ordered_simplices[np.argsort(ordered_simplices[:,0]),:]
	degrees = lookup[:,3]

	# T is map of (simplex, (idx, set))
	T = {}

	def youngest_simplex(chain):
		degree = -1
		youngest = None
		for simplex in chain:
			if degrees[simplex] > degree:
				degree = degrees[simplex]
				youngest = simplex
		return youngest

	# Set of indices (int) of marked simplices
	marked = set()

	# Output, list of 3d vectors [start, stop, k]
	L = []

	for simplex in ordered_simplices:
		d = remove_pivot_rows(simplex, T, marked, youngest_simplex)
		if len(d) == 0:
			marked.add(simplex[0])
		else:
			sigma_i = lookup[youngest_simplex(d),:]
			k = dim(sigma_i)
			T[sigma_i[0]] = d
			L.append((degrees[sigma_i[0]], degrees[simplex[0]], k))

	for simplex_id in marked:
		if T.get(simplex_id) is None:
			k = dim(lookup[simplex_id])
			L.append((degrees[simplex_id], 100, k))

	return L

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

def test():
	# From Edelsbrunner, Letscher & Zomorodian
	ordered_simplices = simplices_from_str("s,t,u,st,v,w,sw,tw,uv,sv,su,uw,tu,tuw,suw,stu,suv,stw")
	# for s in ordered_simplices:
	# 	print("{0}: {1}".format(s, boundary(s)))
	# From Zomorodian & Carlsson
	# ordered_simplices = simplices_from_str("a,b,c,d,ab,bc,cd,ad,ac,abc,acd")		
	# degrees = [0,0,1,1,1,1,2,2,3,4,5]

	degrees = {}
	for idx, s in enumerate(ordered_simplices):
		degrees[s] = idx

	print(ordered_simplices)
	barcode = get_bar_code(ordered_simplices, degrees)
	print(sorted(barcode, key=lambda tup: tup[0]))
	plot_barcode_gant(barcode)


def simplices_from_str(str):
	simplices_txt = str.split(',')
	return simplices_txt


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
	inf = 0
	markers = ('s', '*', 'x')
	for idx, tup in enumerate(barcode):
		plt.plot(tup[:2], [idx, idx], marker=markers[tup[2]], c='k', lw=1, ms=3)
		inf = max(inf, tup[1])

	axes = plt.gca()
	axes.set_xlim([-0.5, inf - 1.5])
	axes.set_xticks(range(0,inf,2))
	plt.show()	



if __name__ == "__main__":
	test()

