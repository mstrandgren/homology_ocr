import math
import operator
import numpy as np
from scipy.optimize import linear_sum_assignment

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
		if T.get(i) is None:
			break
		d = simplex_add(d, T[i])
	return d

def get_bar_code(ordered_simplices, degrees = None, degree_values=None):
	"""
	ordered_simplices is a 5 column matrix (i, b1, b2, deg, k)
	degrees is a lookup table from simplex id to degree (birth time)

	a simplex is a flat 5-array

	"""
	m_max = ordered_simplices.shape[0]
	k_max = 2
	lookup = ordered_simplices[np.argsort(ordered_simplices[:,0]),:]

	if degrees is None:
		degrees = lookup[:,3]

	if degree_values is not None:
		degrees = degree_values[degrees]


	# T is map of (simplex, (idx, set))
	T = {}
	L = np.zeros([m_max, 5], dtype=float)
	N_L = 0

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

	for simplex in ordered_simplices:
		d = remove_pivot_rows(simplex, T, marked, youngest_simplex)
		if len(d) == 0:
			marked.add(simplex[0])
		else:
			sigma_i = lookup[youngest_simplex(d),:]
			k = dim(sigma_i)
			T[sigma_i[0]] = d
			L[N_L,:] = [degrees[sigma_i[0]], degrees[simplex[0]], k, sigma_i[0], simplex[0]]
			N_L += 1

	for simplex_id in marked:
		if T.get(simplex_id) is None:
			k = dim(lookup[simplex_id])
			L[N_L,:] = [degrees[simplex_id], math.inf, k, simplex_id, math.inf]
			N_L += 1

	out = L[:N_L,:]
	return out[np.argsort(out[:,0]),:]

# ------------------------------------------------------------------------
	
def bar_code_diff(bar1, bar2, inf = None):
	"""	
	From "A Barcode Shape Descriptor for Curve Point Cloud Data", 2.4

	bars are Nx2 matrices
	result is and N1xN2 matrix

	infinite distances have a lot of impact, figure out what to do about them
	"""
	if inf is None:
		bmax = max(np.max(bar1[bar1!=math.inf]), np.max(bar2[bar2!=math.inf]))
		inf = bmax * 2


	bar1[bar1==math.inf] = inf
	bar2[bar2==math.inf] = inf

	l1, l2 = bar1.shape[0], bar2.shape[0]
	if l1 > l2:
		bar1, bar2 = bar2, bar1

	l1 = bar1[:,1] - bar1[:,0]
	l2 = bar2[:,1] - bar2[:,0]
	l_max = np.maximum.outer(l1,l2)

	x = np.add.outer(bar1[:,1], -bar2[:,0])
	y = np.add.outer(-bar1[:,0], bar2[:,1])

	is_disjoint = np.logical_or(x<=0,y<=0)
	result = np.zeros(x.shape)
	disjoint = np.add.outer(l1, l2)

	result[is_disjoint] = disjoint[is_disjoint]
	is_overlap = np.logical_and(x<l_max, y<l_max)
	overlap = np.abs(np.add.outer(l1, -l2))
	result[is_overlap] = overlap[is_overlap]

	is_partial = np.logical_and(np.logical_not(is_overlap), np.logical_not(is_disjoint))
	partial = np.abs(x - y)
	result[is_partial] = partial[is_partial]

	np.set_printoptions(suppress=True)
	np.set_printoptions(precision=3, linewidth = 150)
	print(result)
	i,j = linear_sum_assignment(result)
	nonmatched = list(set(range(result.shape[1])).difference(set(j)))
	matched_sum = np.sum(result[i,j])
	nonmatched_sum = np.sum(result[:,nonmatched])
	print(i)
	print(matched_sum)
	print(nonmatched_sum)
	return matched_sum + nonmatched_sum


# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

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


def plot_barcode_gant(barcode, plt, annotate=False):
	bars = barcode[:,:3]
	inf = np.max(bars[bars != math.inf]) + 1
	markers = ('s', '*', 'x')
	for idx, row in enumerate(barcode):
		start,end = row[:2]
		if row[1] == math.inf: end = inf
		plt.plot([start,end], [idx, idx], marker=markers[row[2].astype(int)], c='k', lw=1, ms=3)
		if annotate:
			plt.annotate("{0:4.0f}".format(barcode[idx,3]), (start,idx), horizontalalignment='right')
			plt.annotate("{0:4.0f}".format(barcode[idx,4]), (end,idx), horizontalalignment='left')


	try: 
		axes = plt.gca()
		axes.set_xlim([-0.5, inf - 0.5])
		axes.set_xticks(range(0,math.ceil(inf),2))
	except: 
		plt.set_xlim([-0.5, inf - 0.5])
		plt.set_xticks(range(0,math.ceil(inf),2))



def test(): 
	test_bars = [
	[(0,2), (3,5)], # No overlap, l=4
	[(0,4), (2,6)], # Partial overlap, l=4
	[(0,4), (2,3)], # Full overlap, l=3
	[(0,2), (0,5)], # l=3
	[(0,5), (3,5)], # l=3
	[(1,2), (0,2)], # l=1
	[(1,4), (0,math.inf)], # l=inf
	[(1,2), (5,math.inf)], # l=inf
	[(1,math.inf), (5,math.inf)] # l=inf
	]

	bar1 = np.zeros([len(test_bars),2])
	bar2 = np.zeros([len(test_bars) + 5,2])
	for idx,b in enumerate(test_bars):
		bar1[idx,:] = b[0]
		bar2[idx,:] = b[1]

	for idx in range(5):
		bar2[idx + len(test_bars), :] = [2,4]

	print(bar1)
	print(bar2)

	print(barcode_diff(bar2, bar1))


if __name__ == "__main__":
	test()