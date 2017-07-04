from time import time
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
import homology as hm



def draw_complex(letter = 'axis'):
	fig = plt.figure()

	is_dragging = False
	start_time = 0
	end_time = 0
	last_time = 0
	edges = None
	vertices = np.array([])
	first_vertex = None
	last_start_idx = 0


	def clear():
		nonlocal vertices, edges
		print("\n")
		vertices = np.array([])
		edges = None
		plt.cla()
		plt.xlim(-1,1)
		plt.ylim(-1,1)
		plt.draw()
		return


	def dragged(event):
		nonlocal is_dragging, last_time, vertices, first_vertex, last_start_idx
		if is_dragging:
			dt = time() - last_time
			if first_vertex is not None:
				last_start_idx = vertices.shape[0]
				if vertices.size == 0:
					vertices = np.array(first_vertex).reshape(1,2)
				else:
					vertices = np.concatenate((vertices, np.array(first_vertex).reshape(1,2)), axis = 0)
				first_vertex = None
			if dt > .03:
				vertices = np.concatenate((vertices, np.array([event.xdata, event.ydata]).reshape(1,2)), axis = 0)
				last_time = time()


	def clicked(event):
		nonlocal is_dragging, last_time, vertices, start_time, first_vertex
		if event.button == 3:
			return clear()

		first_vertex = [event.xdata, event.ydata]
		is_dragging = True
		start_time = time()
		last_time = time()


	def released(event):
		nonlocal is_dragging, edges, first_vertex, last_start_idx
		is_dragging = False

		if first_vertex is not None and vertices.size > 0:
			kd_tree = spatial.KDTree(vertices)
			_, edge = kd_tree.query(first_vertex, 2)
			edges = np.concatenate((edges, edge.reshape(1,2)), axis = 0)
			edges = hm.remove_duplicate_edges(edges)
		else:
			N = vertices.shape[0]
			if N > last_start_idx:
				new_edges = np.array([np.arange(last_start_idx, N-1), np.arange(last_start_idx, N-1) + 1]).T
				if edges is not None and edges.size > 0:
					edges = np.concatenate((edges, new_edges), axis = 0)
				else:
					edges = new_edges

				edges = hm.remove_duplicate_edges(edges)

		if vertices.size > 0:
			plot_complex(vertices, edges)

	def key_pressed(event):
		if event.key == 'p':
			print(json.dumps({
				'vertices': vertices.tolist(),
				'edges': edges.tolist()
			}))
		else:
			print(event.key)

	def plot_complex(vertices, edges): 
		plt.cla()
		plt.xlim(-1,1)
		plt.ylim(-1,1)
		if vertices.size == 0: return
		plt.scatter(vertices[:,0], vertices[:,1], marker='+')
		for edge in edges:
			plt.plot(vertices[edge,0], vertices[edge,1])

		plt.draw()

	fig.canvas.mpl_connect('button_press_event', clicked)
	fig.canvas.mpl_connect('button_release_event', released)
	fig.canvas.mpl_connect('motion_notify_event', dragged)	
	fig.canvas.mpl_connect('key_press_event', key_pressed)	

	plt.xlim(-1,1)
	plt.ylim(-1,1)
	plt.show()


if __name__ == "__main__":
	draw_complex('A')
