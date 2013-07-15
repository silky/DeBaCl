##############################################################
## Brian P. Kent
## forest.py
## Created: 20130714
## Updated: 20130714
##############################################################

##############
### SET UP ###
##############

"""
Implements methods that act on a group of level set trees (from geom_tree.py or
cd_tree.py.
"""

import utils as utl
import geom_tree as gtree
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import scipy.ndimage as spimg
from scipy import stats

palette = utl.Palette(use='scatter')
n_clr = np.alen(palette.colorset)



###############################
### FOREST CLASS DEFINITION ###
###############################

class Forest(object):
	"""
	Documentation.
	"""

	def __init__(self, trees, group=None, cluster=None):
		self.trees = trees
		self.distances = None
		self.n_tree = len(trees)
		self.group = np.array(group)
		self.cluster = cluster


	def plotForest(self, alpha=0.7, show_group=False):
		"""
		Plot all trees in the forest on top of each other.
		"""

		## Set up the plot        
		fig, ax = plt.subplots()
		ax.set_position([0.07, 0.05, 0.83, 0.93])
		ax.set_xlim((-0.04, 1.04))
		ax.set_xticks([])
		ax.set_xticklabels([])
		ax.set_ylabel("Mass", rotation=270)
		ax.yaxis.grid(color='gray')
		ax.set_ylim(-0.02, 1.03)


		## Loop over trees
		for i, tree in enumerate(self.trees):

			# find the color for tree i
			if show_group == True and self.group is not None:
				clr = palette.colorset[self.group[i] % n_clr]
			else:
				clr = palette.colorset[i % n_clr]

			clr = np.append(clr, alpha)


			# get all roots and root sizes
			ix_root = np.array([u for u, v in tree.nodes.iteritems()
				if v.parent is None])
			n_root = len(ix_root)
			census = np.array([len(tree.nodes[x].members) for x in ix_root],
				dtype=np.float)

			# order the roots and root masses by mass
			seniority = np.argsort(census)[::-1]
			ix_root = ix_root[seniority]
			census = census[seniority]
			n_pt = sum(census)

			# set up initial root silos
			weights = census / n_pt
			intervals = np.cumsum(weights)
			intervals = np.insert(intervals, 0, 0.0)

			# get line segment coordinates
			segments = {}
			splits = {}
			segmap = []
			splitmap = []

			for i, ix in enumerate(ix_root):
				branch = tree.constructBranchMap(ix, (intervals[i],
					intervals[i+1]), scale='alpha', width='mass', sort=True)
				branch_segs, branch_splits, branch_segmap, \
					branch_splitmap = branch
				segments = dict(segments.items() + branch_segs.items())
				splits = dict(splits.items() + branch_splits.items())
				segmap += branch_segmap
				splitmap += branch_splitmap

				verts = [segments[k] for k in segmap]
				lats = [splits[k] for k in splitmap]

				ax.add_collection(LineCollection(verts, colors=np.tile(clr,
					(len(segmap), 1))))
				ax.add_collection(LineCollection(lats, colors=np.tile(clr,
					(len(splitmap), 1))))

		return fig


	def plotCanvas(self, gridsize=100, sigma=1.0):
		"""
		Documentation.
		"""

		## Set up the gridded canvas
		grid = np.linspace(0, 1, gridsize+1)
		z = np.zeros((gridsize, gridsize), dtype=np.int)
		
		
		## Loop over trees
		for tree in self.trees:

			fig, seg, segmap, split, splitmap = tree.plot(form='alpha',
				width='mass')		
		
			# add paint to the canvas
			for s in seg.values():
				x = s[0][0]
				y1 = s[0][1]
				y2 = s[1][1]

				i = min(np.where(grid >= x)[0]) - 1
				j1 = min(np.where(grid >= y1)[0])
				j2 = min(np.where(grid >= y2)[0])

				z[i, j1:(j2+1)] += 1

			for s in split.values():
				y = s[0][1]
				x1 = s[0][0]
				x2 = s[1][0]
	
				j = min(np.where(grid >= y)[0])	
				i1 = min(np.where(grid >= min(x1, x2))[0])
				i2 = min(np.where(grid >= max(x1, x2))[0])
	
				z[i1:(i2-1), j] += 1

		## Draw the canvas
		z = np.rot90(z)
		z_gauss = spimg.filters.gaussian_filter(z, sigma=sigma)
		fig = plt.figure()
		plt.imshow(z_gauss, cmap='jet')
	
		return fig

		

	def distanceMatrix(self, sigma=1.0):
		"""
		Documentation. For now assume this is the canvas distance. Eventually,
		this should dispatch any one of several level set tree distances.
		"""		
		pass
		
		n_tree = len(self.trees)
		d = np.zeros((n_tree, n_tree), dtype=np.float)

		for i in range(n_tree-1):
			for j in range(i+1, n_tree):
				print i, j
				d[i, j] = canvasDistance(self.trees[i], self.trees[j],
					gridsize=100, sigma=sigma)
				
		d = d + d.T
		return d
		
		
	def metaTree(self, sigma, k):
		"""
		Construct a level set tree on the trees in the forest. Use
		pseudo-densities as though the trees have some sort of random
		distribution, just like random curves.
		"""
		
		p = 1  
		self.distances = self.distanceMatrix(sigma=sigma)
	
		rank = np.argsort(self.distances, axis=1)
		ix_nbr = rank[:, 0:k]
		ix_row = np.tile(np.arange(self.n_tree), (k, 1)).T
	
		W = np.zeros(self.distances.shape, dtype=np.bool)
		W[ix_row, ix_nbr] = True
		W = np.logical_or(W, W.T)
		np.fill_diagonal(W, False)
	
		k_nbr = ix_nbr[:, -1]
		k_radius = self.distances[np.arange(self.n_tree), k_nbr]
		fhat = utl.knnDensity(k_radius, self.n_tree, p, k)

		bg_sets, levels = utl.constructDensityGrid(fhat, mode='mass',
			n_grid=None)
		T = gtree.constructTree(W, levels, bg_sets, mode='density',
			verbose=False)		

		return T
		
		
	def knnClassify(self, tree_star, k, gridsize=100, sigma=1.0):
		"""
		Predict the class of an unknown tree based on the canvas distance to
		trees in the forest and the clusters of the forest.
		"""
		
		d = np.zeros((self.n_tree, ), np.float) - 1
		z_star = treeToCanvas(tree_star, gridsize, sigma)

		for i, tree in enumerate(self.trees):
			z = treeToCanvas(tree, gridsize, sigma)
			d[i] = np.sqrt(np.sum((z - z_star)**2))
		
		print d
		ix_vote = np.argsort(d)[:k]
		votes = self.cluster[ix_vote]
		print ix_vote, votes
		votes = votes[votes >= 0]  # discount the background points
		y_star = int(stats.mode(votes)[0])
		
		return y_star



########################
### HELPER FUNCTIONS ###
########################
def canvasDistance(tree1, tree2, gridsize=100, sigma=1.0):
	"""
	Compute a distance between two level set trees by converting them to
	smoothed grayscale images then finding the Euclideaen distance between the
	images.
	"""
	
	## Set up the gridded canvases
	grid = np.linspace(0, 1, gridsize+1)
	z1 = np.zeros((gridsize, gridsize), dtype=np.int)
	z2 = np.zeros((gridsize, gridsize), dtype=np.int)

	fig, seg1, segmap, split1, splitmap = tree1.plot(form='alpha',
		width='mass')
	fig, seg2, segmap, split2, splitmap = tree2.plot(form='alpha',
		width='mass')
		
		
	# Paint canvas 1
	for s in seg1.values():
		x = s[0][0]
		y1 = s[0][1]
		y2 = s[1][1]

		i = min(np.where(grid >= x)[0]) - 1
		j1 = min(np.where(grid >= y1)[0])
		j2 = min(np.where(grid >= y2)[0])

		z1[i, j1:(j2+1)] += 1

	for s in split1.values():
		y = s[0][1]
		x1 = s[0][0]
		x2 = s[1][0]

		j = min(np.where(grid >= y)[0])	
		i1 = min(np.where(grid >= min(x1, x2))[0])
		i2 = min(np.where(grid >= max(x1, x2))[0])

		z1[i1:(i2-1), j] += 1

	z1 = np.rot90(z1)
	z1_gauss = spimg.filters.gaussian_filter(z1, sigma=sigma)


	## Paint canvas 2
	for s in seg2.values():
		x = s[0][0]
		y1 = s[0][1]
		y2 = s[1][1]

		i = min(np.where(grid >= x)[0]) - 1
		j1 = min(np.where(grid >= y1)[0])
		j2 = min(np.where(grid >= y2)[0])

		z2[i, j1:(j2+1)] += 1

	for s in split2.values():
		y = s[0][1]
		x1 = s[0][0]
		x2 = s[1][0]

		j = min(np.where(grid >= y)[0])	
		i1 = min(np.where(grid >= min(x1, x2))[0])
		i2 = min(np.where(grid >= max(x1, x2))[0])

		z2[i1:(i2-1), j] += 1

	z2 = np.rot90(z2)
	z2_gauss = spimg.filters.gaussian_filter(z2, sigma=sigma)
	
	distance = np.sqrt(np.sum((z1 - z2)**2))
	return distance


def treeToCanvas(tree, gridsize=100, sigma=1.0):
	"""
	Convert a tree (defined in terms of line segments) into a grayscale image.
	"""
	
	grid = np.linspace(0, 1, gridsize+1)
	z = np.zeros((gridsize, gridsize), dtype=np.int)

	fig, seg, segmap, split, splitmap = tree.plot(form='alpha',
			width='mass')		
		
	for s in seg.values():
		x = s[0][0]
		y1 = s[0][1]
		y2 = s[1][1]

		i = min(np.where(grid >= x)[0]) - 1
		j1 = min(np.where(grid >= y1)[0])
		j2 = min(np.where(grid >= y2)[0])

		z[i, j1:(j2+1)] += 1

	for s in split.values():
		y = s[0][1]
		x1 = s[0][0]
		x2 = s[1][0]

		j = min(np.where(grid >= y)[0])	
		i1 = min(np.where(grid >= min(x1, x2))[0])
		i2 = min(np.where(grid >= max(x1, x2))[0])

		z[i1:(i2-1), j] += 1

	z = np.rot90(z)
	z_gauss = spimg.filters.gaussian_filter(z, sigma=sigma)

	return z_gauss
	
	
	
	
	
	
	


