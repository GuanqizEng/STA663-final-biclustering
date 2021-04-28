import numpy as np
import SSVD
from SSVD.SSVD import SSVD
from ClusterPlot.ClusterPlot import ClusterPlot

brac = pd.read_csv('bracsample.txt', sep = " ")

# look at the original data, it's pretty messy.
brac = np.array(brac)
sns.heatmap(brac, vmin=-1, vmax=1, cmap = 'bwr')
pass

niter, u, v, s, _, _= SSVD(brac, 2, 2, tol = 1e-06)

# defining the clusters
clusters = np.concatenate((np.ones(30), 2*np.ones(20), 3*np.ones(15), 4*np.ones(10)))
# it performs pretty well. Almost all the noise is removed from our dataset
ClusterPlot(u.reshape(-1), v.reshape(-1), s, clusters, 0)

#aftering the SSVD, the heatmap becomes much more tidy. And we can almost successfully classify them.
#except for the very small stratums, we almost classify all of them.