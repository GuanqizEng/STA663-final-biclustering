import numpy as np
import SSVD
from SSVD.SSVD import SSVD
from ClusterPlot.ClusterPlot import ClusterPlot

lungcancer = pd.read_csv('LungCancerData.txt', sep=' ', header = None)
lungcancer = (np.array(lungcancer)).T
lungcancer.shape

lungcancer = pd.read_csv('LungCancerData.txt', sep=' ', header = None)
lungcancer = (np.array(lungcancer)).T
sns.heatmap(lungcancer, vmin=-1, vmax=1, cmap = 'bwr')
pass

clusters = np.concatenate((np.ones(20), 2*np.ones(33-20), 3*np.ones(50-33), 4*np.ones(56-50)))
U, S, V = np.linalg.svd(lungcancer)
u = U.T[0]
s = S[0]
v = V.T[0]
ClusterPlot(u.reshape(-1), v.reshape(-1), s, clusters, 0)

niter, u, v, s, _, _= SSVD(lungcancer, 2, 2)

# it performs pretty well. Almost all the noise is removed from our dataset
ClusterPlot(u.reshape(-1), v.reshape(-1), s, clusters, 0)