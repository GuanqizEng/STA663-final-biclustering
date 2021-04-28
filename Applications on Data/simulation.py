import numpy as np
import SSVD
from SSVD.SSVD import SSVD
from ClusterPlot.ClusterPlot import ClusterPlot

# create the data according to the page 1093
u_tilde = np.r_[np.array([10,9,8,7,6,5,4,3]), 2*np.ones(17), np.zeros(75)]
u = u_tilde/la.norm(u_tilde)
v_tilde = np.r_[np.array([10,-10,8,-8,5,-5]), 3*np.ones(5), -3*np.ones(5), np.zeros(34)]
v = v_tilde/la.norm(v_tilde)
s = 50

# then, we can specify the cluster (to see how this true data looks like)
clusters = np.concatenate((np.ones(8), 2*np.ones(17), 3*np.ones(75)))
ClusterPlot(u, v, s, clusters, 0)

# then, according to the paper, we add some noise to the matrix

#the 'true' background X
X_star = s * u.reshape((-1,1)) @ v.reshape((-1,1)).T

# get the error matrix, and plus it to the true X
np.random.seed(663)
X = X_star + np.random.normal(0, 1, size = X_star.shape)

U, S, V = np.linalg.svd(X)
u = U.T[0]
s = S[0]
v = V.T[0]
ClusterPlot(u.reshape(-1), v.reshape(-1), s, clusters, 0)

# apply it to our function
niter, u, v, s, _, _= SSVD(X, 2, 2)

# it performs pretty well. Almost all the noise is removed from our dataset
ClusterPlot(u.reshape(-1), v.reshape(-1), s, clusters, 0) 
