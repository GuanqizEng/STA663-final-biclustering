from SSVD.SSVD_numba import SSVD_numba
from SSVD.ClusterPlot import ClusterPlot

u_tilde = np.r_[np.array([10,9,8,7,6,5,4,3]), 2*np.ones(17), np.zeros(75)]
u = u_tilde/la.norm(u_tilde)
v_tilde = np.r_[np.array([10,-10,8,-8,5,-5]), 3*np.ones(5), -3*np.ones(5), np.zeros(34)]
v = v_tilde/la.norm(v_tilde)
s = 50
clusters = np.concatenate((np.ones(8), 2*np.ones(17), 3*np.ones(75)))
X_star = s * u.reshape((-1,1)) @ v.reshape((-1,1)).T
np.random.seed(663)
X = X_star + np.random.normal(0, 1, size = X_star.shape)
niter, u, v, s, _, _= SSVD_numba(X, 2, 2)
ClusterPlot(u.reshape(-1), v.reshape(-1), s, clusters, 0)