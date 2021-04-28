import numpy as np
import seaborn as sns

def ClusterPlot(u, v, s, cluster, drop):
    """Plotting the heatmap for clusters of rank 1 approximation
    
    u, v, s = returned objects from SSVD function
    cluster = the vector of cluster that are known, None if no cluster is known
    drop = the indexes to be dropped
    
    return:
    heatmap for clusters of rank 1 approximation
    """
    
    row_index = np.empty(0, dtype = 'int')
    
    layer1 = s * u.reshape((-1, 1)) @ v.reshape((1, -1))
    #layer1 = s * u @ v.T
    cluster_set = np.unique(cluster)
    
    for i in range(len(cluster_set)):
        index = np.where(cluster == cluster_set[i])[0]
        index_ordered = index[np.argsort(u[index])] # make it ordered
        row_index = np.concatenate((row_index, index_ordered))
        
    col_selected = np.argsort(np.abs(v))[drop:]
    v_selected = v[col_selected]
    col_index = np.argsort(v_selected)
    start = layer1[:,col_selected]
    ax = sns.heatmap(start[np.ix_(row_index, col_index)], vmin=-1, vmax=1, cmap = 'bwr')