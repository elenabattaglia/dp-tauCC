import sys
import os
import numpy as np
import logging as l
import datetime
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari
from algorithms.coclust_DP import CoClust as CoClust_DP

def execute_test_dp(f, V, results, noise=0, eps = 1, init = [], n_iterations=4, verbose = False):
    '''
    Execute CoClust algorithm and write an output file (already existing and open)

    Parameters:
    ----------

    f: output file (open). See CreateOutputFile for a description of the fields.      
    V: tensor
    model: 'CoClust' or 'CC'
    x: target on mode 0
    y: target on mode 1
    z: target on mode 2
    noise: only for synthetic tensors. Amount of noise added to the perfect tensor
    sparsity: sparsity of the tensor (number of entries != 0 / total number of entries)

    '''
    if len(init)!=2:
        raise ValueError('init must be a list of two integers')
    model = CoClust_DP(eps = eps, n_iterations = n_iterations, n_iter_per_mode = 5, k = init[0], l = init[1], verbose = verbose)
    model.fit(V)
  
    l_nmi = []
    l_ari = []
    _assignment = [model._row_assignment, model._col_assignment]
    for i in range(len(results)):
        l_nmi.append(nmi(results[i], _assignment[i], average_method='arithmetic'))
        l_ari.append(ari(results[i], _assignment[i]))


    n = ','.join(str(e) for e in l_nmi)
    a = ','.join(str(e) for e in l_ari)
    init_clusters_x = model.k
    init_clusters_y = model.l

    f.write(f"{eps}, {V.shape[0]}, {V.shape[1]}, {np.max(results[0]) + 1}, {np.max(results[1]) + 1},{noise},{model.tau_x[-1]},{model.tau_y[-1]},{n},{a},{model._n_row_clusters},{model._n_col_clusters},{model.execution_time_},{model._actual_n_iterations},{init_clusters_x},{init_clusters_y},{n_iterations}\n")
    if verbose:
        return model



def CreateOutputFile(partial_name, own_directory = False, date = True, overwrite = False):
    '''
    Create and open a file containing the header described below.

    Parameters:
    ----------
    partial_name: partial name of the file and the directory that will contain the file.
    own_directory: boolean. Default: False.
        If true, a new directory './output/_{partial_name}/aaaa-mm-gg_hh.mm.ss' will be created.
        If flase, the path of the file will be './output/_{partial_name}'.
    date: boolean. Default: True.
        If true, the file name will include datetime.
        If false, it will not.
    overwrite: boolean. Default: False.
        If true, overwrite the existent file (if there exists a file with the same name)
        If false, append the new results.
                

    Output
    ------
    f: file (open). Each record contains the following fields, separated by commas (csv file):
        - model: 'CoClust' or 'CC'
        - dim_x: dimension of the tensor on mode 0
        - dim_y: dimension of the tensor on mode 1
        - x_num_classes: correct number of clusters on mode 0
        - y_num_classes: correct number of clusters on mode 1
        - noise: only for synthetic tensors. Amount of noise added to the perfect tensor
        - tau_x: final tau_{x|y}
        - tau_y: final tau_{y|x}
        - nmi_x: normalized mutual information score on mode 0
        - nmi_y: normalized mutual information score on mode 1
        - ari_x: adjusted rand index on mode 0
        - ari_y: adjusted rand index on mode 1
        - x_num_clusters: number of clusters on mode 0 detected by CoClust
        - y_num_clusters: number of clusters on mode 1 detected by CoClust
        - execution time
        - iter: total number of iterations
        - init_clusters_x: number of initial clusters on mode 0
        - init_clusters_y: number of initial clusters on mode 1

        File name:{partial_name}_aaaa-mm-gg_hh.mm.ss.csv or {partial_name}_results.csv
    dt: datetime (as in the directory/ file name)

    
    '''

    
    dt = f"{datetime.datetime.now()}"
    if own_directory:
        data_path = f"./output/_{partial_name}/" + dt[:10] + "_" + dt[11:13] + "." + dt[14:16] + "." + dt[17:19] + "/"
    else:
        data_path = f"./output/_{partial_name}/"
    directory = os.path.dirname(data_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    new = True
    if date:
        file_name = partial_name + "_" + dt[:10] + "_" + dt[11:13] + "." + dt[14:16] + "." + dt[17:19] + ".csv"
    else:
        file_name = partial_name + '_results.csv'
        if os.path.isfile(data_path + file_name):
            if overwrite:
                os.remove(data_path + file_name)
            else:
                new = False
            
            
    f = open(data_path + file_name, "a",1)
    if new:
        f.write("model,dim_x,dim_y,x_num_classes,y_num_classes,noise,tau_x,tau_y,nmi_x,nmi_y,ari_x,ari_y,x_num_clusters,y_num_clusters,execution_time,iter, init_clusters_x,init_clusters_y,initialization\n")
        
    return f, dt




def CreateLogger(input_level = 'INFO'):
    level = {'DEBUG':l.DEBUG, 'INFO':l.INFO, 'WARNING':l.WARNING, 'ERROR':l.ERROR, 'CRITICAL':l.CRITICAL}
    logger = l.getLogger()
    logger.setLevel(level[input_level])

    return logger
