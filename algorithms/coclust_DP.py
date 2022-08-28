import logging
from random import choice
from random import randint
from time import time
from typing import List

import numpy as np
import scipy
from scipy.sparse import issparse
from sklearn.base import BaseEstimator
from sklearn.base import ClusterMixin
from sklearn.base import TransformerMixin
from sklearn.cluster import KMeans
from sklearn.utils import check_array
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score as ari
from algorithms.CreateMatrix import CreateMatrix, replaceRandomBase


class CoClust(BaseEstimator, ClusterMixin, TransformerMixin):
    """ CoStar algorithm (Ienco, 2013).

    CoStar is an algorithm created to deal with multi-view data.
    It finds automatically the best number of row / column clusters.

    Parameters
    ------------

    n_iterations : int, optional, default: 300
        The number of iterations to be performed.

    min_n_row_clusters : int, optional, default: 2,
        The number of clusters for rows, should be at least 2

    init : {'standard_with_merge', 'standard_without_merge', 'random', 'kmeans'}, optional, default: 'standard_without_merge'
        The initialization methods:

        'standard_with_merge' : the initial clusters are discrete (one item per cluster), but identical
            rows are merged into one single cluster

        'standard_without_merge' : the initial clusters are discrete (one item per cluster)

        'random' : the initial clusters are random and are less than the number of elements

        'kmeans' : the initial cluster for both rows and columns are chosen by an execution of kmeans.
                    the number of cluster is equal to number_of_elements * scaling_factor

    scaling_factor : float, optional, default: 0,01
        The scaling factor to use to define the number of cluster desired in case of 'kmeans' initialization method.

    Attributes
    -----------

    rows_ : array, length n_rows
        Results of the clustering on rows. `rows[i]` is `c` if
        row `i` is assigned to cluster `c`. Available only after calling ``fit``.

    columns_ : array, length n_columns
        Results of the clustering on columns. `columns[i]` is `c` if
        column `i` is assigned to cluster `c`. Available only after calling ``fit``.

    execution_time_ : float
        The execution time.

    References
    ----------

    * Ienco, Dino, et al., 2013. `Parameter-less co-clustering for star-structured heterogeneous data.`
        Data Mining and Knowledge Discovery 26.2: 217-254

    """

    def __init__(self, eps = 1,n_iterations=5, n_iter_per_mode = 5, initialization= 'sphere', k = 0, l = 0, row_clusters = np.zeros(1), col_clusters = np.zeros(1), initial_prototypes = np.zeros(1), verbose = False):
        """
        Create the model object and initialize the required parameters.
        
        :type eps: float
        :param eps: privacy budget
        :type n_iterations: int
        :param n_iterations: the max number of iterations to perform
        :type n_iter_per_mode: int
        :param n_iter_per_mode: the max number of sub-iterations for eac iteration and each mode
        :type initialization: string
        :param initialization: the initialization method, default = 'sphere'
        :type k: int
        :param k: number of clusters on rows. It is only needed if initialization in {'random', 'extract_centroids'}. In 'random' k and l can be equal to 0 . 
        :type l: int
        :param l: number of clusters on columns. It is only needed if initialization in {'random', 'extract_centroids'}. In 'random' k and l can be equal to 0 . 
        :type row_clusters: np.array of integers
        :param row_clusters: initial cluster assignments of rows. Only needed if initialization in {}
        :type col_clusters: np.array of integers
        :param col_clusters: initial cluster assignments of columns. Only needed if initialization in {}
        :type initial_prototypes: np.array of integers
        :param initial_prototypes: ???
        :type verbose: boolean
        :param verbose: if True, it prints details of the computations
        """

        self.eps = eps
        self.n_iterations = n_iterations
        self.n_iter_per_mode = n_iter_per_mode
        self.initialization = initialization
        self.k = k
        self.l = l
        self.row_clusters = row_clusters
        self.col_clusters = col_clusters
        self.initial_prototypes = initial_prototypes
        self.verbose = verbose

        # these fields will be available after calling fit
        self.rows_ = None
        self.columns_ = None

        np.seterr(all='ignore')

    def _init_all(self, V):
        """
        Initialize all variables needed by the model.

        :param V: the dataset
        :return:
        """
        # verify that all matrices are correctly represented
        # check_array is a sklearn utility method
        self._dataset = None #a differenza della versione non privata, il dataset non è normalizzato

        self._dataset = check_array(V, accept_sparse='csr', dtype=[np.float64, np.float32, np.int32])

        self._csc_dataset = None
        if issparse(self._dataset):
            # transform also to csc
            self._csc_dataset = self._dataset.tocsc()
            

        # the number of documents and the number of features in the data (n_rows and n_columns)
        self._n_documents = self._dataset.shape[0]
        self._n_features = self._dataset.shape[1]


        # the number of row/ column clusters
        self._n_row_clusters = 0
        self._n_col_clusters = 0


        # a list of n_documents (n_features) elements
        # for each document (feature) d contains the row cluster index d is associated to
        self._row_assignment = np.zeros(self._n_documents)
        self._col_assignment = np.zeros(self._n_features)


        # computation time
        self.execution_time_ = 0

        self._tot = np.sum(self._dataset)
        #self._dataset = self._dataset/self._tot
        self.tau_x = []
        self.tau_y = []
        

        if self.initialization == 'sphere':
            self._sphere_initialization()
        else:
            raise ValueError("The only valid initialization method is 'sphere'")


        logging.debug("[INFO] Row's initialization: {0}".format(list(self._row_assignment)))
        logging.debug("[INFO] Col's initialization: {0}".format(list(self._col_assignment)))


    def fit(self, V, y=None):
        """
        Fit CoClust to the provided data.

        Parameters
        -----------

        V : array-like or sparse matrix;
            shape of the matrix = (n_documents, n_features)

        y : unused parameter

        Returns
        --------

        self

        """

        # Initialization phase
        self._init_all(V)

        #_, _ = self._init_contingency_matrix(0)  #qui a differenza del metodo non DP viene aggiornato self._T (cioè la matrice giusta, senza rumore) e poila funzione restituisce dataset e cont_table con rumore
        #self._noisy_T = np.copy(self._T)
        tau_x, tau_y = self.compute_taus()
        self.tau_x.append(tau_x)
        self.tau_y.append(tau_y)

        start_time = time()

        # Execution phase
        self._actual_n_iterations = 0
        actual_n_iterations = 0
        eps = self.eps/((self.n_iterations+1)*2)
        eps1 = eps
        
        while actual_n_iterations < self.n_iterations:
            actual_iteration_x = 0
            cont = True
            h,t = self._estimate_parameters(0,eps1)
            if actual_n_iterations == 0:
                self._exponential_assignment(0,h,eps1, init = True)
            else:
                self._exponential_assignment(0,h,eps1)
                self._check_clustering(1)
            while cont:
                logging.debug("[INFO] ##############################################\n" +
                             "\t\t### Iteration {0}".format(self._actual_n_iterations))
                logging.debug("[INFO] Row assignment: {0}".format(self._row_assignment))
                logging.debug("[INFO] Col assignment: {0}".format(self._col_assignment))


                # perform a move within the rows partition                
                #cont = self._perform_row_move((1-h)*eps1/t)
                #print( '############################' )
                #self._perform_col_move()
                #print( '############################' )
                d = self._update_dataset(0)
                self._init_contingency_matrix(d,0,(1-h)*eps1/t)
                # righe
                
                #self._check_clustering(0)
                
                cont = False

                actual_iteration_x += 1
                self._actual_n_iterations +=1 
                
                
                logging.debug("[INFO] # row clusters: {0}; # col clusters: {1}".format(self._n_row_clusters, self._n_col_clusters))
                if actual_iteration_x == t:
                    cont = False

##                if self.verbose:
##                    _, _ = self._init_contingency_matrix(0)
##                    tau_x, tau_y = self.compute_taus()
##                    self.tau_x.append(tau_x)
##                    self.tau_y.append(tau_y)

##            if actual_iteration_x < t:
##                eps1 = eps - (1-h)*eps1/t*(actual_iteration_x -t)

            actual_iteration_y = 0
            h,t = self._estimate_parameters(1,eps1)
            self._exponential_assignment(1,h,eps1)
            #self._check_clustering(0)
            cont = True
            while cont:
                logging.debug("[INFO] ##############################################\n" +
                             "\t\t### Iteration {0}".format(self._actual_n_iterations))
                logging.debug("[INFO] Row assignment: {0}".format(self._row_assignment))
                logging.debug("[INFO] Col assignment: {0}".format(self._col_assignment))

                # perform a move within the rows partition

                #cont = self._perform_col_move((1-h)*eps1/t)
                #print( '############################' )
                #self._perform_col_move()
                #print( '############################' )
                d = self._update_dataset(1)
                self._init_contingency_matrix(d,1,(1-h)*eps1/t)
                
                cont = False

                actual_iteration_y += 1
                self._actual_n_iterations +=1 
                
                logging.debug("[INFO] # row clusters: {0}; # col clusters: {1}".format(self._n_row_clusters, self._n_col_clusters))
                if actual_iteration_y == t:
                    cont = False

##                if self.verbose:
##                    self._T = self._init_contingency_matrix(0)[1]
##                    tau_x, tau_y = self.compute_taus()
##                    self.tau_x.append(tau_x)
##                    self.tau_y.append(tau_y)
##            if actual_iteration_y < t:
##                eps1 = eps - (1-h)*eps1/t*(actual_iteration_y -t)

                
            if (actual_iteration_x == 1) and (actual_iteration_y == 1) and (t > 1):
                actual_n_iterations = self.n_iterations
            else:
                actual_n_iterations += 1

        # colonne
        m = np.argmax(np.sum(self._noisy_T, axis = 0))
        self._noisy_T[:, np.sum(self._noisy_T, axis = 0)< np.sum(self._noisy_T)/(5*self._n_col_clusters)] =0
        rr = [j for j in range(self._n_col_clusters) if np.sum(self._noisy_T, axis = 0)[j] == 0]
                #print(len(rr))
        self._noisy_T = np.delete(self._noisy_T, rr, axis = 1)
                #print(np.shape(self._noisy_T))
                
        for g in rr:
            self._col_assignment[self._col_assignment== g] = m
        self._check_clustering(1)
                #print(self._n_col_clusters)
        self._exponential_assignment(0,.9,eps1)
        self._init_contingency_matrix(d,1,.1*eps1/t)
                #print(self._n_row_clusters, self._n_col_clusters)

        m = np.argmax(np.sum(self._noisy_T, axis = 1))
        self._noisy_T[np.sum(self._noisy_T, axis = 1)< np.sum(self._noisy_T)/(5*self._n_row_clusters)] =0
        rr = [j for j in range(self._n_row_clusters) if np.sum(self._noisy_T, axis = 1)[j] == 0]
        self._noisy_T = np.delete(self._noisy_T, rr, axis = 0)
                
        for g in rr:
            self._row_assignment[self._row_assignment== g] = m
            self._check_clustering(0)
                #self._exponential_assignment(1,1,eps1)
        
        self._perform_row_move(1)

        

        end_time = time()
        if not self.verbose:
            tau_x, tau_y = self.compute_taus()
            self.tau_x.append(tau_x)
            self.tau_y.append(tau_y)

        execution_time = end_time - start_time
        #self._tau = self._compute_tau()
        logging.info('#####################################################')
        logging.info("[INFO] Execution time: {0}".format(execution_time))

        # clone cluster assignments and transform in lists
        self.rows_ = np.copy(self._row_assignment).tolist()
        self.columns_ = np.copy(self._col_assignment).tolist()
        self.execution_time_ = execution_time

        logging.info("[INFO] Number of row clusters found: {0}".format(self._n_row_clusters))
        logging.info("[INFO] Number of column clusters found: {0}".format(self._n_col_clusters))

        return self

     

##    def _sphere_initialization(self):  #### DA FARE SULLA FALSARIGA DEL METODO DELLE SFERE PER IL k-means PRIVATO
##
##        if (self.k > self._n_documents) or (self.l > self._n_features):
##            raise ValueError("The number of clusters must be <= the number of objects, on both dimensions")
##        if self.k == 0 :
##            self._n_row_clusters = np.random.choice(self._n_documents)
##        else:
##            self._n_row_clusters = self.k
##        if self.l == 0:
##            self._n_col_clusters = np.random.choice(self._n_features)
##        else:
##            self._n_col_clusters = self.l
##
##        
##        h = int(self._n_features/self.k)+1
##        self._n_row_clusters = self.k
##
##
##        T = np.zeros((self.k,self._n_features))
##        p = np.ones(self._n_features) * 1/self._n_features
##
##        for i in range(self.k):
##            a = np.random.choice(self._n_features, h,p=p, replace = False)
##            T[i,a] =1
##            p[a] = 0.00001
##            x = [b for b in range(self._n_features) if b not in a]
##            p[x] = (1 - h*0.00001)/len(x)
##
##        T = replaceRandomBase(T,.1)[0]  #aggiunge rumore
##        dataset = self._dataset/self._tot
##
##        for i in range(self._n_documents):
##            all_tau = np.sum(np.true_divide(dataset[i],np.sum(dataset,0))*T, 1) - np.sum(dataset[i])*np.sum(T, 1)
##            max_tau = np.max(all_tau)
##                
##            e_max = np.where(max_tau == all_tau)[0][0]
##            self._row_assignment[i] = e_max
##
##        j = int(self._n_documents/self.l)+1
##        self._n_col_clusters = self.l
##        
##        T = np.zeros((self.l,self._n_documents))
##        p = np.ones(self._n_documents) * 1/self._n_documents
##        
##        for i in range(self.l):
##            a = np.random.choice(self._n_documents, j,p=p, replace = False)
##            T[i,a] =1
##            p[a] = 0.00001
##            x = [b for b in range(self._n_documents) if b not in a]
##            p[x] = (1 - j*0.00001)/len(x)
##
##        T = replaceRandomBase(T,.1)[0]  #aggiunge rumore
##        self._noisy_T = T.T
##        dataset = self._dataset.T/self._tot
##        for i in range(self._n_features):
##            all_tau = np.sum(np.true_divide(dataset[i],np.sum(dataset,0))*T, 1) - np.sum(dataset[i])*np.sum(T, 1)
##            max_tau = np.max(all_tau)
##            
##            e_max = np.where(max_tau == all_tau)[0][0]
##            self._col_assignment[i] = e_max
##
##        self._check_clustering(0)
##        self._check_clustering(1)

    def _sphere_initialization(self):  #### DA FARE SULLA FALSARIGA DEL METODO DELLE SFERE PER IL k-means PRIVATO

        if (self.k > self._n_documents) or (self.l > self._n_features):
            raise ValueError("The number of clusters must be <= the number of objects, on both dimensions")
        if self.k == 0 :
            self._n_row_clusters = np.random.choice(self._n_documents)
        else:
            self._n_row_clusters = self.k
        if self.l == 0:
            self._n_col_clusters = np.random.choice(self._n_features)
        else:
            self._n_col_clusters = self.l


        V,x,y = CreateMatrix(self._n_documents, self._n_features, self.k, self.l, .15)
        #self._n_row_clusters = self._n_documents
        self._n_col_clusters = self.l
        #self._row_assignment = np.arange(self._n_documents)
        #self._row_assignment = x
        #w = np.random.permutation(V.shape[1])
        #V = V[:, w]
        #y = y[w]

        new_t = np.zeros((self._n_row_clusters, self._n_col_clusters), dtype=float)
        t = np.zeros((self._n_documents, self._n_col_clusters), dtype = float)

        for i in range(self._n_col_clusters):
            t[:,i] = np.sum(V[:,y == i], axis = 1)
        self._noisy_T = np.copy(t)

        self._n_row_clusters = self.k
        
        for i in range(self._n_row_clusters):
            new_t[i] = np.sum(t[x == i], axis = 0)


        self._T = np.copy(new_t)


    def _check_clustering(self, dimension):
        if dimension == 1:
            k = len(set(self._col_assignment))
            if k == 1:
                i = np.random.choice(self._n_features, max(1,int(self._n_features/100)))
                self._col_assignment[i] = np.max(self._col_assignment[i]) +1
                k = 2
            h = [j for j in range(k) if j not in set(self._col_assignment)]
            p = 0
            for i in set(self._col_assignment):
                if i >=k:
                    self._col_assignment[self._col_assignment==i] = h[p]
                    p +=1
            self._n_col_clusters = k

        elif dimension == 0:
            k = len(set(self._row_assignment))
            if k == 1:
                i = np.random.choice(self._n_documents, max(1,int(self._n_documents/100)))
                self._row_assignment[i] = np.max(self._row_assignment[i]) +1
                k = 2
            h = [j for j in range(k) if j not in set(self._row_assignment)]
            p = 0
            for i in set(self._row_assignment):
                if i >=k:
                    self._row_assignment[self._row_assignment==i] = h[p]   
                    p +=1
            self._n_row_clusters = k        
            
    def _init_contingency_matrix(self, dataset, dimension, eps = 0):
        """
        Initialize the T contingency matrix
        of shape = (n_row_clusters, n_col_clusters)

        :return:
        """
        
        logging.debug("[INFO] Compute the contingency matrix...")

        # dense case
        #dataset = self._update_dataset(dimension)

        new_t = np.zeros((self._n_row_clusters, self._n_col_clusters), dtype=float)
        if dimension == 0:   # qui dataset ha dimensione n_rows X n_col_clusters
            for i in range(self._n_row_clusters):
                new_t[i] = np.sum(dataset[self._row_assignment == i], axis = 0)
        else:
            for i in range(self._n_col_clusters):  # qui dataset ha dimensione n_cols X n_row_clusters
                new_t[:,i] = np.sum(dataset[:,self._col_assignment == i], axis = 1)

        self._T = np.copy(new_t)
        if eps > 0:
            #b = (self.n_iterations**2)/self.eps  ##### SISTEMARE IL RUMORE #####
            noise = np.random.laplace(0, 1/eps, new_t.shape)
            new_t += noise
            for i in range(self._n_row_clusters):
                for j in range(self._n_col_clusters):
                    if new_t[i,j] < 0:
                        new_t[i,j] = 0

        self._noisy_T = np.copy(new_t)
        logging.debug("[INFO] End of contingency matrix computation...")


    def _update_dataset(self, dimension):
        if dimension == 0:
            new_t = np.zeros((self._n_documents, self._n_col_clusters), dtype = float)

            for i in range(self._n_col_clusters):
                new_t[:,i] = np.sum(self._dataset[:,self._col_assignment == i], axis = 1)
            
            
        else:
            new_t = np.zeros((self._n_row_clusters, self._n_features), dtype = float)
            for i in range(self._n_row_clusters):
                new_t[i] = np.sum(self._dataset[self._row_assignment == i], axis = 0)
        return new_t


    def _perform_row_move(self, eps):
        """
        Perform a single move to improve the partition on rows.

        :return:
        """
        d = self._update_dataset(0)
        dataset = d/self._tot
        T = self._noisy_T/np.sum(self._noisy_T)
        moves = 0
        for i in range(self._n_documents):
            all_tau = np.sum(dataset[i]*np.true_divide(T,np.sum(T,0)), 1) - np.sum(dataset[i])*np.sum(T, 1)
            max_tau = np.max(all_tau)
            #print('max_tau ', max_tau)

            if (max_tau >=0):
                equal_solutions = np.where(max_tau == all_tau)[0]
                e_min = equal_solutions[0]
                if e_min != self._row_assignment[i]:
                    moves += 1
##                if ((e_min == self._row_assignment[i]) and (self._actual_n_iterations==0)):
##                    all_tau[i] = 0
##                    max_tau = np.max(all_tau)
##                    e_min = np.where(max_tau == all_tau)[0][0]
                    
            else:
                e_min = self._n_row_clusters
                self._n_row_clusters += 1
                if (np.sum(self._row_assignment == self._row_assignment[i])>1):
                    moves +=1
            self._row_assignment[i] = e_min

        self._check_clustering(0)
        #self._init_contingency_matrix(d,0,eps)

        if self.verbose:
            tx, ty = self.compute_taus()
            print(f"iteration {self._actual_n_iterations}, moving rows, n_clusters: ({self._n_row_clusters}, {self._n_col_clusters}), tau: {(tx, ty)}")
            self.tau_x.append(tx)
            self.tau_y.append(ty)
        if moves ==0:
            return True
        else:
            return True


    def _perform_col_move(self,eps):
        """
        Perform a single move to improve the partition on rows.

        :return:
        """
        d = self._update_dataset(1)
        dataset = d.T/self._tot
        T = self._noisy_T.T/np.sum(self._noisy_T)

        moves = 0
        for i in range(self._n_features):
            all_tau = np.sum(dataset[i]*np.true_divide(T,np.sum(T,0)), 1) - np.sum(dataset[i])*np.sum(T, 1)
            max_tau = np.max(all_tau)
            
            if (max_tau >=0):
                equal_solutions = np.where(max_tau == all_tau)[0]
                e_min = equal_solutions[0]
                if e_min != self._col_assignment[i]:
                    moves += 1
##                if ((e_min == self._col_assignment[i]) and (self._actual_n_iterations==0)):
##                    all_tau[i] = 0
##                    max_tau = np.max(all_tau)
##                    e_min = np.where(max_tau == all_tau)[0][0]
            else:
                e_min = self._n_col_clusters
                self._n_col_clusters += 1
                if (np.sum(self._col_assignment == self._col_assignment[i])>1):
                    moves +=1
            self._col_assignment[i] = e_min

        self._check_clustering(1)
        self._init_contingency_matrix(d,1,eps)

        if self.verbose:
            tx, ty = self.compute_taus()
            print(f"iteration {self._actual_n_iterations}, moving columns, n_clusters: ({self._n_row_clusters}, {self._n_col_clusters}), tau: {(tx, ty)}")
            self.tau_x.append(tx)
            self.tau_y.append(ty)
        if moves ==0:
            return True
        else:
            return True


    def _compute_delta_u(self, dimension): 
##        if dimension == 1:
##            T = self._noisy_T.T/np.sum(self._noisy_T)
##        else:
##            T = self._noisy_T/np.sum(self._noisy_T)
##        X = np.zeros(T.shape)
##        for i in range(T.shape[1]):
##            X[:,i] = np.sum(T, axis = 1)
##
##        return np.max(np.nan_to_num(abs(X- T/np.sum(T,0)), nan = 0))
        #dataset, T = self._init_contingency_matrix(0)
        if dimension == 1:
            T = np.copy(self._noisy_T).T
        else:
            T = np.copy(self._noisy_T)
        
        S = np.repeat(np.sum(T, axis = 1).reshape((-1,1)), repeats = T.shape[1], axis = 1)
        B = T/np.sum(T, axis = 0) - S/np.sum(T)

        A = np.zeros(B.shape[1])
        for j in range(B.shape[1]):
            A[j] = np.max(B[:,j]) - np.min(B[:,j])
        return np.max(A)



    def _compute_varM(self, dimension):
        
        if dimension == 1:
            d = self._noisy_T.T
            T = self._noisy_T.T/np.sum(self._noisy_T)
            k = self._n_col_clusters
        else:
            d = np.copy(self._noisy_T)
            T = self._noisy_T/np.sum(self._noisy_T)
            k = self._n_row_clusters
        sum_per_row = np.sum(T, axis = 1)
        sum_per_col = np.sum(T, axis = 0)
        sum_per_col[sum_per_col==0] =1
        l = np.zeros(k)

        for i in range(k):
            all_tau = np.sum(d[i]*np.true_divide(T,np.sum(T,0)), 1) - np.sum(d[i])*np.sum(T, 1)
            max_tau = np.max(all_tau)
            all_tau[all_tau == max_tau]=-np.inf
            h = np.argmax(all_tau)
            a = np.sum(d[i]/(k*sum_per_row[i])*(T[i] - T[h])/sum_per_col)
            b = np.sum(T[i] - T[h])
            l[i] = sum_per_row[i]*(a-np.sum(d[i]/(k*sum_per_row[i]))*b)

        return np.sum(l)

    def _estimate_parameters(self, dimension, eps):
        delta_u = self._compute_delta_u(dimension)
        varM = self._compute_varM(dimension)
        if dimension == 0:
            k = self._n_row_clusters
        else:
            k = self._n_col_clusters

        h = 2*delta_u*np.log(k+2.99)/(eps*varM)
        if h < 0.01:
            h = 0.01
        if h <1:
            t = np.sum(self._noisy_T)*(1-h)*eps/(self._n_row_clusters * self._n_col_clusters*np.log(self._n_row_clusters * self._n_col_clusters/.05))*.05
            if t > self.n_iter_per_mode:
                t = self.n_iter_per_mode
                h = 1 - self._n_row_clusters * self._n_col_clusters*np.log(self._n_row_clusters * self._n_col_clusters/.05)/(.05*np.sum(self._noisy_T)*eps)
                if h > .9:
                    h =.9
            #return h,max(int(t),1)
            
                       
        h = .9
        t = np.sum(self._noisy_T)*(1-h)*eps/(self._n_row_clusters * self._n_col_clusters*np.log(self._n_row_clusters * self._n_col_clusters/.05))*.05
        t = 1 #riga da commentare
        return h, max(int(t),1)

    def _exponential_assignment(self, dimension, h, eps, init = False):
        delta_u = self._compute_delta_u((dimension + 1)%2)
        self.d = dimension

        if dimension == 0:
            if not init:
                dataset = self._update_dataset(1).T
            else:
                dataset = self._dataset.T
            T = self._noisy_T.T/np.sum(self._noisy_T)
            k = self._n_features
            c = self._n_col_clusters

        else:
            dataset = self._update_dataset(0)
            T = self._noisy_T/np.sum(self._noisy_T)              
            k = self._n_documents
            c = self._n_row_clusters

        a = np.zeros(k)
        sum_per_row = np.sum(T, axis = 1)
        sum_per_col = np.sum(T, axis = 0)
        sum_per_col[sum_per_col==0] =1
        for i in range(k):
            all_tau = np.sum(dataset[i]*np.true_divide(T,sum_per_col), 1) - np.sum(dataset[i])*sum_per_row
            if np.max(all_tau)>100:
                delta = max(all_tau) - 100
                all_tau = all_tau-delta
                
            p = np.nan_to_num(np.exp(all_tau*eps*h/(delta_u)), nan = 0)
            if np.sum(p) == 0:
                p[:] = 1/len(p)
            if np.sum(p) == np.inf:
                p = p/np.max(p)
            a[i] = np.random.choice(T.shape[0],p = p/np.sum(p)) #if dim = 0, a = col_assignment

        if len(set(a)) == 1:
            #print('un solo cluster')
            for j in range(c):
                b = np.random.choice(k)
                a[b] = j

        if len(set(a)) < c:
            h = [j for j in range(c) if j not in set(a)]
            #print(len(set(a)), c, h)
            self._noisy_T = np.delete(self._noisy_T, h, axis = (dimension + 1)%2)

        if dimension == 0:
            self._n_col_clusters = self._noisy_T.shape[1]
            for i, x in enumerate(list(set(a))):
                self._col_assignment[a == x] = i
        else:
            self._n_row_clusters = self._noisy_T.shape[0]
            for i, x in enumerate(list(set(a))):
                self._row_assignment[a == x] = i
        

    def compute_taus(self):
        tot = np.sum(self._T)
        tot_per_x = np.sum(self._T/tot, 1)
        tot_per_y = np.sum(self._T/tot, 0)
        t_square = np.power(self._T/tot, 2)

        a_x = np.sum(np.nan_to_num(np.true_divide(np.sum(t_square, axis = 0), tot_per_y)))
        b_x = np.sum(np.power(tot_per_x, 2))
        

        a_y = np.sum(np.nan_to_num(np.true_divide(np.sum(t_square, axis = 1), tot_per_x)))
        b_y = np.sum(np.power(tot_per_y, 2))


        tau_x = np.nan_to_num(np.true_divide(a_x - b_x, 1 - b_x))
        tau_y = np.nan_to_num(np.true_divide(a_y - b_y, 1 - b_y))
        


        return tau_x, tau_y



def read_matrix(file):
    m = open(file, 'r')
    for i, line in enumerate(m):
        if i == 2:
            dim_x, dim_y, dim_val = line.split()
            M = np.zeros((int(dim_x), int(dim_y)))
        if i >= 3:
            x, y, val = line.split()
            M[int(x) - 1, int(y) - 1] = int(val)
    m.close()
    print(M.shape)
    return M

def read_target(file, num):
    t = open(file, 'r')
    for i, line in enumerate(t):
        if i == 2:
            dim_x, dim_val = line.split()
            if int(dim_x) != num:
                raise('Deve esserci una riga per ogni oggetto!')
            else:
                T = np.zeros((int(dim_x)))

        if i >= 3:
            val = int(line)
            T[i - 3] = val
    t.close()
    return T



def plot_clusters(model):
    arr1inds = model._row_assignment.argsort()
    U = np.ones(model._dataset.shape)
    X = (U.T * (model._row_assignment + 1)).T
    #plt.matshow((model._dataset*X)[arr1inds])
    plt.matshow((model._dataset*X))
    #plt.show()

    arr2inds = model._col_assignment.argsort()
    U = np.ones(model._dataset.shape)
    X = (U * (model._col_assignment + 1))
    #plt.matshow((model._dataset*X)[:,arr2inds])
    plt.matshow((model._dataset*X))    

def MovieLens():
    final = pd.read_pickle('../resources/movielens_final_3g_6_2u.pkl')
    n = np.shape(final.groupby('userID').count())[0]
    m = np.shape(final.groupby('movieID').count())[0]
    l = np.shape(final.groupby('tagID').count())[0]
    T = np.zeros((n,m,l))
    y = np.zeros(m)
    for index, row in final.iterrows():
        T[row['user_le'], row['movie_le'], row['tag_le']] = 1
        y[row['movie_le']] = row['genre_le']
    T1 = np.sum(T, axis = 2)
    return T1, y
        
            
