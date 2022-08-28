import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import normalized_mutual_info_score as nmi
import matplotlib.pyplot as plt

from algorithms.coclust_DP import CoClust

from utils import CreateOutputFile, execute_test_dp
import sys


dataset = 'cstr'
n_test = 30
k = 15


dt = pd.read_csv(f'./data/{dataset}.txt')
t = pd.read_csv(f'./data/{dataset}_target.txt', header = None)
target = np.array(t).T[0]


n = len(dt.doc.unique())
m = len(dt.word.unique())
T = np.zeros((n,m), dtype = int)


for g in dt.iterrows():
    T[g[1].doc,g[1].word] = g[1].cluster
    #T[g[1].doc,g[1].word] = 1

f, date = CreateOutputFile(dataset)
ty = np.zeros(m, dtype = int)
for t in range(n_test):
    for eps in [.1, .2, .3, .4, .5, 1, 1.5,2,2.5,3,3.5,4,4.5,5,6,7,8,9,10]:
        model = execute_test_dp(f, T, [target, ty], noise=0,n_iterations = 3, eps = eps, init = [k,k], verbose = False)
    
f.close()
