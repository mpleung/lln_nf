import numpy as np, pandas as pd, snap, os
from scipy import sparse
from estimation_module import *
from gen_net_module import *

####### CREATE OUTPUT DIRECTORIES #######

if not os.path.exists('../output'):
    os.makedirs('../output')
os.chdir('../output')

for directory in ['id_set', 'stats']:
    if not os.path.exists(directory):
        os.makedirs(directory)
 
####### MAIN #######

seed = 1
np.random.seed(seed=seed)

d = 2 # dimension of node positions
n = 10000 # number of nodes
B = 100 # number of simulations
estimate_r = True # set to False to run simulations with r known

# true parameter: intercept, attribute, transitivity, standard deviation of random utility shock
theta = np.array([-0.2, 0.5, 0.3, 0.9])

for b in range(B):

    ####### GENERATE NETWORKS #######

    N = np.random.poisson(n)
    
    # RGG parameters
    (r, kappa) = gen_r(theta, d, N)
    if kappa < 1.44:
        print 'No giant component.'

    # positions uniformly distributed on unit cube
    positions = np.random.uniform(0,1,(N,d))
    # binary attribute
    Z = np.random.binomial(1, 0.5, N)
    # random utility terms
    eps = sparse.triu(np.random.normal(size=(N,N)), 1) * theta[3]
    # V minus endogenous statistic
    V_exo = gen_V_exo(Z, eps, theta)
    
    # generate random graphs 
    RGG = gen_RGG(positions, r) # initial RGG
    (D, RGG_minus, RGG_exo) = gen_D(RGG, V_exo, theta[2])
    # generate pairwise-stable network
    G = gen_G(D, RGG_minus, RGG_exo, V_exo, theta[2], N)

    # summary statistics
    graph_sumstats(G, 'G', 'stats/statsG_' + str(n) + '_b_' + str(b) + '.txt')
    graph_sumstats(D, 'D', 'stats/statsD_' + str(n) + '_b_' + str(b) + '.txt')
    graph_sumstats(RGG, 'RGG', 'stats/statsR_' + str(n) + '_b_' + str(b) + '.txt')
    graph_sumstats(RGG_exo, \
            'RGG_exo', 'stats/statsRexo_' + str(n) + '_b_' + str(b) + '.txt')

    # verify pairwise stability
    #if not check_pw(G, RGG, V_exo, theta[2]):
    #    raise ValueError('Simulation ' + str(b) + ' is not pairwise stable.')
    
    ####### ESTIMATION #######
    
    # estimate r
    if estimate_r:
        r = gen_rhat(G, positions)
        RGG = gen_RGG(positions, r) 

    # estimate identified set
    (moments_grid, Q_grid, id_set) = grid_search(G, RGG, Z, 21, theta, True)

    # save output
    pd.DataFrame(id_set).to_csv(
            'id_set/id_set_N_' + str(n) + '_b_' + str(b) + '.csv', \
            index=False, header=False)

    directory = 'moments'
    if not os.path.exists(directory):
        os.makedirs(directory)
    pd.DataFrame(Q_grid).to_csv(
        directory + '/Q_grid_N_' + str(n) + '_b_' + str(b) + '.csv', \
                index=False, header=False)
    pd.DataFrame(moments_grid).to_csv(
        directory + '/moments_true_N_' + str(n) + '_b_' + str(b) + '.csv', \
                index=False, header=False)

