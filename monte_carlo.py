# results in paper obtained by first running with n_vec = [100,500,2000] and then with n_vec = [5000] due to server walltime limitations. this goes for both sub- and supercritical designs

import numpy as np, snap, os, sys
from gen_net_module import *

seed = 0 
np.random.seed(seed=seed)

### Key user parameters ###

simulate_only = False # set to true to simulate limit expectation
subcritical = False # set to false for supercritical design

d = 2 # dimension of node positions
n_vec = [100,500,2000,5000] # number of nodes
B = 3000 # number of simulations for computing population quantities

# true parameter: intercept, attribute, transitivity, standard deviation of random utility shock
theta = np.array([-1, 0.25, 0.25, 1])
kappa = gen_kappa(np.array([0, 0.25, 0.25, 1]), d)
if not subcritical: theta[2] += 0.75

### Main ###

if subcritical:
    filename = 'results_sub.txt'
else:
    filename = 'results_super.txt'
if os.path.isfile(filename):
    os.remove(filename)
f = open(filename, 'a')
sys.stdout = f

### Output ###

avg_cc = np.zeros(len(n_vec)) # estimates of average clustering
avg_deg = np.zeros(len(n_vec)) # estimates of average degree
exp_cc = np.zeros(len(n_vec)) # expected clustering (finite model)
exp_deg = np.zeros(len(n_vec)) # expected degree (finite model)
std_cc = np.zeros(len(n_vec)) # stdev of clustering (finite model)
std_deg = np.zeros(len(n_vec)) # stdev of average degree (finite model)
limit_cc = 0
limit_deg = 0

if simulate_only:

    ### limit model estimand ###
    
    print theta
    print "\nLimit model"

    for b in range(B):
        # Simulate Poisson(kappa f(i)), where f(i) = 1 because positions are drawn from U[0,1]^2 in the finite model. 
        # Since it's not possibly to simulate an infinite number of points, we instead simulate a Poisson(kappa f(i)) process restricted to [-4,4]^2. Given the RGG link radius is 1 and we're interested in only the statistic of the node at the origin, this should be good enough.
        # To simulate from this restricted process we make use of the fact that it has the same distribution as (X_1, ..., X_N) i.i.d. draws from U([-4,4]^2) where N is Poisson(kappa/4*64)
        N = np.random.poisson(kappa * 400, 1)[0]
        G = gen_SNF(theta, d, N+1, r, True)

        # extract clustering coefficient, degree of node at origin
        limit_cc += snap.GetNodeClustCf(G, 0) / float(B)
        limit_deg += len([i for i in G.GetNI(0).GetOutEdges()]) / float(B)

    print "  Limit clustering: %s" % limit_cc 
    print "  Limit degree: %s" % limit_deg 

else:
    
    ### finite model estimand ###

    print "\nFinite model"
    print n_vec

    # plug in from results with simulate_only = True
    if subcritical:
        limit_cc = 0.182368518519
        limit_deg = 2.65166666667
    else:
        limit_cc = 0.403850919451
        limit_deg = 4.84866666667

    for i in range(len(n_vec)):
        n = n_vec[i]
        r = (kappa/float(n))**(1/float(d))

        ccs = np.zeros(B) # avg clustering within graph
        degs = np.zeros(B) # avg degree within graph
        
        for b in range(B):
            # simulate network
            G = gen_SNF(theta, d, n, r, True)

            # compute average clustering
            cc_vector = snap.TIntFltH()
            snap.GetNodeClustCf(G, cc_vector)
            cc = np.array([cc_vector[j] for j in cc_vector])
            ccs[b] = cc.mean()

            # compute average degree
            degs[b] = G.GetEdges() / float(n) * 2

        exp_cc[i] = ccs.mean()
        exp_deg[i] = degs.mean()
        std_cc[i] = ccs.std()
        std_deg[i] = degs.std()

    print "  Expected clustering: %s" % exp_cc
    print "  Expected degree: %s" % exp_deg
    print "\nDeviations from limit expectation"
    print "  Clustering: %s" % (exp_cc - limit_cc)
    print "  Degree: %s" % (exp_deg - limit_deg)

    print "  Stdev clustering: %s" % std_cc
    print "  Stdev avg degree: %s" % std_deg

    print "\n  Root-n rate:"
    print "    (size ratio, std cluster, std degree)"
    for i in range(len(n_vec)-1):
        print "    (%s,%s,%s)" % (math.sqrt(n_vec[i])/math.sqrt(n_vec[i+1]),  std_cc[i+1]/std_cc[i], std_deg[i+1]/std_deg[i])

    ### finite model estimator ###
    
    for i in range(len(n_vec)):
        # simulate network
        n = n_vec[i]
        r = (kappa/float(n))**(1/float(d))
        G = gen_SNF(theta, d, n, r, True)

        # compute average clustering
        cc_vector = snap.TIntFltH()
        snap.GetNodeClustCf(G, cc_vector)
        cc = np.array([cc_vector[j] for j in cc_vector])
        avg_cc[i] += cc.mean()

        # compute average degree
        avg_deg[i] += G.GetEdges() / float(n) * 2

    print "\n  Average clustering: %s" % avg_cc
    print "  Average degree: %s" % avg_deg

    print "\nDeviations from finite expectation"
    print "  Clustering: %s" % (avg_cc - exp_cc)
    print "  Degree: %s" % (avg_deg - exp_deg)

    print "\nDeviations from limit expectation"
    print "  Clustering: %s" % (avg_cc - limit_cc)
    print "  Degree: %s" % (avg_deg - limit_deg)

f.close()
