import numpy as np, snap, os, math, scipy.special
from scipy.stats import norm
from scipy import spatial, sparse

def ball_vol(d,r):
    """
    Returns the volume of a d-dimensional ball of radius r. Used for computing kappa in gen_kappa().
    """
    return math.pi**(d/2) * float(r)**d / scipy.special.gamma(d/2+1)


def gen_kappa(theta, d):
    """
    Returns kappa. Recall r_n = (kappa/n)^{1/d}.

    theta = true parameter.
    d = dimension of node positions.
    n = mean number of nodes.
    """
    vol = ball_vol(d,1)

    Phi2 = norm.cdf(-(theta[0] + 2*theta[1])/theta[3]) - norm.cdf(-(theta[0] + 2*theta[1] + theta[2])/theta[3])
    Phi1 = norm.cdf(-(theta[0] + theta[1])/theta[3]) - norm.cdf(-(theta[0] + theta[1] + theta[2])/theta[3])
    Phi0 = norm.cdf(-theta[0]/theta[3]) - norm.cdf(-(theta[0] + theta[2])/theta[3])
    gamma = math.sqrt( Phi2**2*0.25 + Phi1**2*0.5 + Phi0**2*0.25)
    
    return 1/(vol*gamma) - 0.3


def gen_V_exo(Z, eps, theta):
    """ 
    Returns 'exogenous' part of joint surplus function for each pair of nodes as a
    sparse upper triangular matrix.

    NB: This function is specific to the joint surplus function used in our
    simulations. 

    eps = sparse NxN upper triangular matrix. 
    Z = N-vector of binary attributes. 
    """
    N = Z.shape[0]
    sparse_ones = sparse.triu(np.ones((N,N)),1)
    Z_sum = sparse.triu(np.tile(Z, (N,1)) + np.tile(Z[:,np.newaxis], N),1)
    U = theta[0] * sparse_ones + theta[1] * Z_sum + eps
    return U

def gen_RGG(positions, r):
    """ 
    Returns an RGG from a given N-vector of dx1 positions (Nxd matrix). 
    
    positions = vector of node positions.
    r = linking threshold.
    """
    kdtree = spatial.KDTree(positions)
    pairs = kdtree.query_pairs(r) # default is Euclidean norm
    RGG = snap.GenRndGnm(snap.PUNGraph, len(positions), 0)
    for edge in (i for i in list(pairs)):
        RGG.AddEdge(edge[0],edge[1])
    return RGG

def gen_D(Pi, V_exo, theta2):
    """
    Returns a triplet of three snap graphs:
    D = opportunity graph with robust links removed.
    Pi_minus = subgraph of Pi without robustly absent potential links.
    Pi_exo = subgraph of Pi with only robust links.

    NB: This function is specific to the joint surplus used in our simulations.

    Pi = opportunity graph (in our case, the output of gen_RGG).
    V_exo = 'exogenous' part of joint surplus (output of gen_V_exo).
    theta2 = transitivity parameter (theta[2]).
    """
    N = V_exo.shape[0]
    D = snap.ConvertGraph(snap.PUNGraph, Pi)
    Pi_minus = snap.ConvertGraph(snap.PUNGraph, Pi)
    Pi_exo = snap.GenRndGnm(snap.PUNGraph, N, 0) 

    for edge in Pi.Edges():
        i = min(edge.GetSrcNId(), edge.GetDstNId())
        j = max(edge.GetSrcNId(), edge.GetDstNId())
        if V_exo[i,j] + min(theta2,0) > 0:
            D.DelEdge(i,j) 
            Pi_exo.AddEdge(i,j)
        if V_exo[i,j] + max(theta2,0) <= 0:
            D.DelEdge(i,j)
            Pi_minus.DelEdge(i,j)
 
    return (D, Pi_minus, Pi_exo)

def gen_G_subgraph(component, D, Pi_minus, Pi_exo, V_exo, theta2, Z, randomize):
    """ 
    Returns a pairwise-stable network for nodes in component, via myopic best-
    response dynamics. This subnetwork is pairwise-stable taking as given the
    links in the rest of the network. Initial network for best-response dynamics
    is the opportunity graph. 

    NB: This function is specific to the joint surplus used in our simulations.
    
    component = component of D for which we want a pairwise-stable subnetwork.
    D, Pi_minus, Pi_exo = outputs of gen_D().
    V_exo = 'exogenous' part of joint surplus (output of gen_V_exo).
    theta2 = transitivity parameter (theta[2]).

    randomize = boolean, set to True if you want the initial condition of the best response dynamics to randomize between the "empty" and "complete" networks. Set to False to set the initial condition to the "complete" network.
    """
    stable = False
    meetings_without_deviations = 0

    D_subgraph = snap.GetSubGraph(D, component)

    # Initial network on component of D.
    if randomize:
        # public signal
        nu = Z[np.array([i for i in component]).max()]
        if nu == 1:
            G = snap.GetSubGraph(Pi_minus, component) # initialize network of non-robustly absent links on the D-component
        else:
            G = snap.GetSubGraph(Pi_exo, component) # initialize network of robust links on the D-component
    else:
        G = snap.GetSubGraph(Pi_exo, component)
    
    # For each node pair (i,j) linked in Pi_exo (i.e. their links are robust),
    # with either i or j in component, add their link to G.  Result is the
    # subgraph of Pi_minus on an *augmented* component of D.
    for i in component:
        for j in Pi_exo.GetNI(i).GetOutEdges():
            if not G.IsNode(j): G.AddNode(j)
            G.AddEdge(i,j)

    while not stable:
        # Need only iterate through links of D, since all other links are
        # robust.
        for edge in D_subgraph.Edges():
            # Iterate deterministically through default edge order order. Add or
            # remove link in G according to myopic best-response dynamics. If we
            # cycle back to any edge with no changes to the network, conclude
            # it's pairwise stable.
            i = min(edge.GetSrcNId(), edge.GetDstNId())
            j = max(edge.GetSrcNId(), edge.GetDstNId())
            Sij = snap.GetCmnNbrs(G, i, j) > 0
            if V_exo[i,j] + theta2*Sij > 0: # specific to model of V
                if G.IsEdge(i,j):
                    meetings_without_deviations += 1
                else:
                    G.AddEdge(i,j)
                    meetings_without_deviations = 0
            else:
                if G.IsEdge(i,j):
                    G.DelEdge(i,j)
                    meetings_without_deviations = 0
                else:
                    meetings_without_deviations += 1

        if meetings_without_deviations >= D_subgraph.GetEdges():
            stable = True

    return snap.GetSubGraph(G, component)

def gen_G(D, Pi_minus, Pi_exo, V_exo, theta2, N, Z, randomize):
    """
    Returns pairwise-stable network on N nodes. 
    
    D, Pi_minus, Pi_exo = outputs of gen_D().
    V_exo = 'exogenous' part of joint surplus (output of gen_V_exo).
    theta2 = transitivity parameter (theta[2]).
    """
    G = snap.GenRndGnm(snap.PUNGraph, N, 0) # initialize empty graph
    Components = snap.TCnComV()
    snap.GetWccs(D, Components) # collects components of D
    NIdV = snap.TIntV() # initialize vector
    for C in Components:
        if C.Len() > 1:
            NIdV.Clr()
            for i in C:
                NIdV.Add(i)
            tempnet = gen_G_subgraph(NIdV, D, Pi_minus, Pi_exo, V_exo, theta2, Z, randomize)
            for edge in tempnet.Edges():
                G.AddEdge(edge.GetSrcNId(), edge.GetDstNId())
 
    # add robust links
    for edge in Pi_exo.Edges():
        G.AddEdge(edge.GetSrcNId(), edge.GetDstNId())

    return G

def gen_SNF(theta, d, N, r, randomize):
    """
    Generates pairwise-stable network on N nodes.
    """
        
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
    
    return gen_G(D, RGG_minus, RGG_exo, V_exo, theta[2], N, Z, randomize)
