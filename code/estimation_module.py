import numpy as np, pandas as pd, snap, math, time
from scipy.stats import norm

##############################
### Dyadic outcome moments ###
##############################

def dyad_moment_obs(G, Z, Pi):
    """
    Empirical dyadic outcome moments.
    
    Returns 3x4 matrix of entries 1/n \sum_{i,j} \mathbf{Y}_{ij}(r_n)
    h(X_{ij}). Columns correspond to dyadic outcomes (1,1), (0,1), (1,0),
    (0,0).  Rows correspond to the instruments 1{Z_i+Z_j=0}, 1{Z_i+Z_j=1}, and
    1{Z_i+Z_j=2}. 

    G = pairwise-stable network on N nodes (snap object, output of gen_G() in gen_net_module).
    Z = N-vector of binary attributes.
    Pi = opportunity graph (snap object, e.g. output of gen_RGG() in gen_net_module).
    """
    output = np.zeros((3,4))
    for edge in Pi.Edges(): # only include pairs that are linked in Pi
        i = min(edge.GetSrcNId(), edge.GetDstNId())
        j = max(edge.GetSrcNId(), edge.GetDstNId())
        if i == j:
            raise ValueError('Error in gen_observed_moment: no self-links \
                    allowed.')

        cfriend = snap.GetCmnNbrs(G, i, j) > 0 # 1{# common friends > 0}
        Z_ij = Z[i]+Z[j]
        if G.IsEdge(i,j):
            if Z_ij == 0:
                if cfriend:
                    output[0,0] += 2 # transitive triad
                else:
                    output[0,2] += 2 # linked, no common friends
            elif Z_ij == 1:
                if cfriend:
                    output[1,0] += 2
                else:
                    output[1,2] += 2
            elif Z_ij == 2:
                if cfriend:
                    output[2,0] += 2
                else:
                    output[2,2] += 2
            else:
                raise ValueError('Z[i]+Z[j] not in {0,1,2}')
        else:
            if Z_ij == 0:
                if cfriend:
                    output[0,1] += 2 # intransitive triad
                else:
                    output[0,3] += 2 # unlinked, no common friends
            elif Z_ij == 1:
                if cfriend:
                    output[1,1] += 2
                else:
                    output[1,3] += 2
            elif Z_ij == 2:
                if cfriend:
                    output[2,1] += 2
                else:
                    output[2,3] += 2
            else:
                raise ValueError('Z[i]+Z[j] not in {0,1,2}')
        
    return output / float(Z.shape[0])

def dyad_instr_counts(Z, Pi):
    """
    Used in dyad_moment_sim(). Returns length-3 vector. Component k counts number of dyads in Pi for which Z_i+Z_j=k, multiplied by 1/N.

    Z = N-vector of binary attributes.
    Pi = opportunity graph.
    """
    instr = np.zeros(3)

    for edge in Pi.Edges(): # restriction ||i-j|| \leq r_n
        i = min(edge.GetSrcNId(), edge.GetDstNId())
        j = max(edge.GetSrcNId(), edge.GetDstNId())
        if i == j:
            raise ValueError('Error in gen_model_moment: i = j')
        Z_ij = Z[i] + Z[j]
        if Z_ij not in [0,1,2]:
            raise ValueError('Z[i]+Z[j] not in {0,1,2}')
        else:
            instr[Z_ij] += 2
            
    return instr / float(Z.shape[0])
    
def dyad_moment_sim(theta, instr):
    """
    Estimate of 'simulated part' of moment inequalities for dyadic outcome moments.
    
    Returns 3x3 matrix, where the columns correspond respectively to the
    instruments 1{Z_i+Z_j=0}, 1{Z_i+Z_j=1}, and 1{Z_i+Z_j=2}. In the notation
    of the example in Section 6.1, the row components are analogous to
    P( (1,1),(1,0) | X_ij ), P( (1,1),(0,0) | X_ij ), P( (0,1),(1,0) | X_ij ), and P( (0,1),(0,0) | X_ij ).

    NB: This function is specific to the joint surplus function used in our
    simulations. 

    theta = parameter vector.
    instr = output of dyad_instr_counts().
    """
    output = np.zeros((3,4)) # matrix of conditional probabilities

    t0,t1,t2 = theta[0]/theta[3], theta[1]/theta[3], theta[2]/theta[3]
    V_exo = t0 + np.array([0,1,2])*t1
    norm_V_exo = norm.cdf(V_exo)
    norm_V_all = norm.cdf(V_exo + t2)

    output[:,0] = norm.cdf(V_exo + min(t2,0)) # P( (1,1),(1,0) | X_ij )
    output[:,1] = np.maximum(norm_V_all-norm_V_exo, 0) # P( (1,1),(0,0) | X_ij )
    output[:,2] = np.maximum(norm_V_exo-norm_V_all, 0) # P( (0,1),(1,0) | X_ij )
    output[:,3] = norm.cdf(-V_exo - max(t2,0)) # P( (0,1),(0,0) | X_ij )
    
    return output * instr[:,np.newaxis]

def dyad_moments(M_obs, M_sim):
    """
    Returns array of all dyadic outcome moments. Dimension is # instruments (3) by |U| (2^4-1). 

    M_obs = output of observed_moment().
    M_sim = output of model_moment().
    """
    # values in \mathcal{U} to use for generating moments
    U = np.mgrid[0:2:1, 0:2:1, 0:2:1, 0:2:1].reshape(4,-1).T
    U = U[1:U.shape[0]]
    
    output = np.zeros((3,U.shape[0]))
 
    for i in range(3): 
        for u in range(U.shape[0]):
            output[i,u] = np.dot(U[u,:], M_obs[i,:]) \
                    - max(U[u,0], U[u,2]) * M_sim[i,0] \
                    - max(U[u,0], U[u,3]) * M_sim[i,1] \
                    - max(U[u,1], U[u,2]) * M_sim[i,2] \
                    - max(U[u,1], U[u,3]) * M_sim[i,3]
    
    return output

###############################
### Triadic outcome moments ###
###############################

def conn_triples(G):
    """
    Used in triad_moment_obs(). Recursively creates list of node triplet lists. A triplet is included if it forms an intransitive or transitive triad in G. Output is a global list of lists.

    G = network.
    """
    global triads
    triads = []
    for i_iter in G.Nodes():
        N = [j for j in i_iter.GetOutEdges() if j > i_iter.GetId()]
        i = i_iter.GetId()
        ct_recur([i], N, i, G)
    return triads

def ct_recur(S, N, i, G):
    """
    Recursion for conn_triples(). 
    
    S,N = lists of node labels.
    i = node label. 
    G = network.
    """
    if len(S) == 3:
        triads.append(S)
    else:
        while len(N) != 0:
            k = N[0]
            N.remove(k)
            ct_recur(S + [k], N + [l for l in G.GetNI(k).GetOutEdges() if l > i and l not in S+[m for m in G.GetNI(i).GetOutEdges()]+N], i, G)

def triad_moment_obs(G, Z):
    """
    Empirical triadic outcome moments without using knowledge of opportunity graph.

    Returns vector of length 24. Each component equals 1/n \sum_{i, j\neq i, k\neq i,j} 1{Y_{ijk}=y} 1{X_{ijk}=x} for values of triadic outcomes y and triplet-level covariates x.

    G = pairwise-stable network on N nodes.
    Z = N-vector of binary attributes.
    """
    output = np.zeros(24)
    c3 = conn_triples(G)
    for t in c3:
        l01 = G.IsEdge(t[0],t[1])
        l12 = G.IsEdge(t[1],t[2])
        l02 = G.IsEdge(t[0],t[2])
        
        if l01+l12+l02 == 2: # intransitive triad
            # permute indices to make the first element the node with two links
            if l01 and l02:
                t2 = t
            elif l12 and l01:
                t2 = [t[1], t[0], t[2]] 
            elif l02 and l12:
                t2 = [t[2], t[0], t[1]]

            X_ijk = np.array([Z[t2[0]], Z[t2[1]], Z[t2[2]]])
            S_01 = snap.GetCmnNbrs(G, t2[0], t2[1]) > 0 # 1{# common friends > 0}
            S_02 = snap.GetCmnNbrs(G, t2[0], t2[2]) > 0
            
            if S_01+S_02 == 0: # comment format: Z_j <-S_ij-> Z_i <-S_ik-> Z_k, meaning the outcome is G_ij=G_ik=1-G_jk=1 (intransitive triad) and S_ij,S_ik,Z_i,Z_j,Z_k take on the specified values
                if X_ijk.sum() == 0:
                    output[0] += 2 # 0 <-0-> 0 <-0-> 0
                    # we add two is because this outcome would get double-counted in the sum because permuting j and k would result in the same unlabeled subnetwork
                elif X_ijk.sum() == 1:
                    if X_ijk[0] == 1:
                        output[1] += 2 # 0 <-0-> 1 <-0-> 0
                    else:
                        output[2] += 1 # 1 <-0-> 0 <-0-> 0 OR 0 <-0-> 0 <-0-> 1
                elif X_ijk.sum() == 2: 
                    if X_ijk[0] == 0:
                        output[3] += 2 # 1 <-0-> 0 <-0-> 1
                    else:
                        output[4] += 1 # 0 <-0-> 1 <-0-> 1 OR 1 <-0-> 1 <-0-> 0
                elif X_ijk.sum() == 3: 
                    output[5] += 2 # 1 <-0-> 1 <-0-> 1
            elif S_01+S_02 == 1: 
                if X_ijk.sum() == 0:
                    output[6] += 1 # 0 <-1-> 0 <-0-> 0 OR 0 <-0-> 0 <-1-> 0
                elif X_ijk.sum() == 1:
                    if X_ijk[0] == 1:
                        output[7] += 1 # 0 <-1-> 1 <-0-> 0 OR 0 <-0-> 1 <-1-> 0
                    elif X_ijk[1] == 1:
                        if S_01 == 1:
                            output[8] += 1 # 1 <-1-> 0 <-0-> 0
                        else:
                            output[9] += 1 # 1 <-0-> 0 <-1-> 0
                    elif X_ijk[2] == 1:
                        if S_01 == 1:
                            output[9] += 1 # 0 <-1-> 0 <-0-> 1
                        else:
                            output[8] += 1 # 0 <-0-> 0 <-1-> 1
                elif X_ijk.sum() == 2:
                    if X_ijk[0] == 0:
                        output[10] += 1 # 1 <-1-> 0 <-0-> 1 OR 1 <-0-> 0 <-1-> 1
                    elif X_ijk[1] == 0:
                        if S_01 == 1:
                            output[11] += 1 # 0 <-1-> 1 <-0-> 1
                        else:
                            output[12] += 1 # 0 <-0-> 1 <-1-> 1
                    elif X_ijk[2] == 0:
                        if S_01 == 1:
                            output[12] += 1 # 1 <-1-> 1 <-0-> 0
                        else:
                            output[11] += 1 # 1 <-0-> 1 <-1-> 0
                elif X_ijk.sum() == 3:
                    output[13] += 1 # 1 <-1-> 1 <-0-> 1 OR 1 <-0-> 1 <-1-> 1
            elif S_01+S_02 == 2:
                if X_ijk.sum() == 0:
                    output[14] += 2 # 0 <-1-> 0 <-1-> 0
                elif X_ijk.sum() == 1:
                    if X_ijk[0] == 1:
                        output[15] += 2 # 0 <-1-> 1 <-1-> 0
                    else:
                        output[16] += 1 # 1 <-1-> 0 <-1-> 0 OR 0 <-1-> 0 <-1-> 1
                elif X_ijk.sum() == 2:
                    if X_ijk[0] == 0:
                        output[17] += 2 # 1 <-1-> 0 <-1-> 1
                    else:
                        output[18] += 1 # 0 <-1-> 1 <-1-> 1 OR 1 <-1-> 1 <-1-> 0
                elif X_ijk.sum() == 3:
                    output[19] += 2 # 1 <-1-> 1 <-1-> 1
        elif l01+l12+l02 == 3: # transitive triad
            X_ijk = np.array([Z[t[0]], Z[t[1]], Z[t[2]]])
            if X_ijk.sum() == 0: # Z_i,Z_j,Z_k=0
                output[20] += 6
            elif X_ijk.sum() == 1: # one of Z_i,Z_j,Z_k equals one
                output[21] += 2
            elif X_ijk.sum() == 2:
                output[22] += 2
            elif X_ijk.sum() == 3:
                output[23] += 6
    return output.T / Z.shape[0]

def triad_instr_counts(Z, Pi):
    """
    Used in triad_moment_sim(). Returns length-10 vector. Component k = number of triads whose subgraph Pi equals a certain connected subgraph (intransitive triad or transitive triad), times 1{X_{ijk}=x}, times 1/N.

    Z = N-vector of binary attributes.
    Pi = opportunity graph.
    """
    output = np.zeros(10)
    c3 = conn_triples(Pi)
    for t in c3:
        l01 = Pi.IsEdge(t[0],t[1])
        l12 = Pi.IsEdge(t[1],t[2])
        l02 = Pi.IsEdge(t[0],t[2])
        
        if l01+l12+l02 == 2: # intransitive triad
            # permute indices to make the first element the node with two links
            if l01 and l02:
                t2 = t
            elif l12 and l01:
                t2 = [t[1], t[0], t[2]] 
            elif l02 and l12:
                t2 = [t[2], t[0], t[1]]

            X_ijk = np.array([Z[t2[0]], Z[t2[1]], Z[t2[2]]])
            if X_ijk.sum() == 0: # comment format: Z_j -- Z_i -- Z_k
                output[0] += 2 # 0 -- 0 -- 0
            elif X_ijk.sum() == 1:
                if X_ijk[0] == 1:
                    output[1] += 2 # 0 -- 1 -- 0
                else:
                    output[2] += 1 # 1 -- 0 -- 0 OR 0 -- 0 -- 1
            elif X_ijk.sum() == 2: 
                if X_ijk[0] == 0:
                    output[3] += 2 # 1 -- 0 -- 1
                else:
                    output[4] += 1 # 0 -- 1 -- 1 OR 1 -- 1 -- 0
            elif X_ijk.sum() == 3: 
                output[5] += 2 # 1 -- 1 -- 1
        elif l01+l12+l02 == 3: # transitive triad
            X_ijk = np.array([Z[t[0]], Z[t[1]], Z[t[2]]])
            if X_ijk.sum() == 0:
                output[6] += 6
            elif X_ijk.sum() == 1:
                output[7] += 2
            elif X_ijk.sum() == 2:
                output[8] += 2
            elif X_ijk.sum() == 3:
                output[9] += 6
    return output / float(Z.shape[0])

def triad_moment_sim(theta):
    """
    Conditional probabilities used in 'simulated part' of moment inequalities for triadic outcome moments, without using knowledge of opportunity graph.
    
    Similar to dyad_moment_sim() but for triadic outcome moments. Returns 5x8 array.

    Consider the triplet of nodes (i,j,k). Let Gi be an indicator for the
    intransitive triad j -- i -- k, Gj for i -- j -- k, and Gk for i -- k -- j,
    and Gt the transitive triad indicator. Let
    y11 = (Gi,0,0,1), y12 = (Gi,1,0,1), y14 = (Gi,1,1,1), y41 = (Gt,1,1,1),
    where the last 3 components of each element correspond to S_ij, S_jk, S_ik.

    The first 3 columns of the output array correspond to the conditional probabilities P(y11), P(y12), P(y14), all conditional on (Z_i,Z_j,Z_k) and Pi_ij=1, Pi_jk=1, Pi_ik=0.
    The 4th column corresponds to P(y41), conditional on (Z_i,Z_j,Z_k) and Pi_ij=1, Pi_jk=1, Pi_ik=1.
    The 5th column equals P(V(s,W_ik,theta) < 0 | Z_i,Z_j,Z_k, Pi_ik=1). It will be later used to construct the analogs of the first 9 columns, where we condition instead on Pi_ij=1, Pi_jk=1, Pi_ik=1.
    
    The rows of the output array are the instruments -- indicators for equality
    of (Z_j,Z_i,Z_k) and (0,0,0), (1,0,0), (0,1,0), (0,0,1), (0,1,1), (1,0,1),
    (1,1,0), (1,1,1).

    NB: This function is specific to the joint surplus function used in our
    simulations. 

    theta = parameter vector.
    """
    X_ij = np.array([0,1,1,0,1,1,2,2]) # each component equals Z_i+Z_j for the corresponding instrument
    X_jk = np.array([0,1,0,1,1,2,1,2]) # Z_j+Z_k
    X_ik = np.array([0,0,1,1,2,1,1,2]) # Z_i+Z_k
    
    t0,t1,t2 = theta[0]/theta[3], theta[1]/theta[3], theta[2]/theta[3]
    norm_V_all = norm.cdf(t0+X_ik*t1+t2)
    pre_output_ij = norm.cdf( (t0+X_ij*t1)[:,np.newaxis] + np.array([0, t2, t2, t2]) )
    pre_output_jk = norm.cdf( (t0+X_jk*t1)[:,np.newaxis] + np.array([0, 0, t2, t2]) )
    pre_output_ik = np.vstack([np.ones((3,8)), norm_V_all]).T

    return np.hstack([pre_output_ij * pre_output_jk * pre_output_ik, (1-norm_V_all)[:,np.newaxis]])
    
def triad_moments(M_obs, M_sim, instr):
    """
    Returns an array of triadic outcome moments without using knowledge of opportunity graph. Note this is a small subset of all possible triadic outcome moments. 

    M_obs = output of triad_moment_obs().
    M_sim = output of triad_moment_sim().
    instr = output of triad_instr_counts().
    """
    # Variable names signify the triadic outcome type and value of (Z_i,Z_j,Z_k). Last three digits of each name give value of Z_i's in order of i,j,k. The part before the underscore signifies the triadic outcome. (Recall definition of e.g. y11 from help text of triad_moment_sim().)
    # Comments list unlabeled triadic outcomes in stable set with associated covariates, same as the format in triad_moment_obs().
    y11_000 = M_obs[0] - M_sim[0,0]*(instr[0] + M_sim[0,4]*instr[6]) # 0 <-0-> 0 <-0-> 0
    y11_100 = M_obs[1] - M_sim[1,0]*(instr[1] + M_sim[1,4]*instr[7]) # 0 <-0-> 1 <-0-> 0 
    y11_010 = M_obs[2] - M_sim[2,0]*(instr[2] + M_sim[2,4]*instr[7]) # 1 <-0-> 0 <-0-> 0
    y11_011 = M_obs[3] - M_sim[4,0]*(instr[3] + M_sim[4,4]*instr[8]) # 1 <-0-> 0 <-0-> 1
    y11_101 = M_obs[4] - M_sim[5,0]*(instr[4] + M_sim[5,4]*instr[8]) # 1 <-0-> 1 <-0-> 0
    y11_111 = M_obs[5] - M_sim[7,0]*(instr[5] + M_sim[7,4]*instr[9]) # 1 <-0-> 1 <-0-> 1
    y12_000 = M_obs[6] - M_sim[0,1]*(instr[0] + M_sim[0,4]*instr[6]) # 0 <-1-> 0 <-0-> 0
    y12_100 = M_obs[7] - M_sim[1,1]*(instr[1] + M_sim[1,4]*instr[7]) # 0 <-1-> 1 <-0-> 0
    y12_010 = M_obs[8] - M_sim[2,1]*(instr[2] + M_sim[2,4]*instr[7]) # 1 <-1-> 0 <-0-> 0
    y12_001 = M_obs[9] - M_sim[3,1]*(instr[2] + M_sim[3,4]*instr[7]) # 0 <-1-> 0 <-0-> 1
    y12_011 = M_obs[10] - M_sim[4,1]*(instr[3] + M_sim[4,4]*instr[8]) # 1 <-1-> 0 <-0-> 1
    y12_101 = M_obs[11] - M_sim[5,1]*(instr[4] + M_sim[5,4]*instr[8]) # 0 <-1-> 1 <-0-> 1
    y12_110 = M_obs[12] - M_sim[6,1]*(instr[4] + M_sim[6,4]*instr[8]) # 1 <-1-> 1 <-0-> 0
    y12_111 = M_obs[13] - M_sim[7,1]*(instr[5] + M_sim[7,4]*instr[9]) # 1 <-1-> 1 <-0-> 1
    y14_000 = M_obs[14] - M_sim[0,2]*(instr[0] + M_sim[0,4]*instr[6]) # 0 <-1-> 0 <-1-> 0
    y14_100 = M_obs[15] - M_sim[1,2]*(instr[1] + M_sim[1,4]*instr[7]) # 0 <-1-> 1 <-1-> 0
    y14_010 = M_obs[16] - M_sim[2,2]*(instr[2] + M_sim[2,4]*instr[7]) # 1 <-1-> 0 <-1-> 0
    y14_011 = M_obs[17] - M_sim[4,2]*(instr[3] + M_sim[4,4]*instr[8]) # 1 <-1-> 0 <-1-> 1
    y14_101 = M_obs[18] - M_sim[5,2]*(instr[4] + M_sim[5,4]*instr[8]) # 0 <-1-> 1 <-1-> 1
    y14_111 = M_obs[19] - M_sim[7,2]*(instr[5] + M_sim[7,4]*instr[9]) # 1 <-1-> 1 <-1-> 1
    y41_000 = (M_obs[20] - M_sim[0,3]*instr[6])/6 # transitive triad
    y41_100 = (M_obs[21] - M_sim[1,3]*instr[7])/2
    y41_011 = (M_obs[22] - M_sim[4,3]*instr[8])/2
    y41_111 = (M_obs[23] - M_sim[7,3]*instr[9])/6

    return np.array([y11_000, y11_100, y11_010, y11_011, y11_101, y11_111, y12_000, y12_100, y12_010, y12_001, y12_011, y12_101, y12_110, y12_111, y14_000, y14_100, y14_010, y14_011, y14_101, y14_111, y41_000, y41_100, y41_011, y41_111])

##################
### Estimation ###
##################

def gen_rhat(G, positions):
    """
    Returns estimate of RGG linking threshold = max{||i-j||: G_{ij}=1}. See
    final remark of Section 6.2.

    G = pairwise-stable network on N nodes.
    positions = Nx2 matrix of node positions in R^2.
    """
    output = 0
    for edge in G.Edges():
        i = edge.GetSrcNId()
        j = edge.GetDstNId()
        dist = math.sqrt((positions[i,0] - positions[j,0])**2 + \
                (positions[i,1] - positions[j,1])**2)
        output = max(output, dist)
    return output

def grid_search(G, Pi, Z, m, theta, triad):
    """
    Returns estimate of identified set.

    G = pairwise-stable network on N nodes.
    Pi = opportunity graph.
    Z = N-vector of binary attributes.
    m = grid the step size in each dimension. In the last dimension (corresponding to standard deviation) the step size is m/2+1.
    theta = TRUE parameter.
    triad = boolean for whether or not to include triadic outcome moments.
    """
    N = Z.shape[0]
    k = complex(0,m)

    # grid of parameters
    theta_grid = np.mgrid[-1:1:k, -1:1:k, -1:1:k, 0:1:complex(0,m/2+1)].reshape(4,-1).T
    # can't have zero variance
    theta_grid[:,3] += np.tile(np.hstack([0.0001, np.zeros(m/2)]), m**3)
    theta_grid = np.vstack([theta, theta_grid])
    tot = theta_grid.shape[0]

    if triad:
        triad_dim = 24 # number of triadic outcome moments
    else:
        triad_dim = 0
    
    moments_grid = np.zeros((tot,(2**4-1)*3+triad_dim))
    
    dyad_obs_M = dyad_moment_obs(G, Z, Pi)
    dyad_instr = dyad_instr_counts(Z, Pi)
    if triad:
        triad_obs_M = triad_moment_obs(G, Z)
        triad_instr = triad_instr_counts(Z, Pi)
    
    for t in range(tot):
        if t == 0:
            t0 = time.clock()
        
        dyad_sim_M = dyad_moment_sim(theta_grid[t,:], dyad_instr)
        dyad_M = dyad_moments(dyad_obs_M, dyad_sim_M).flatten()
        if triad:
            triad_sim_M = triad_moment_sim(theta_grid[t,:])
            triad_M = triad_moments(triad_obs_M, triad_sim_M, triad_instr)
            moments_grid[t,:] = np.hstack([triad_M, dyad_M])
        else:
            moments_grid[t,:] = dyad_M
        
        if t == 0:
            t1 = time.clock()
            print "%f seconds for one parameter." % (t1-t0)
    
    # CHT objective function for each parameter
    Q_grid = np.sum(np.square(np.maximum(moments_grid, 0)),1)
    
    c = math.log(N)/N
    id_set = theta_grid[Q_grid < c,:]
   
    return (moments_grid[Q_grid < c,:], Q_grid[Q_grid < c], id_set)

