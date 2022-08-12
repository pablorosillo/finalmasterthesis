"""
Discrete time algorithm - model for speakers in networks

Pablo Rosillo Rodes, 2022
Final Master thesis

"""


import numpy as np
import numba
import time
import csv

"""
import graph_tool.all as gta
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import networkx.algorithms.community as nxc
import scipy.stats as scst
import scipy.linalg as scla
from scipy.optimize import curve_fit
import scipy.special as scsp
import pyperclip
"""

c1 = 0.2; c2 = 0.4;
L = 1500
p = 0.8


@numba.jit(nopython=True, parallel=True)
def simsis(T, nststate, Adj, pX1, pX2, pY1, c1, c2, nss):
    L = int(len(Adj))
    s1vec = [numba.float64(x) for x in range(0)]
    s2vec = [numba.float64(x) for x in range(0)]
    meanx = [numba.float64(x) for x in range(0)]
    stdx = [numba.float64(x) for x in range(0)]
    meanw = [numba.float64(x) for x in range(0)]
    stdw = [numba.float64(x) for x in range(0)]
    meanz = [numba.float64(x) for x in range(0)]
    stdz = [numba.float64(x) for x in range(0)]
    
    vertices = np.arange(0, L, 1)
    np.random.shuffle(vertices)

    htoday0 = np.zeros(L, dtype=numba.int64)
    
    x1g = int(pX1*L); x2g = int(pX2*L); y1g = int(pY1*L);
    y2g = L-x1g-x2g-y1g;
    
    for i in range(x1g):
        htoday0[vertices[i]] = 1;
    for i in range(x1g,x1g+x2g):
        htoday0[vertices[i]] = 2;
    for i in range(x1g+x2g,x1g+x2g+y1g):
        htoday0[vertices[i]] = 3;
    
    s1vec0 = np.linspace(0.5,1,nss);
    
    for i in range(len(s1vec0)):
        
        s1 = s1vec0[i];
        
        for s2 in np.linspace(0.5,s1,i+1):
            
            s1vec.append(s1);
            s2vec.append(s2);
            
            print("s1 = ", s1)
            
            xststate = np.zeros(nststate, dtype=numba.float64)
            wststate = np.zeros(nststate, dtype=numba.float64)
            zststate = np.zeros(nststate, dtype=numba.float64)
    
            for m in numba.prange(nststate):
                
                x1g = int(pX1*L); x2g = int(pX2*L); y1g = int(pY1*L);
                y2g = L-x1g-x2g-y1g;
                
                htoday = htoday0.copy();
                
                for t in range(T*L):
                    
                    u = np.random.choice(vertices)
                        
                    x1l = 0; x2l = 0; y1l = 0; y2l = 0;
                    
                    aux = np.delete(vertices,u)
                    
                    auxprop = 0;
                    
                    for v in aux:
                        if Adj[u][v] == 1:
                            auxprop += 1;
                            if htoday[v] == 1:
                                x1l += 1;
                            elif htoday[v] == 2:
                                x2l += 1;
                            elif htoday[v] == 3:
                                y1l += 1;
                            else:
                                y2l += 1;
                    
                    
                    if auxprop != 0:
                        
                        auxrand = np.random.rand()
                    
                        if htoday[u] == 1:
                            if auxrand <= ((1-s1)*(y1l+y2l)/auxprop)/((1-s1)*(y1l+y2l)/auxprop+c2*(x2l+y2l)/auxprop):
                                htoday[u] = 3;
                                x1g -= 1; y1g += 1;
                            else:
                                htoday[u] = 2;
                                x1g -= 1; x2g += 1;
                        elif htoday[u] == 2: 
                            if auxrand <= (s2*(y1l+y2l)/auxprop)/(s2*(y1l+y2l)/auxprop+(1-c2)*(y1l+x1l)/auxprop):
                                htoday[u] = 0;
                                x2g -= 1; y2g += 1;
                            else:
                                htoday[u] = 1;
                                x2g -= 1; x1g += 1;
                        elif htoday[u] == 3: 
                            if auxrand <= (s1*(x1l+x2l)/auxprop)/(s1*(x1l+x2l)/auxprop+c1*(y2l+x2l)/auxprop):
                                htoday[u] = 1;
                                y1g -= 1; x1g += 1;
                            else:
                                htoday[u] = 0;
                                y1g -= 1; y2g += 1;
                        elif auxrand <= ((1-s2)*(x1l+x2l)/auxprop)/((1-s2)*(x1l+x2l)/auxprop+auxrand <= (1-c1)*(x1l+x2l)/auxprop):         
                            htoday[u] = 2;
                            y2g -= 1; x2g += 1;
                        else:
                            htoday[u] = 3;
                            y2g -= 1; y1g += 1;
                
                            
                xststate[m] = (x1g+x2g)/L;
                wststate[m] = (x1g-x2g)/L;
                zststate[m] = (y2g-y1g)/L;
            meanx.append(np.mean(xststate));
            stdx.append(np.std(xststate));
            meanw.append(np.mean(wststate));
            stdw.append(np.std(wststate));
            meanz.append(np.mean(zststate));
            stdz.append(np.std(zststate));
            
    return s1vec, s2vec, meanx, stdx, meanw, stdw, meanz, stdz


def evendegreelognormal(mu, sigma, L, repm):
    
    # Generates even arrays 50.0654% of the times

    "Generates a log-normal distribution with even sum of degrees."
    "mu is the mean, sigma the std of the inherent Gaussian." 
    "L the size of the degree vector"

    comp = 0; rep = 0;

    while comp == 0 and rep < repm:
        deg = np.around(np.absolute(np.random.lognormal(mu, sigma, L))).astype(int)
        if (sum(deg) % 2) == 0 and all(i < L for i in deg):
            comp = 1
        rep += 1
    return deg



def evendegreebeta(X0, beta, maxdeg, L, repm):
    
    # Generates even arrays 49.954% of the times

    "Generates a Beta distribution with even sum of degrees."
    "L the size of the degree vector"

    comp = 0; rep = 0;

    while comp == 0 and rep < repm:
        deg = np.around((maxdeg*np.absolute(np.random.beta(X0, beta, L)))).astype(int)
        if (sum(deg) % 2) == 0 and all(i < L for i in deg):
            comp = 1
        rep += 1
    return deg

def evendegreegamma(k, theta, L, repm):
    
    # Generates even arrays 49.9946% of the times

    "Generates a Beta distribution with even sum of degrees."
    "L the size of the degree vector"

    comp = 0; rep = 0;

    while comp == 0 and rep < repm:
        deg = np.around((np.absolute(np.random.gamma(k, theta, L)))).astype(int)
        if (sum(deg) % 2) == 0 and all(i < L for i in deg):
            comp = 1
        rep += 1
    return deg


def evendegreePoisson(lam, L, repm):
    
    # Generates even arrays 49.9828% of the times

    "Generates a Poisson distribution with even sum of degreeswith exponent lam"

    comp = 0; rep = 0;
    while comp == 0 and rep < repm:
        deg = np.around(np.random.poisson(lam, L)).astype(int)
        if (sum(deg) % 2) == 0 and all(i < L for i in deg):
            comp = 1
        rep += 1
    return deg

def evendegreeWeibull(a, lamb, L, repm):
    
    # Generates even arrays 49.9648% of the times

    "Generates a Weibull distribution with even sum of degreeswith exponent lam"

    comp = 0; rep = 0;
    while comp == 0 and rep < repm:
        deg = np.around(lamb*np.random.weibull(a, L)).astype(int)
        if (sum(deg) % 2) == 0 and all(i < L for i in deg):
            comp = 1
        rep += 1
    return deg

def evendegreeLomax(a, m, L, repm):
    
    # Generates even arrays 49.9858% of the times

    "Generates a Lomax distribution with even sum of degreeswith exponent lam"

    comp = 0; rep = 0;
    while comp == 0 and rep < repm:
        deg = np.around((m*(np.random.pareto(a, L)+1))).astype(int)
        if (sum(deg) % 2) == 0 and all(i < L for i in deg):
            comp = 1
        rep += 1
    return deg

# Adjacency matrix creation function following configuration model

def create_Adj(deg, vertices, L):
    
    A = np.zeros((L,L))
    vdist = []
    
    for v in vertices:
        for i in range(int(deg[v])):
            vdist.append(v)
    
    
    while len(vdist) > 0:
        u = np.random.choice(vdist)
        v = np.random.choice(vdist)
        if u != v:
            vdist.remove(v)
            vdist.remove(u)
            A[u][v] = 1
        else:
            if vdist.count(vdist[0]) == len(vdist):
                vdist = []
    
    A = A + A.transpose()
    
    return A, vdist


"""
## Barabási-Albert model (undirected)

Adj = np.zeros((L,L))

nmin = 4
ncon = 2

for i in range(nmin):
    for j in range(i+1, nmin):
        Adj[i][j] = 1
Adj = np.maximum(Adj, Adj.transpose()) # Fully connected initial network


for i in range(nmin, L):
    
    deg = np.array([sum(Adj[0])])
    vdist = np.array([0])
    
    for j in range(int(deg[0]-1)):
        vdist = np.append(vdist, 0)
    
    for j in range(1, i):
        deg = np.append(deg, np.array(sum(Adj[j])))
        for k in range(int(deg[j])):
            vdist = np.append(vdist, j)
            
    for j in range(ncon): 
        aux = np.random.choice(vdist)
        if Adj[i][aux] == 0:
            Adj[i][aux] = 1
            Adj[aux][i] = 1
            
deg = np.array([sum(Adj[0])])
vdist = np.array([0])

for j in range(int(deg[0]-1)):
    vdist = np.append(vdist, 0)

for j in range(1, L):
    deg = np.append(deg, np.array(sum(Adj[j])))
    for k in range(int(deg[j])):
        vdist = np.append(vdist, j)

"""



## Erdős–Rényi model

# Adjacency matrix

Adj = np.zeros((L,L))

for i in range(L):
    for j in range(i+1,L):
        if np.random.rand() < p:
            Adj[i][j] = 1

Adj = np.maximum(Adj, Adj.transpose())

deg = np.array(sum(Adj[0]))
for i in range(1, L):
    deg = np.append(deg, sum(Adj[i]))


"""
repm = 100; mu = 3; sigma = 1;

deg = evendegreelognormal(mu, sigma, L, repm)
vertices = np.linspace(0, L-1, L, dtype=int)
Adj = create_Adj(deg, vertices, L)
"""

T = 2000; # in MC steps
nrealizations = 10;
pX1 = 0.25; pX2 = 0.25; pY1 = 0.25; nss = 50;

write=1;
tic = time.time()
s1vec, s2vec, meanx, stdx, meanw, stdw, meanz, stdz = simsis(T, nrealizations, Adj, pX1, pX2, pY1, c1, c2, nss)

toc = time.time()
print(toc-tic, 's elapsed')

errmeanx = [aux/np.sqrt(nrealizations) for aux in stdx]
errmeanw = [aux/np.sqrt(nrealizations) for aux in stdw]
errmeanz = [aux/np.sqrt(nrealizations) for aux in stdz]


if write == 1:
    rows = zip(s1vec, s2vec, meanx, errmeanx, meanw, errmeanw, meanz, errmeanz)
    
    name = f"resultsnureddunageneral_er_L_{L}_p_{p}_c1_{c1}_c2_{c2}"
    
    f = open(name, "w")
    f.write(f"#Time elapsed: {toc-tic}\n")
    f.write(f"#c11: {c1}, c2: {c2}, px1: {pX1}, pX2: {pX2}, pY1: {pY1}, T max: {T}, n realizations: {nrealizations}, n nodes: {L}, ns: {nss}\n#s1_s2_x_errx\n")
    
    
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)
        
    f.close()






