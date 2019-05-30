import zen
import scipy.integrate as spi
import matplotlib.pyplot as plt
plt.ioff()
import numpy
from math import e
from numpy import *
from numpy.linalg import eig,norm
import sys
sys.path.insert(0, '../../zend3js/')
import d3js
from time import sleep
import colorsys
import numpy.linalg as la


def print_top(G,v, num=10):
    idx_list = [(i,v[i]) for i in range(len(v))]
    idx_list = sorted(idx_list, key = lambda x: x[1], reverse=True)
    for i in range(min(num,len(idx_list))):
        nidx, score = idx_list[i]
        print ("%i. %s (%1.4f)" % (i+1,G.node_object(nidx),score))

def index_of_max(v):
    return numpy.where(v == max(v))[0]

def calc_powerlaw(G,kmin):
    N = G.num_nodes
    ddist = zen.degree.ddist(G,normalize=False)
    cdist = zen.degree.cddist(G,inverse=True)
    k = numpy.arange(len(ddist))
    
    plt.figure(figsize=(10,8))
    plt.subplot(211)
    plt.bar(k,ddist, width=0.8, bottom=0, color='b')

    plt.subplot(212)
    plt.loglog(k,cdist)

    sub = 0
    for z in range(0,len(ddist) - 1):
        if z < kmin:
            sub = sub + ddist[z]
    N = G.num_nodes - sub

    sum = 0
#     print ddist
#     print len(ddist)
    for k_i in range(kmin, len(ddist) - 1):
        fraction = k_i / (kmin - 0.5)
        iLn = ddist[k_i] * math.log(fraction,e)
        sum = sum + iLn
        if (sum != 0):
            sum_inv = 1/sum
    alpha = 1 + N * sum_inv # calculate using formula!
    print ('alpha is %1.2f' %alpha)   
    plt.show()

def modularity(G,c):
    d = dict()
    for k,v in c.iteritems():
        for n in v:
            d[n] = k
    Q, Qmax = 0,1
    for u in G.nodes_iter():
        for v in G.nodes_iter():
            if d[u] == d[v]:
                Q += ( int(G.has_edge(v,u)) - G.in_degree(u)*G.out_degree(v)/float(G.num_edges) )/float(G.num_edges)
                Qmax -= ( G.in_degree(u)*G.out_degree(v)/float(G.num_edges) )/float(G.num_edges)
    return Q, Qmax

def propagate(G,d3,x,steps,slp=0.5,keep_highlights=False,update_at_end=False):
    interactive = d3.interactive
    d3.set_interactive(False)
    A = G.matrix().T  # adjacency matrix of the network G
    d3.highlight_nodes_(list(where(x>0)[0]))
    d3.update()
    sleep(slp)
    cum_highlighted = sign(x)
    for i in range(steps): # the brains
        x = sign(dot(A,x)) # the brains
        cum_highlighted = sign(cum_highlighted+x)
        if not update_at_end:
            if not keep_highlights:
                d3.clear_highlights()
            d3.highlight_nodes_(list(where(x>0)[0]))
            d3.update()
            sleep(slp)
    if update_at_end:
        if not keep_highlights:
            d3.clear_highlights()
            d3.highlight_nodes_(list(where(x>0)[0]))
        else:
            d3.highlight_nodes_(list(where(cum_highlighted>0)[0]))
        d3.update()
    d3.set_interactive(interactive)
    if keep_highlights:
        return cum_highlighted
    else:
        return x

def main():
    G = zen.io.gml.read('GameOfThrones.gml',weight = True, directed = False)


    #d3 = d3js.D3jsRenderer(G, interactive=False, autolaunch=False)
    d3 = d3js.D3jsRenderer(G, interactive=False)
    d3.update()
    sleep(1)


    A = G.matrix()
    N = G.num_nodes

    print ('\n=============================================')
    print ('\nDegree Centrality:')
    vv = [0] * N
    for i in range(N):
            v1 = G.neighbors_(i)
            sum = 0
            for j in range(len(v1)):
                    sum += G.weight(G.node_object(i),G.node_object(v1[j]))
            vv[i] = sum
    print_top(G,vv)

    print ('\n=============================================')
    # Eigenvector Centrality
    print ('\nEigenvector Centrality (by Zen):')
    v2 = zen.algorithms.centrality.eigenvector_centrality_(G,weighted = True)
    print_top(G,v2)

    print ('\n=============================================')
    # Betweenness Centrality
    print ('\nBetweenness Centrality')
    v = zen.algorithms.centrality.betweenness_centrality_(G)
    print_top(G,v)


    print ('\n==============================================')
    print ('\nPOWER LAW')
    calc_powerlaw(G,3)    # need to change kmin appropriately


    print ('\n==============================================')
    print ('\nClustering Coefficients')
    c = zen.algorithms.clustering.gcc(G)
    print ('Clustering: %s' % c)


    print ('\n==============================================')
    print ('\nDiameter and Path')
    D,P = zen.algorithms.shortest_path.all_pairs_dijkstra_path_(G)
    d = 0
    for i in range(len(D)):
            for j in range(len(D[0])):
                    if D[i][j] > d:
                            d = D[i][j]


    uidx,vidx = where(D==d)
    path = zen.algorithms.shortest_path.pred2path_(uidx[1],vidx[1],P)
    pathname = []
    for i in path:
            a = G.node_object(i)
            pathname.append(a)
    d = len(path)
    print ('The network has a diameter of %i.' % d)
    print (pathname)
           
    # code to visualize the path for i in range(len(path) + 1):
    for i in range(len(path) - 1):
         a = path[i]
         b = path[i + 1]
         c = G.edge_idx_(a,b)    
         d3.highlight_edges_([c])
         d3.update()
    sleep(3) 


    d3.stop_server()

if __name__ == '__main__':
    main()
