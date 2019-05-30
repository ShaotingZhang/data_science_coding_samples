import zen
import urllib
import time
import unicodedata
import numpy
import scipy.integrate as spi
import matplotlib.pyplot as plt
from math import e
from numpy import *
from numpy.linalg import eig,norm
import sys
sys.path.append('../zend3js/')
import d3js
from time import sleep
import colorsys
import numpy.linalg as la


def add_edge(names,G):      #  add edge into graph
	for i in range(len(names)-1):
		for j in range(i+1,len(names)):
			if G.has_edge(names[i],names[j]):
#				print 'exist'
				eidx = G.edge_idx(names[i],names[j])
				G.set_weight_(eidx,G.weight_(eidx)+1)
			else:
#				print 'new'
				G.add_edge(names[i],names[j],weight=1)

def edit_name(names,G):   # rewrite the name to compare nodes
	for i in range(len(names)):
		d = names[i].find('(')
		if d > 1:
			names[i] = names[i][:d-1]
	names = list(set(names))

	temp = []
	for name in names:		
		space = name.find(' ')
		if space < 0:
			name = name[0].upper() + name[1:].lower()
		else:
			name = name[0].upper() + name[1:space].lower()
		temp.append(name)
	names = temp

	for i in range(len(names)):        # add nodes to graph
		if names[i] == 'High':
			names[i] = 'Sparrow'
		if names[i] not in G:
			G.add_node(names[i])

#	print names
	return names

def GenerateGraph(filename,G):       #  for each file to generate the graph
	f = open(filename)
	lines = f.readlines()
	f.close
	names = []
	for line in lines:
		if line[0] == '-':
			names = edit_name(names,G)
			if len(names) > 1:
				add_edge(names,G)
			names = []
	
		p = line.find(':')
		if p >= 0:
			name = line[:p]
			if name != 'CUT TO':
				if name.find('\xa1') < 0:
					names.append(name)
	return G


def generate_graph():
    address = []
    address.append('http://gameofthrones.wikia.com/wiki/Category:Characters?display=page&sort=mostvisited')
    address.append('http://gameofthrones.wikia.com/wiki/Category:Characters?display=page&sort=mostvisited&pagefrom=Forel%2C+Syrio%0ASyrio+Forel#mw-pages')
    address.append('http://gameofthrones.wikia.com/wiki/Category:Characters?display=page&sort=mostvisited&pagefrom=Lannister%2C+Tommen+II%0ATommen+II+Lannister#mw-pages')
    address.append('http://gameofthrones.wikia.com/wiki/Category:Characters?display=page&sort=mostvisited&pagefrom=Selmy%2C+Barristan%0ABarristan+Selmy#mw-pages')
    #address.append('http://gameofthrones.wikia.com/wiki/Category:Characters?display=page&sort=mostvisited&pagefrom=White+Walker+%28Winter+is+Coming%29#mw-pages')

    PAGE_START = 'mw-pages'
    PAGE_END = 'class=\"bodyAd'

    CATEGORY_BREAK = '/wiki/'
    CATEGORY_URL = 'http://gameofthrones.wikia.com/wiki/'

    G = zen.Graph()
    nodes = []
    index = [2,3,3,2]

    for i in range(4):
        url = address[i]
        f = urllib.urlopen(url)
        s = f.read()
        f.close()

        contents = s.split(PAGE_START)[index[i]] # keep only the part after the first occurance
        contents = contents.split(PAGE_END)[0]
        hrefs = contents.split('<a')[1:]

        for href in hrefs:
            href = href.split('</a>')[0]
            if href.find(CATEGORY_BREAK) >= 0:
                caturl = href.split(CATEGORY_BREAK)[1]
                temp1 = caturl.find('title')
                if temp1 != -1:
                    caturl = caturl[:temp1-2]
                    nodes.append(caturl)

    #    print len(nodes)
    #print nodes
    #for i in range(len(nodes)):    # add nodes to graph
    #   node = nodes[i]
    #   if node not in G:
    #       G.add_node(node)

    for i in range(1,7):
            filename = 'script' + str(i) + '.txt'
            G = GenerateGraph(filename,G)

    listnames = []
    for i in range(G.num_nodes):
            listnames.append(G.node_object(i))
    #print listnames
    #print G.num_nodes
    #print G.num_edges

    zen.io.gml.write(G,'GameOfThrones.gml')

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


def analysis():
    G = zen.io.gml.read('GameOfThrones.gml',weight = True, directed = False)


    d3 = d3js.D3jsRenderer(G, interactive=False, autolaunch=False)
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

def RGBToHTMLColor(rgb_tuple):
	""" convert an (R, G, B) tuple to #RRGGBB """
	hexcolor = '#%02x%02x%02x' % rgb_tuple
	# that's it! '%02x' means zero-padded, 2-digit hex values
	return hexcolor

def HTMLColorToRGB(colorstring):
	""" convert #RRGGBB to an (R, G, B) tuple """
	colorstring = colorstring.strip()
	if colorstring[0] == '#': colorstring = colorstring[1:]
	if len(colorstring) != 6:
		raise ValueError, "input #%s is not in #RRGGBB format" % colorstring
	r, g, b = colorstring[:2], colorstring[2:4], colorstring[4:]
	r, g, b = [int(n, 16) for n in (r, g, b)]
	return (r, g, b)

def color_interp(color1,color2,v,m=0,M=1):
	c1 = array(HTMLColorToRGB(color1))
	c2 = array(HTMLColorToRGB(color2))
	if v > M:
		c = tuple(c2)
	elif v < m:
		c = tuple(c1)
	else:
		#c = tuple( c1 + (c2-c1)/(M-m)*(v-m) ) # linear interpolation of color
		c = tuple( c1 + (c2-c1)*(1 - exp(-2*(v-m)/(M-m))) ) # logistic interpolation of color
	return RGBToHTMLColor(c)

def color_by_value(d3,G,x,color1='#77BEF5',color2='#F57878'):
	d3.set_interactive(False)
	m = min(x)
	M = max(x)
	for i in G.nodes_iter_():
		d3.stylize_node_(i, d3js.node_style(fill=color_interp(color1,color2,x[i])))
	d3.update()
	d3.set_interactive(True)

def diffusion():
     G = zen.io.gml.read('Diffusion_script.gml',weight = True, directed = False)

    d3 = d3js.D3jsRenderer(G, interactive=False, autolaunch=False)
    d3.update()
    sleep(1)

    dt = 0.05 # the "infintesimal" size steps we take to integrate
    T = 6 # the end of the simulation time
    time = linspace(0,T,int(T/dt)) # the array of time points spaced by dt

    print '============================\nDIFFUSION\n'
    N = G.num_nodes
    d = [0] * N
    for i in range(N):    # compute degree
            v1 = G.neighbors_(i)
            summ = 0
            for j in range(len(v1)):
                    summ += G.weight(G.node_object(i),G.node_object(v1[j]))
            d[i] = summ
    D = numpy.diag(d)
    A = G.matrix()
    L = D - A

    x = zeros(G.num_nodes) # the state vector
    x[4] = 30
    x[3] = 30
    x[7] = 30
    x[10] = 30
    color_by_value(d3,G,x) # this colors the network according to the value of x
    sleep(2)

    print 'simulating diffusion...'


    for i,t in enumerate(time):      # each time
            x = x + 0.1 * numpy.dot((A - D),x) * dt    # compute  state X
            color_by_value(d3,G,x)
            sleep(0.8)
            
    d3.stop_server()

def main():
    generate_graph()

