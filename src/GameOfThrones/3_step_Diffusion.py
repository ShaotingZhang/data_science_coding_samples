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



def main():
    G = zen.io.gml.read('Diffusion_script.gml',weight = True, directed = False)

    #d3 = d3js.D3jsRenderer(G, interactive=False, autolaunch=False)
    d3 = d3js.D3jsRenderer(G, interactive=False)
    d3.update()
    sleep(1)

    dt = 0.05   # the "infintesimal" size steps we take to integrate
    T = 6    # the end of the simulation time
    time = linspace(0,T,int(T/dt))  # the array of time points spaced by dt

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

if __name__ == '__main__':
    main()
