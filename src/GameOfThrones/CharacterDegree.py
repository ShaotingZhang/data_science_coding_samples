import zen
import scipy.integrate as spi
import matplotlib.pyplot as plt
plt.ioff()
import numpy
from math import e
from numpy import *
from numpy.linalg import eig,norm
import sys
sys.path.append('../zend3js/')
import d3js
from time import sleep
import colorsys
import numpy.linalg as la


G = zen.io.gml.read('GameOfThrones5.gml',weight = True, directed = False)

#d3 = d3js.D3jsRenderer(G, interactive = False, autolaunch = False)
#d3.update()
#d3.stop_server()

A = G.matrix()
N = G.num_nodes

def degreeineachseason(name, G):
	if name not in G: return 0
	A = G.matrix()
	index = G.node_idx(name)
	nei = G.neighbors_(index)
	degree = 0
	for i in nei:
		degree += A[index][i]
	return degree

names = ['Jon', 'Cersei', 'Daenerys', 'Sansa', 'Arya', 'Bran']

listdegree = []
for i in range(1,7):
	filename = 'GameOfThrones' + str(i) + '.gml'
	Gd = zen.io.gml.read(filename,weight = True, directed = False)
	each = []
	for name in names:
		degree = degreeineachseason(name, Gd)
		each.append(degree)
	listdegree.append(each)
season = [1, 2, 3, 4, 5, 6]

plt.figure()
plt.title('Sansa Stark')
plt.plot(season,[i[3] for i in listdegree],'k')
plt.xlabel('Season')
plt.ylabel('Degree')
plt.show()
	
plt.figure()
plt.title('Arya Stark')
plt.plot(season,[i[4] for i in listdegree],'r')
plt.xlabel('Season')
plt.ylabel('Degree')
plt.show()
	
plt.figure()
plt.title('BranStark')
plt.plot(season,[i[5] for i in listdegree],'b')
plt.xlabel('Season')
plt.ylabel('Degree')
plt.show()
	
plt.figure()
plt.title('Jon Snow')
plt.plot(season,[i[0] for i in listdegree],'g')
plt.xlabel('Season')
plt.ylabel('Degree')
plt.show()
	
plt.figure()
plt.title('Cersei Lannister')
plt.plot(season,[i[1] for i in listdegree],'y')
plt.xlabel('Season')
plt.ylabel('Degree')
plt.show()
	
plt.figure()
plt.title('Daenerys Targaryen')
plt.plot(season,[i[2] for i in listdegree],'r')
plt.xlabel('Season')
plt.ylabel('Degree')
plt.show()


