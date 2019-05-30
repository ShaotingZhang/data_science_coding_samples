import zen
import d3js
import random
import math
from time import sleep

G = zen.generating.erdos_renyi(100,0.03)
#G = zen.io.gml.read('2013-actor-actor.gml')
#G = zen.io.edgelist.read('japanese.edgelist', ignore_duplicate_edges=True)
print G.num_edges

d3 = d3js.D3jsRenderer(G, event_delay=0.03, interactive=False,
						node_dstyle=d3js.node_style(size=4), 
						node_hstyle=d3js.node_style(fill='#EB4343'),
						edge_hstyle=d3js.edge_style(stroke='#EB4343',stroke_width=5))

d3.update()
sleep(2)

# make the visualization update after every step
d3.set_interactive(True)

d3.set_title('Positioning Nodes...')
for nidx in G.nodes_iter_():
	x = 250 + 230 * math.cos(math.radians(360.0*nidx/G.num_nodes))
	y = 300 + 150 * math.sin(math.radians(360.0*nidx/G.num_nodes))
	d3.position_node(nidx,x,y)

sleep(2)

d3.set_title('Highlighting Edges...')
p = 0.1
active_edges = set()
for eidx in G.edges_iter_():
	if random.random() <= p:
		active_edges.add(eidx)
		uidx,vidx = G.endpoints_(eidx)
		d3.highlight_edges_([eidx])
		d3.highlight_nodes_([uidx,vidx])
d3.highlight_edges_(active_edges)

sleep(2)

d3.set_title('Adding and styling a single node...')
nidx = G.add_node('new node')
G.add_edge_(0,nidx)
d3.stylize_node('new node', d3js.node_style(fill='#FFA929',size=12))

sleep(3)

d3.set_title('Switching to a directed network...')
G2 = zen.generating.barabasi_albert(100,3,directed=True)
d3.set_interactive(False)
d3.set_graph(G2)

d3.update()


d3.stop_server()
