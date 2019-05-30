import socket, threading
from hashlib import sha1
from base64 import b64encode
from struct import pack
from json import dumps
from time import sleep
from webbrowser import open_new
from os import getcwd
from os.path import join as pathjoin
from sys import path as pythonpath

from zen.graph import Graph
from zen.digraph import DiGraph

# set node location (x,y)
# directed edges

_STYLE_SHAPE = 'shape'
_STYLE_SIZE = 'size'
_STYLE_FILL = 'fill'
_STYLE_STROKE = 'stroke'
_STYLE_STROKE_WIDTH = 'strokewidth'

def node_style(shape=None, size=None, fill=None, stroke=None, stroke_width=None):
	style = {_STYLE_SHAPE:"circle",_STYLE_SIZE:8,_STYLE_FILL:'#77BEF5',_STYLE_STROKE:'#FFFFFF',_STYLE_STROKE_WIDTH:2}
	if shape:
		style[_STYLE_SHAPE] = shape
	if size:
		style[_STYLE_SIZE] = size
	if fill:
		style[_STYLE_FILL] = fill
	if stroke:
		style[_STYLE_STROKE] = stroke
	if stroke_width:
		style[_STYLE_STROKE_WIDTH] = stroke_width
	return style

def edge_style(stroke=None, stroke_width=None):
	style = {_STYLE_STROKE:'#494949',_STYLE_STROKE_WIDTH:2}
	if stroke:
		style[_STYLE_STROKE] = stroke
	if stroke_width:
		style[_STYLE_STROKE_WIDTH] = stroke_width
	return style

class D3jsRenderer(object):

	MAGICKEY = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11'
	
	def __init__(self,graph,event_delay=0.03,interactive=True,
				canvas_size=(800,600),autolaunch=True,port=9876,
				node_dstyle=None,edge_dstyle=None,node_hstyle=None,edge_hstyle=None):
		
		if autolaunch:
			zendir = ''
			for d in pythonpath:  # look for the zend3js folder in the python path
				if 'zend3js' in d:
					zendir = d
					break
			if zendir == getcwd():  # if running the program from the same folder as the zend3js folder
				zendir = ''
			#print 'launching webbrowser!'
			p = pathjoin('file://'+getcwd(),zendir,"index.html?width=%i&height=%i" % canvas_size )
			open_new(p)
		
		self.server_socket = None
		self.d3js_socket = None
		self.event_delay = event_delay
		self.port = port
		self.interactive = interactive
		
		self.default_node_style = node_dstyle
		self.default_edge_style = edge_dstyle
		self.highlighted_node_style = node_hstyle
		self.highlighted_edge_style = edge_hstyle
		if not self.default_node_style:
			self.default_node_style = node_style()
		if not self.default_edge_style:
			self.default_edge_style = edge_style()
		if not self.highlighted_node_style:
			self.highlighted_node_style = node_style(size=10,fill='#FD5411')
		if not self.highlighted_edge_style:
			self.highlighted_edge_style = edge_style(stroke='#F79C00',stroke_width=4)
		self._highlighted_nodes = set([])
		self._highlighted_edges = set([])
		
		self.start_server()
		self.set_graph(graph)

	def set_graph(self,graph):
		self.clear()
		self._graph = graph
		self._directed = isinstance(graph,DiGraph)
		self._graph.add_listener(self)
		self._load_existing_graph()

	def _load_existing_graph(self):
		# Add nodes and edges if graph already has them
		if self._graph.num_nodes > 0:
			for nidx in self._graph.nodes_iter_():
				self.node_added(nidx,self._graph.node_object(nidx),self._graph.node_data_(nidx))
			for eidx in self._graph.edges_iter_():
				uidx,vidx = self._graph.endpoints_(eidx)
				self.edge_added(eidx,uidx,vidx,self._graph.edge_data_(eidx),self._graph.weight_(eidx))

	def set_title(self,newtitle):
		ti = {'titlename':newtitle}
		self._send_update('ti'+dumps(ti))
		self.update()

	def clear(self):
		self._send_update('cc')
		self.update()
	
	def update(self):
		self._send_update('up')
		
	def set_event_delay(self,delay):
		self.event_delay = delay
		
	def set_interactive(self,interactive):
		self.interactive = interactive
	
	# Setting the position of nodes
	def position_node_(self,nidx,x,y):
		mv_node = {'nid':nidx, 'fixed':True, 'cx':x, 'cy':y}
		self._send_update('mn'+dumps(mv_node))

	def position_node(self,nobj,x,y):
		self.position_node_(self._graph.node_idx(nobj),x,y)

	# data is a list of tuples with form (node_index, x, y)
	def position_nodes_(self,data):
		for nidx,x,y in data:
			self.position_node_(nidx,x,y)
	
	# data is a list of tuples with form (node_object, x, y)
	def position_nodes(self,data):
		for nobj,x,y in data:
			self.position_node(nobj,x,y)

	# Adding style to nodes
	def stylize_node_(self,nidx,style_dict):
		self._send_update('!n'+ self._node_update(nidx,self._graph.node_object(nidx),style_dict) )

	def stylize_node(self,nobj,style_dict):
		self.stylize_node_(self._graph.node_idx(nobj),style_dict)

	def stylize_nodes_(self,nidxs,style_dict):
		for nidx in nidxs:
			self.stylize_node_(nidx,style_dict)

	def stylize_nodes(self,nobjs,style_dict):
		for nobj in nobjs:
			self.stylize_node(nobj,style_dict)

	# Adding style to edges
	def stylize_edge_(self,eidx,style_dict):
		uidx, vidx = self._graph.endpoints_(eidx)
		self._send_update('!e'+ self._edge_update(eidx,uidx,vidx,style_dict) )

	def stylize_edges_(self,eidxs,style_dict):
		for eidx in eidxs:
			self.stylize_edge_(eidx,style_dict)

	# Automatically called, do not call directly
	def node_added(self,nidx,nobj,data):
		self._send_update('+n'+ self._node_update(nidx,nobj) )
	
	# Automatically called, do not call directly
	def node_removed(self,nidx,nobj):
		rm_node = {'nid': nidx}
		self._send_update('-n'+dumps(rm_node))
		
	# Automatically called, do not call directly
	def edge_added(self,eidx,uidx,vidx,data,weight):
		self._send_update('+e'+ self._edge_update(eidx,uidx,vidx) )
	
	# Automatically called, do not call directly
	def edge_removed(self,eidx,uidx,vidx):
		rm_edge = {'eid': eidx}
		self._send_update('-e'+dumps(rm_edge))
	
	# Highlighting Nodes
	def highlight_nodes_(self,nodes):
		self.stylize_nodes_(nodes,self.highlighted_node_style)
		self._highlighted_nodes.update(set(nodes))

	def highlight_edges_(self,edges):
		self.stylize_edges_(edges,self.highlighted_edge_style)
		self._highlighted_edges.update(set(edges))

	def highlight_edges(self,edges):
		eidxs = map(lambda x: self._graph.edge_idx(*x),edges)
		self.highlight_edges_(eidxs)

	def highlight_nodes(self,nodes):
		nidxs = map(lambda x: self._graph.node_idx(x),nodes)
		self.highlight_nodes_(nidxs)
		
	def clear_highlights(self):
		self.stylize_nodes_(self._highlighted_nodes,self.default_node_style)
		self.stylize_edges_(self._highlighted_edges,self.default_edge_style)
		self._highlighted_nodes = set([])
		self._highlighted_edges = set([])

	# Helper functions for sending updates
	def _node_update(self,nidx,nobj,style_dict={}):
		update_node = {'nid': nidx}
		update_node['ntitle'] = ''
		if nobj is not None:
			update_node['ntitle'] = str( nobj )
		final_style = self.default_node_style.copy()
		final_style.update(style_dict)
		update_node.update(final_style)
		return dumps(update_node)

	def _edge_update(self,eidx,uidx,vidx,style_dict={}):
		update_edge = {'source':uidx, 'target':vidx}
		update_edge['eid'] = eidx
		update_edge['directed'] = int(self._directed)
		final_style = self.default_edge_style.copy()
		final_style.update(style_dict)
		update_edge.update(final_style)
		return dumps(update_edge)
		
	# Socket Communication
	def start_server(self):
		self.server_socket = socket.socket()
		self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		self.server_socket.bind(('', self.port))
		self.server_socket.listen(1)

		self.d3js_socket, addr = self.server_socket.accept()
		#print 'Connected by', addr
		self._handshake()

	def stop_server(self):
		self.d3js_socket.close()
		self.server_socket.close()
			
	def _decode(self,data):
		frame = bytearray(data)

		length = frame[1] & 127

		indexFirstMask = 2
		if length == 126:
			indexFirstMask = 4
		elif length == 127:
			indexFirstMask = 10

		indexFirstDataByte = indexFirstMask + 4
		mask = frame[indexFirstMask:indexFirstDataByte]

		i = indexFirstDataByte
		j = 0
		decoded = []
		while i < len(frame):
			decoded.append(frame[i] ^ mask[j%4])
			i += 1
			j += 1
	
		return "".join(chr(byte) for byte in decoded)

	def _encode(self,buf, opcode=0x1, b64=False):
		""" Encode a HyBi style WebSocket frame.
		Optional opcode:
			0x0 - continuation
			0x1 - text frame (base64 encode buf)
			0x2 - binary frame (use raw buf)
			0x8 - connection close
			0x9 - ping
			0xA - pong
		"""
		if b64:
			buf = b64encode(buf)

		b1 = 0x80 | (opcode & 0x0f) # FIN + opcode
		payload_len = len(buf)
		if payload_len <= 125:
			header = pack('>BB', b1, payload_len)
		elif payload_len > 125 and payload_len < 65536:
			header = pack('>BBH', b1, 126, payload_len)
		elif payload_len >= 65536:
			header = pack('>BBQ', b1, 127, payload_len)

		return header + buf

	def _handshake(self):
		data = self.d3js_socket.recv(1024)
		key = ''
		for line in data.split('\n'):
			if line.lower().startswith('sec-websocket-key'):
				key = line.split(':')[1].strip()
		key = key + self.MAGICKEY
		key = sha1(key).digest()
		key = b64encode(key)
	
		handshake = "HTTP/1.1 101 Web Socket Protocol Handshake\r\nUpgrade: WebSocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: %s\r\nWebSocket-Protocol: chat\r\n\r\n" % (key)
		self.d3js_socket.send(handshake)
		data = self.d3js_socket.recv(1024)
	
	def _send_update(self,update):
		update.replace('\'','\"')
		#print update
		self.d3js_socket.sendall(self._encode(update))
		if self.interactive:
			self.d3js_socket.sendall(self._encode('up'))
			sleep(self.event_delay)


if __name__ == '__main__':
	import zen
	import time

	G = zen.Graph()
	#G = zen.generating.local_attachment(40, 6, 4, graph=G)
	
	d3 = D3jsRenderer(G, event_delay=0.03)
	
	d3.start_server()

	n0 = G.add_node(0)
	#time.sleep(1)
	n1 = G.add_node(1)
	#time.sleep(1)
	e1 = G.add_edge(0,1)
	time.sleep(1)
	n2 = G.add_node(2)
	#time.sleep(1)
	e2 = G.add_edge(1,2)
	#time.sleep(1)
	n3 = G.add_node(3)
	#time.sleep(1)
	e3 = G.add_edge(2,3)
	#time.sleep(1)
	e4 = G.add_edge(3,1)
	#time.sleep(1)
	e5 = G.add_edge(1,4)
	
	# time.sleep(2)
	d3.stylize_node_(0,d3.node_style(size=12,fill='#FF0000'))
	time.sleep(0.5)
	#d3.stylize_node_(1,d3.node_style(fill='#FF0000'))
	d3.stylize_edge_(e1,d3.edge_style(stroke_width=5))
	#time.sleep(0.5)
	
	# d3.highlight_nodes_(range(1,37,3))
	# d3.stylize_node(26,node_style(fill='#70EB37',size=16))
	# 
	# hedges = []
	# for eidx in G.edges_iter_():
	# 	if eidx % 4 == 0:
	# 		hedges.append(eidx)
	# d3.highlight_edges_(hedges)
	
	# for nidx in G.nodes_iter_():
	# 	d3.position_node_(nidx,5*nidx,5*nidx)
	
	# f = open('lesmis.edgelist')
	# for line in f.readlines():
	# 	u,v = line.split()
	# 	G.add_edge(u,v)#,weight=float(w))
	# f.close()
	
	# time.sleep(3)
	# G.rm_node_(2)
	
	d3.stop_server()
	#ur.highlight_edges([(1,2),(2,3)])
	#ur.highlight_nodes([1])
