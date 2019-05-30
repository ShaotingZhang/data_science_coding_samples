import random

from zen.graph import Graph
from zen.digraph import DiGraph
from zen.exceptions import ZenException

__all__ = ['erdos_renyi','barabasi_albert']

def barabasi_albert(n, m, **kwargs):
	"""
	Generate a random graph using the Barabasi-Albert preferential attachment model.
	
	**Args**:
	
		* ``n`` (int): the number of nodes to add to the graph
		* ``m`` (int): the number of edges a new node will add to the graph
	
	**KwArgs**:
		* ``directed [=False]`` (boolean): whether to build the graph directed.  If ``True``, then the ``m`` edges created
		  by a node upon its creation are instantiated as out-edges.  All others are in-edges to that node.
		* ``seed [=-1]`` (int): a seed for the random number generator
		* ``graph [=None]`` (:py:class:`zen.Graph` or :py:class:`zen.DiGraph`): this is the actual graph instance to populate. It must be
		  empty and its directionality must agree with the value of ``directed``.
	
	**Returns**:
		:py:class:`zen.Graph` or :py:class:`zen.DiGraph`. The graph generated.  If ``directed = True``, then a :py:class:`DiGraph` will be returned.
	
	.. note::
		Source: A. L. Barabasi and R. Albert "Emergence of scaling in random networks", Science 286, pp 509-512, 1999.
	"""
	seed = kwargs.pop('seed',None)
	directed = kwargs.pop('directed',False)
	graph = kwargs.pop('graph',None)
	
	if graph is not None:
		if len(graph) > 0:
			raise ZenException, 'the graph must be empty, if provided'
		if graph.is_directed() != directed:
			raise ZenException, 'graph and directed arguments must agree'
	else:
		if directed:
			graph = DiGraph()
		else:
			graph = Graph()
	
	if len(kwargs) > 0:
		raise ZenException, 'Unknown arguments: %s' % ', '.join(kwargs.keys())
		
	if seed is None:
		seed = -1
	
	if not directed:
		return __inner_barabasi_albert_udir(n, m, seed, graph)
	else:
		return __inner_barabasi_albert_dir(n, m, seed, graph)
	
def identity_fxn(i):
	return i	
	
def __inner_barabasi_albert_udir(n, m, seed, G):
	# add nodes
	G.add_nodes(m, identity_fxn)
	
	#####
	# add edges
	if seed >= 0:
		random.seed(seed)
	
	# add the first (m+1)th node
	G.add_node(m)
	for i in range(m):
		G.add_edge_(m,i)
	
	# add the remaining nodes
	num_endpoints = 2 * m
	for new_node_idx in range(m+1,n):
		G.add_node(new_node_idx)
		
		# this node drops m edges
		delta_endpoints = 0
		for e in range(m):
			rnd = random.random() * (num_endpoints-delta_endpoints)
			
			# now loop through nodes and find the one whose endpoint has the running sum
			# note that we ignore nodes that we already have a connection to
			running_sum = 0
			for i in range(new_node_idx):
				if G.has_edge_(new_node_idx,i):
					continue
					
				running_sum += G.degree_(i)
				if running_sum > rnd:
					G.add_edge_(new_node_idx,i)
					
					# this node can no longer be selected.  So we remove this node's degree
					# from the total degree of the network - making sure that a node will get
					# selected next time.  We decrease by 1 because the node's degree has just
					# been updated by 1 because it gained an endpoint from the node being
					# added.  This edge isn't included in the number of endpoints until the 
					# node has finished being added (since the node can't connect to itself).
					# As a result the delta endpoints must not include this edge either.
					delta_endpoints += G.degree_(i) - 1
					break
					
		num_endpoints += m * 2
		
	return G

def __inner_barabasi_albert_dir(n, m, seed, G):
	# add nodes
	G.add_nodes(m, identity_fxn)

	#####
	# add edges
	if seed >= 0:
		random.seed(seed)

	# add the first (m+1)th node
	G.add_node(m)
	for i in range(m):
		G.add_edge_(m,i)

	# add the remaining nodes
	num_endpoints = 2 * m
	for new_node_idx in range(m+1,n):
		G.add_node(new_node_idx)
		
		# this node drops m edges
		delta_endpoints = 0
		for e in range(m):
			rnd = random.random() * (num_endpoints-delta_endpoints)

			# now loop through nodes and find the one whose endpoint has the running sum
			# note that we ignore nodes that we already have a connection to
			running_sum = 0
			for i in range(new_node_idx):
				if G.has_edge_(new_node_idx,i):
					continue

				node_degree = G.in_degree(i) + G.out_degree_(i)
				running_sum += node_degree
				if running_sum > rnd:
					G.add_edge_(new_node_idx,i)
					
					# this node can no longer be selected.  So we remove this node's degree
					# from the total degree of the network - making sure that a node will get
					# selected next time.
					delta_endpoints += node_degree
					break

		num_endpoints += m * 2
		
	return G

def erdos_renyi(n, p, **kwargs):
	"""
	Generate an Erdos-Renyi graph.
	
	**Args**:
	 	* ``num_nodes`` (int): the number of nodes to populate the graph with.
	 	* ``p`` (0 <= float <= 1): the probability p given to each edge's existence.
	
	**KwArgs**:
		* ``directed [=False]`` (boolean): indicates whether the network generated is directed.
		* ``self_loops [=False]`` (boolean): indicates whether self-loops are permitted in the generated graph.
		* ``seed [=-1]`` (int): the seed provided to the random generator used to drive the graph construction.
		* ``graph [=None]`` (:py:class:`zen.Graph` or :py:class:`zen.DiGraph`): this is the actual graph instance to populate. It must be
		  empty and its directionality must agree with the value of ``directed``.
	"""
	directed = kwargs.pop('directed',False)
	self_loops = kwargs.pop('self_loops',False)
	seed = kwargs.pop('seed',None)
	graph = kwargs.pop('graph',None)
	
	if graph is not None:
		if len(graph) > 0:
			raise ZenException, 'the graph must be empty, if provided'
		if graph.is_directed() != directed:
			raise ZenException, 'graph and directed arguments must agree'
	else:
		if directed:
			graph = DiGraph()
		else:
			graph = Graph()
			
	if len(kwargs) > 0:
		raise ZenException, 'Unknown arguments: %s' % ', '.join(kwargs.keys())
		
	if seed is None:
		seed = -1
	
	if directed:
		return __erdos_renyi_directed(n,p,self_loops,seed,graph)
	else:
		return __erdos_renyi_undirected(n,p,self_loops,seed,graph)

def __erdos_renyi_undirected(num_nodes, p, self_loops, seed, G):
	if seed >= 0:
		random.seed(seed)
	
	# add nodes
	for i in range(num_nodes):
		G.add_node(i)
		
	# add edges
	for i in range(num_nodes):
		if self_loops:
			first_j = i
		else:
			first_j = i+1
			
		for j in range(first_j,num_nodes):
			rnd = random.random()
			if rnd < p:
				G.add_edge_(i,j)
	
	return G
	
def __erdos_renyi_directed(num_nodes, p, self_loops, seed, G):
	if seed >= 0:
		random.seed(seed)
	
	# add nodes
	for i in range(num_nodes):
		G.add_node(i)
	
	# add edges
	for i in range(num_nodes):
		for j in range(num_nodes):
			if i == j and not self_loops:
				continue
				
			rnd = random.random()
			if rnd < p:
				G.add_edge_(i,j)

	return G