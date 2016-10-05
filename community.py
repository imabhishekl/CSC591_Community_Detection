"""
	Name :: UnityId
	Shruti Kuber :: skuber
	Abhishek Lingwal :: aslingwa
	Raunaq Saxena :: rsaxena
"""

import numpy as np
import networkx as nx
from scipy.spatial import distance
import sys

#Structure for Nodes of Dendogram

class Node:
    def __init__(self, name):
        self.name = name
        self.left = None
        self.right = None
        self.parent = None
        self.num_edges = 0
        self.vertices = set()
        self.density = 0
        
        
class Tree:
	def __init__(self):
		self.root = None

	#Find Lowest Common Ancestor
	def findLCA_Node(self,src_node,dest_node):
		while src_node is not None:
			if dest_node.name in src_node.vertices:
				return src_node
			src_node = src_node.parent
		return None

	def print_Tree(self,root):
		if root==None:
			return
		print(root.vertices)
		self.print_Tree(root.left)
		self.print_Tree(root.right)

	def print_nodes(self,nodes):
		for i in range(0,len(nodes)):
			print(nodes[i].name,nodes[i].parent.name)

	def count_vertices_and_edges(self,edges_list,nodes_list):		
		"""
		Count all edges of nodes in a dendogram
		"""
		for edge in edges_list:
			lca_node = None
			#print edge[0],edge[1]
			#lca_node = self.findLCA_Node(next((x for x in nodes_list if x.name == edge[0]),None),next((y for y in nodes_list if y.name == edge[1]),None))
			src_node = nodes_list[edge[0]] if nodes_list.__contains__(edge[0]) else None
			dst_node = nodes_list[edge[1]] if nodes_list.__contains__(edge[1]) else None
			if src_node is not None and dst_node is not None:
				lca_node = self.findLCA_Node(src_node,dst_node )
			#print 'LCA:',lca_node
			if lca_node is not None:
				lca_node.num_edges = lca_node.num_edges + 1

	def count_vertices_and_edges_wrap(self,root):
		"""
		Summation of number of edges of children to parent
		"""
		if root.left != None and root.right != None:
			self.count_vertices_and_edges_wrap(root.left)
			self.count_vertices_and_edges_wrap(root.right)
		if root.left != None and root.right != None:
			root.num_edges = root.left.num_edges + root.right.num_edges + root.num_edges
			#print root.name, root.num_edges

	def compute_density(self,root):
		"""
		Computing Density of an undirected Graph
		"""
		if root.left is None and root.right is None:
			return 
		total_vertices = float(len(root.vertices))
		max_vertices = total_vertices*(total_vertices - 1)/2		
		root.density = root.num_edges/max_vertices
		self.compute_density(root.left)
		self.compute_density(root.right)

	def extract_sub_graph(self,root,min_density):
		"""
		Print nodes whose density is greater than Density threshold ,ie , min_density
		For other nodes look test the densities of their sub graphs/children
		"""
		if root is None:
			return
		if root.density > min_density:
			print "Community Detected:",root.vertices
		else:
			self.extract_sub_graph(root.left,min_density)
			self.extract_sub_graph(root.right,min_density)

def MakeSet(r):
	r.parent = None
	r.vertices.add(r.name)
	

def SetFind(r):
	while r.parent!=None:
		r = r.parent
	return r

# Building a new Node as Union of two sets
def SetUnion(x,y):
    r = Node("P"+ str(x.name) + str(y.name))
    r.left = x
    r.right = y
    x.parent = r
    y.parent = r
    r.vertices= r.vertices.union(x.vertices,y.vertices)        
    return r

min_threshold = 0.75
if(len(sys.argv)>2):
	graph_file = open(sys.argv[1])	
	min_threshold = float(sys.argv[2])
else:
	print "Please enter graph file and density threshold as argument"
	#graph_file = open("./amazon/amazon.graph.small")

#Build Edges and Vertices List from file

edges = graph_file.read().splitlines()

vertices = []
#print "Create vertex"
for edge in edges[1:]:
    vert = edge.split(" ")
    if vert[0] not in vertices:
        vertices.append(int(vert[0]))
    if vert[1] not in vertices:
        vertices.append(int(vert[1]))

#print "Done creating edges"

edges = map(lambda x:(int(x.split(" ")[0]),int(x.split(" ")[1])),edges)

G = nx.Graph()
#nodes = list(range(1,19))
G.add_nodes_from(vertices)
#G.add_edges_from([(8,10),(9,10),(11,13),(12,13),(13,3),(10,2),(10,3),(2,1),(2,7),(2,4),(2,3),(3,1),(3,7),(3,5),(3,4),
 #                 (7,1),(7,6),(7,5),(7,4),(4,5),(1,6),(1,5),(5,6),(5,16),(6,17),(17,18),(16,15),(16,14),(15,14)])

G.add_edges_from(edges)

A = nx.adjacency_matrix(G)
adj_matrix = A.todense()

M = np.zeros(adj_matrix.shape)

row, col = adj_matrix.shape

#Building similarity function matrix, ie, Cosine Function matrix of all Column Vectors

for x in xrange(0,row):
    for y in xrange(x,col):
        M[x][y] = round((1 - distance.cosine(adj_matrix[:,x], adj_matrix[:,y])),2)        

tuples = []    
#Considering only non zero values
for (x,y), value in np.ndenumerate(M):
    if value!=0 and x!=y:
        tuples.append(((x+1,y+1),value))

C = sorted(tuples, key=lambda x: x[1])
#print "done"
t = np.count_nonzero(adj_matrix)
#print(t)
C = C[-t:]

#print(C)
#print 'C done'

ln = len(C)
ln = ln-1

"""
* Building basic data structure to enable efficient implementation of algorithm

* Building a Tree from the largest cosine to smallest values

* Larger values imply lesser similarity, thus using a bottom up approach and building from low to high connected nodes
"""

nodes =dict()
tree = Tree()

for i in range(ln, -1, -1):
    vertices, value = C[i]
    i,j = vertices
    if nodes.__contains__(i) is False:
        a = Node(i)
        MakeSet(a)
        nodes[i] = a
    if nodes.__contains__(j) is False:
        a = Node(j)
        MakeSet(a)
        nodes[j]=a
    
    i = nodes[i]
    j = nodes[j]
    ri = SetFind(i)
    rj = SetFind(j)
    if ri.name != rj.name:
        tree.root = SetUnion(ri,rj)
        
"""
* Below Flow as Algorithm Specified in paper: 'Dense Subgraph Extraction with Application to  Community Detection'
"""
#Counting number of vertices and Edges
tree.count_vertices_and_edges(G.edges(),nodes)
#Summing up number of edges of children to parent
tree.count_vertices_and_edges_wrap(tree.root)
#Computing density of Tree Nodes
tree.compute_density(tree.root)
#Filtering Nodes as Per Density Threshold
tree.extract_sub_graph(tree.root,min_threshold)