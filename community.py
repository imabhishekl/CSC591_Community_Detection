import numpy as np
import networkx as nx
import sys
from scipy.spatial import distance

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

	def findLCA_Node(self,src_node,dest_node):
		while src_node is not None:
			if dest_node.name in src_node.vertices:
				return src_node
			src_node = src_node.parent
		return None

	def print_Tree(self,root):
		if root==None:
			return
		#print(root.vertices)
		self.print_Tree(root.left)
		self.print_Tree(root.right)

	def print_nodes(self,nodes):
		for i in range(0,len(nodes)):
			print(nodes[i].name,nodes[i].parent.name)

	def count_vertices_and_edges(self,edges_list,nodes_list):
		for edge in edges_list:
			src_node = filter(lambda x: x.name == edge[0], nodes_list)
			dst_node = filter(lambda x: x.name == edge[1], nodes_list)
			lca_node = None
			if len(src_node)!=0 and len(dst_node)!=0:
				lca_node = self.findLCA_Node( src_node[0], dst_node[0] )
			#lca_node = self.findLCA_Node(next((x for x in nodes_list if x.name == edge[0]),None),next((y for y in nodes_list if y.name == edge[1]),None))
			if lca_node is not None:
				lca_node.num_edges = lca_node.num_edges + 1
				#print "lca:" + str(lca_node.num_edges)

	def count_vertices_and_edges_wrap(self,root):
		if root.left != None and root.right != None:
			self.count_vertices_and_edges_wrap(root.left)
			self.count_vertices_and_edges_wrap(root.right)
		if root.left != None and root.right != None:
			root.num_edges = root.left.num_edges + root.right.num_edges + root.num_edges

	def compute_density(self,root):
		if root.left is None and root.right is None:
			return 
		total_vertices = float(len(root.vertices))
		max_vertices = total_vertices*(total_vertices - 1)/2
		root.density = root.num_edges/max_vertices

		self.compute_density(root.left)
		self.compute_density(root.right)

	def extract_sub_graph(self,root,min_density):
		if root is None:
			return
		if root.density > min_density:
			print "Community detect with following vertices:" + str(root.vertices)
		else:
			self.extract_sub_graph(root.left,min_density)
			self.extract_sub_graph(root.right,min_density)

def MakeSet(r):
	r.parent = None
	r.vertices.add(r.name)
	

def SetFind(r):
	while r.parent is not None:
		r = r.parent
	return r

def SetUnion(x,y):
    r = Node("P"+ str(x.name) + str(y.name))
    r.left = x
    r.right = y
    x.parent = r
    y.parent = r
    r.vertices = r.vertices.union(x.vertices,y.vertices)
        #r.vertices.extend(x.vertices)
        #r.vertices.extend(y.vertices)
    return r
        
if len(sys.argv) != 2:
	print "Please pass command line argument for file name"
	exit(0)

graph_file = open(sys.argv[1])

edges = graph_file.read().splitlines()

vertices = []
print "Creating Graph........."
for edge in edges:
    vert = edge.split(" ")
    if vert[0] not in vertices:
        vertices.append(int(vert[0]))
    if vert[1] not in vertices:
        vertices.append(int(vert[1]))

edges = map(lambda x:(int(x.split(" ")[0]),int(x.split(" ")[1])),edges)

G = nx.Graph()
#nodes = list(range(1,19))
G.add_nodes_from(vertices)
#G.add_edges_from([(8,10),(9,10),(11,13),(12,13),(13,3),(10,2),(10,3),(2,1),(2,7),(2,4),(2,3),(3,1),(3,7),(3,5),(3,4),
#                 (7,1),(7,6),(7,5),(7,4),(4,5),(1,6),(1,5),(5,6),(5,16),(6,17),(17,18),(16,15),(16,14),(15,14)])

G.add_edges_from(edges)
print "Done creating graph"
A = nx.adjacency_matrix(G)
adj_matrix = A.todense()

M = np.zeros(adj_matrix.shape)

print "Creating cosine data........"

row, col = adj_matrix.shape
for x in xrange(0,row):
    for y in xrange(x,col):
        M[x][y] = round((1 - distance.cosine(adj_matrix[:,x], adj_matrix[:,y])),2)        

tuples = []    
for (x,y), value in np.ndenumerate(M):
    if value!=0 and x!=y:
        tuples.append(((x+1,y+1),value))
   
C = sorted(tuples, key=lambda x: x[1])
t = np.count_nonzero(adj_matrix)
#print(t)
C = C[-t:]
print "Done creating cosine data"
#print(C)

ln = len(C)
ln = ln-1

nodes = []
tree = Tree()

print "Creating dendogram data........"

for i in range(ln, -1, -1):
    vertices, value = C[i]
    i,j = vertices
    if not any(x.name == i for x in nodes):
        a = Node(i)
        MakeSet(a)
        nodes.append(a)
    if not any(x.name == j for x in nodes):
        a = Node(j)
        MakeSet(a)
        nodes.append(a)
    
    i = next((x for x in nodes if x.name == i), None)
    j = next((x for x in nodes if x.name == j), None)
    ri = SetFind(i)
    rj = SetFind(j)
    if ri.name != rj.name:
        tree.root = SetUnion(ri,rj)
        
print "Done creating dendogram."
#tree.print_Tree(tree.root)
print "Start dendogram processing........"
tree.root.parent = None
tree.count_vertices_and_edges(G.edges(),nodes)
tree.count_vertices_and_edges_wrap(tree.root)
tree.compute_density(tree.root)
tree.extract_sub_graph(tree.root,0.75)
print "Done."