'''
Filename: utils.py
Purpose: This file contains helper functions for crowd indictation feature, and bounding box operations
Author: Angelique Taylor <amt062@eng.ucsd.edu;amt298@cornell.edu>
Note: If you use this in your work, please cite our HRI 2022 paper (see README.md for bibtex)
'''
from __future__ import division
import numpy as np 
from collections import namedtuple
import cv2
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict #DFS


def MergeBox(boxes): 
    """
    Return x <center row>, y <center col>, w <width> and h <height>
    """
    min_row = min(boxes[:,1] - boxes[:,3]/2)
    max_row = max(boxes[:,1] + boxes[:,3]/2)
    min_col = min(boxes[:,0] - boxes[:,2]/2)
    max_col = max(boxes[:,0] + boxes[:,2]/2)

    x = max_col - ((max_col - min_col)/2)
    y = min_row - ((min_row - max_row)/2)
    w = max_col - min_col
    h = max_row - min_row
    return x, y, w, h, min_row, max_row, min_col, max_col

def Compute_IOU(boxA, boxB):
    # Compute the IOU of two boxes
    #https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
	# determine the (x, y)-coordinates of the intersection rectangle

    b1a = boxA[0] - boxA[2]/2
    b2a = boxA[1] - boxA[3]/2
    b3a = boxA[0] + boxA[2]/2
    b4a = boxA[1] + boxA[3]/2

    b1b = boxB[0] - boxB[2]/2
    b2b = boxB[1] - boxB[3]/2
    b3b = boxB[0] + boxB[2]/2
    b4b = boxB[1] + boxB[3]/2

    xA = max(b1a, b1b)
    yA = max(b2a, b2b)
    xB = min(b3a, b3b)
    yB = min(b4a, b4b)
 
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (b3a - b1a + 1) * (b4a - b2a + 1)
    boxBArea = (b3b - b1b + 1) * (b4b - b2b + 1)
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou
    
vertices=[]

class Graph: 
    '''
    This class represents a undirected graph using adjacency list representation 
    '''
   
    def __init__(self,vertices): 
        self.V= vertices #No. of vertices 
        self.graph = defaultdict(list) # default dictionary to store graph 
        self.vertices=[]
   
    # function to add an edge to graph 
    def addEdge(self,v,w): 
        self.graph[v].append(w) #Add w to v_s list 
        self.graph[w].append(v) #Add v to w_s list 
   
    # A recursive function that uses visited[] and parent to detect 
    # cycle in subgraph reachable from vertex v. 
    def isCyclicUtil(self,v,visited,parent): 
  
        # Mark the current node as visited  
        visited[v]= 1

        self.vertices.append(v)

        #Recur for all the vertices adjacent to this vertex 
        for i in self.graph[v]: 
            # If the node is not visited then recurse on it 
            if  visited[i]==0 :  
                if(self.isCyclicUtil(i,visited,v)): 
                    return 1
            # If an adjacent vertex is visited and not parent of current vertex, 
            # then there is a cycle 
            elif  parent!=i: 
                return 1
          
        return 0
           
   
    #Returns true if the graph contains a cycle, else false. 
    def isCyclic(self,start): 
        # Mark all the vertices as not visited 
        visited =[0]*(self.V) 
        
        for i in range(start,len(visited)):#self.V): 
            if visited[i] ==0: #Don't recur for u if it is already visited 
                if(self.isCyclicUtil(i,visited,-1))== 1: 
                    return 1
          
        return 0
        
def GraphDriver(graph_):
	'''
	graph_=[[0,0,0,0,0,1,1,1],
        [0,0,1,0,1,0,0,0],
        [0,1,0,0,1,0,1,0],
        [0,0,0,0,0,0,0,0],
        [0,1,1,0,0,0,0,0],
        [1,0,0,0,0,0,0,1],
        [1,0,1,0,0,0,0,1],
        [1,0,0,0,0,1,1,0]]
	'''
	
	g = Graph(len(graph_))
	for i in range(0,len(graph_)):
		for j in range(0,len(graph_)):
		    if(graph_[i][j] == 1):

		        g.addEdge(i,j) 
		        g.addEdge(j,i)

	v=[]
	for i in range(0,len(graph_)):
		g.vertices=[]
		if(sum(graph_[i][:]) != 0):
		    if g.isCyclic(i): 
		        if(len(g.vertices) > len(v)):
		            v=g.vertices
	return np.unique(v)
