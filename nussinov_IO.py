#!/usr/bin/env python3

''' Nussinov (instead of CKY) for mRNA design
    Following pseudocode (Fig. S3) of the LD paper, but using one nonterminal (slow).
    NEXT: USE S AND P nonterminals.

'''
__author__ = "Liang Huang (liang.huang.sh@gmail.com)"

import network_IO
import sys
from collections import defaultdict

is_partition = True
match = {'CG': 3, 'GC': 3, 'AU': 2, 'UA': 2, 'GU': 1, 'UG': 1}
sharpturn = 3
INF = 1e32
lr = 0.5

from time import time
import numpy as np

from inside_outside import inside_forward_log, inside_viterbi
import nussinov

class Float(float): # log-space
    def __iadd__(self, other):
        self = Float(np.logaddexp(self, other)) # must return Float not float
        return self

float_class = Float # Float2 much slower

def nussinov_partition(graph):

    bestS = defaultdict(lambda : float_class(-np.inf)) 
    bestP = defaultdict(lambda : float_class(-np.inf)) 

    protein = graph.aa

    print(protein)
    m = len(protein) # protein lengthh
    n = m * 3 # mRNA length
    for i in range(n):  # between-nuc indices
        for i_node in graph.nodes[i]:
            bestS[i_node, i_node] = float_class(0)
            for iplus1_node, nuc in graph.right_edges[i_node]:
                bestS[i_node, iplus1_node] += 0 # two choices => e^0 + e^0
            
    for span in range(2, n+1):
        for i in range(n-span+1):
            j = i + span
            for i_node in graph.nodes[i]:
                for j_node in graph.nodes[j]:
                    for jminus1_node, j_nuc in graph.left_edges[j_node]:
                        # S -> S N
                        bestS[i_node, j_node] += bestS[i_node, jminus1_node]

                        # P -> ( S )
                        if j - i > sharpturn+1:
                            for iplus1_node, i_nuc in graph.right_edges[i_node]:
                                if i_nuc + j_nuc in match:
                                    bestP[i_node, j_node] += bestS[iplus1_node, jminus1_node] + match[i_nuc + j_nuc]
                            
                    # S -> S P
                    for k in range(i, j): # i..j-1; left S could be epsilon
                        for k_node in graph.nodes[k]:
                            bestS[i_node, j_node] += bestS[i_node, k_node] + bestP[k_node, j_node]
                                        
    print(bestS[(0,0), (n,0)])
    #print(bestS)
    if not is_partition:
        print("\n".join(backtrace(0, (0,0), n, (0,0))))


class Node:

    __slots__ = "hyperedges", "alpha", "beta"

    def __init__(self, alpha=None):
        #self.signature = signature
        self.hyperedges = []
        self.alpha = alpha

    def add_hyperedge(self, children, value):
        self.hyperedges.append(HyperEdge(children, value))

    def value(self): # forward
        if self.alpha is not None: # cache
            return self.alpha
        s = Float()
        for edge in self.hyperedges:
            x = edge.edge_value + sum(sub.value() for sub in edge.children) # float
            s += x # log
        self.alpha = s
        return s

class HyperEdge:

    __slots__ = "children", "edge_value"

    def __init__(self, children:list, edge_value:float):
        self.children = children
        self.edge_value = edge_value        

def fraction_nussinov(graph): # prob_graph

    def get_node(signature, alpha=None):
        if signature not in nodes:
            nodes[signature] = Node(alpha)
        return nodes[signature]


    start = time()
    nodes = {}

    protein = graph.aa
    print(protein)
    m = len(protein) # protein lengthh
    n = m * 3 # mRNA length
    
    for i in range(n):  # between-nuc indices
        for i_node in graph.nodes[i]:
            get_node(("S", i_node, i_node), 0)
            for iplus1_node, nuc, prob in graph.right_edges[i_node]:
                get_node(("S", i_node, iplus1_node)).add_hyperedge([], 0 + np.log(prob)) # two choices => e^0 + e^0

    for span in range(2, n+1):
        for i in range(n-span+1):
            j = i + span
            for i_node in graph.nodes[i]:
                for j_node in graph.nodes[j]:
                    Snode = get_node(("S", i_node, j_node))                    
                    Pnode = get_node(("P", i_node, j_node))
                    
                    for jminus1_node, j_nuc, j_prob in graph.left_edges[j_node]:
                        # S -> S N
                        x = nodes.get(("S", i_node, jminus1_node), None)
                        if x is not None:
                            Snode.add_hyperedge([x], np.log(j_prob))

                        # P -> ( S )
                        if j - i > sharpturn+1:
                            for iplus1_node, i_nuc, i_prob in graph.right_edges[i_node]:
                                if i_nuc + j_nuc in match:
                                    x = nodes.get(("S", iplus1_node, jminus1_node), None)
                                    if x is not None:
                                        Pnode.add_hyperedge([x], match[i_nuc + j_nuc] + np.log(i_prob) + np.log(j_prob))
                            
                    # S -> S P
                    for k in range(i, j): # i..j-1; left S could be epsilon
                        for k_node in graph.nodes[k]:
                            x = nodes.get(("S", i_node, k_node), None)
                            y = nodes.get(("P", k_node, j_node), None)
                            if x is not None and y is not None:
                                Snode.add_hyperedge([x, y], 0)
             
    obj = nodes["S", (0,0), (n,0)]                           
    print("build CG: %.2f secs" % (time() - start))
    print("CG built")
    sys.stdout.flush()

    t = time()
    print(obj.value())
    print("forward: %.2f secs" % (time() - t))

def get_solution(graph):

    seq = ""
    for i, para in enumerate(graph.parameters):
        aa = graph.aa[i]
        seq += codon_table[aa][np.argmax(para.value())]
    return seq

def projection_simplex_np(v, z=1):
    v = np.array(v)
    n_features = v.shape[0]
    print(f'n_features: {n_features}')
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 1e-32) # lhuang: epsilon value
    return w
    
if __name__ == "__main__":
    aa_graphs, codon_table = network_IO.read_wheel("coding_wheel.txt")
    
    for line in sys.stdin:
        protein = line.split()
        t = time()
        graph = network_IO.Lattice()

        for aa in protein:
            graph += aa_graphs[aa]
        
        #graph.pp()
        #nussinov(graph)
        mfe, (seq, struct) = nussinov.nussinov(graph)
        v = inside_forward_log(seq)
        print("iteration -1\tobj value %.5f\tseq value %.5f seq mfe %d ====" % (v, v, mfe))
        print(seq)
        print("MFE time: %.2f secs" % (time() - t))

        print("start")
        t = time()
        prob_graph = network_IO.Prob_Lattice(graph)
        #prob_graph.pp()
        print("build lattice: %.2f secs" % (time() - t))
        sys.stdout.flush()
        fraction_nussinov(prob_graph)
