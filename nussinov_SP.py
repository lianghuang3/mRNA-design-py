#!/usr/bin/env python3

''' Nussinov (instead of CKY) for mRNA design
    Following pseudocode (Fig. S3) of the LD paper.

'''
__author__ = "Liang Huang (liang.huang.sh@gmail.com)"

import network
import sys
from collections import defaultdict

is_partition = True
match = {'CG': 3, 'GC': 3, 'AU': 2, 'UA': 2, 'GU': 1, 'UG': 1}
sharpturn = 3
INF = 1e32
lr = 0.5

import numpy as np
import dynet as dy

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


def fraction_nussinov(graph): # prob_graph

    def update(type, q_i, q_j, value):
        best[type, q_i, q_j] = dy.logsumexp([best[type, q_i, q_j], value])

    best = defaultdict(lambda : dy.scalarInput(-INF)) 

    protein = graph.aa
    print(protein)
    m = len(protein) # protein lengthh
    n = m * 3 # mRNA length
    
    dummy = dy.scalarInput(0.)
    graph.parameters[0] += dummy 
    #dy.renew_cg()

    for i in range(n):  # between-nuc indices
        for i_node in graph.nodes[i]:
            best["S", i_node, i_node] = dy.scalarInput(0)
            for iplus1_node, nuc, prob in graph.right_edges[i_node]:
                update("S", i_node, iplus1_node, 0 + dy.log(prob)) # two choices => e^0 + e^0

    for span in range(2, n+1):
        for i in range(n-span+1):
            j = i + span
            for i_node in graph.nodes[i]:
                for j_node in graph.nodes[j]:
                    for jminus1_node, j_nuc, j_prob in graph.left_edges[j_node]:
                        # S -> S N
                        update("S", i_node, j_node, 
                                best["S", i_node, jminus1_node] + dy.log(j_prob))

                        # P -> ( S )
                        if j - i > sharpturn+1:
                            for iplus1_node, i_nuc, i_prob in graph.right_edges[i_node]:
                                if i_nuc + j_nuc in match:
                                    update("P", i_node, j_node,
                                           best["S", iplus1_node, jminus1_node] + match[i_nuc + j_nuc] + dy.log(i_prob) + dy.log(j_prob))
                            
                    # S -> S P
                    for k in range(i, j): # i..j-1; left S could be epsilon
                        for k_node in graph.nodes[k]:
                            update("S", i_node, j_node,
                                   best["S", i_node, k_node] + best["P", k_node, j_node])                       
             
    obj = best["S", (0,0), (n,0)]                           
    print("CG built")
    sys.stdout.flush()

    last_v = -INF
    last_seq = None
    for it in range(1000):
        v = obj.value()
        seq = get_solution(graph)
        realv = inside_forward_log(seq) # more accurate
        viterbiv = inside_viterbi(seq)
        print("iteration %d\tobj value %.5f\tseq value %.5f seq mfe %d NEW %d ================" % (it, 
            v, realv, viterbiv, seq != last_seq))
        print(seq)
        last_seq = seq
        if np.fabs(v - last_v) < 1e-5 and np.fabs(v - realv) < 1e-5: # converged to integral solution
            break
        last_v = v

        obj.backward()
        for i, para in enumerate(graph.parameters):
            aa = protein[i]
            if i > 0 and len(codon_table[aa]) > 1:
                print(i, aa, 
                      "; ".join("%s:%.3f, grad:%.3f" % (c,v,g) for (c,v,g) in zip(codon_table[aa], para.value(), para.gradient()) \
                         if v > -0.01))
                new_para = para.value() + lr * para.gradient()
                #print(para)
                para.set_value(projection_simplex_np(new_para))
    #    print(bestS)
        #dy.renew_cg()
        dummy.set(0.)
        #print(obj.value())
    print("%.5f\t%s"  % (realv, get_solution(graph)))
    
    # param = graph.parameters[7].value()

    # param[0] = 0
    # param[-1] = 1
    
    # graph.parameters[7].set_value(param)

    # param = graph.parameters[4].value()

    # param[0] += 0.1
    # #param[-1] = 0
    
    # graph.parameters[4].set_value(param)
    # dummy.set(0.)
    # print(obj.value())  


    # param[0] -= 0.1
    # param[-1] += 0.1
    
    # graph.parameters[4].set_value(param)
    # dummy.set(0.)
    # print(obj.value())  
    # # param[0] += 1
    # # #param[-1] += 1
    # # graph.parameters[4].set_value(param)
    # # dummy.set(0.)
    # # print(obj.value())    

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
    aa_graphs, codon_table = network.read_wheel("coding_wheel.txt")
    old_aa_graphs, codon_table = network.read_wheel("coding_wheel.txt", old=True)

    for line in sys.stdin:
        protein = line.split()
        graph = network.Lattice()

        for aa in protein:
            graph += aa_graphs[aa]

        #graph.pp()
        #nussinov(graph)
        mfe, (seq, struct) = nussinov.nussinov(graph)
        v = inside_forward_log(seq)
        print("iteration -1\tobj value %.5f\tseq value %.5f seq mfe %d ====" % (v, v, mfe))
        print(seq)

        prob_graph = network.Prob_Lattice(graph)
        prob_graph.pp()
        sys.stdout.flush()
        fraction_nussinov(prob_graph)
