#!/usr/bin/env python3

''' Nussinov (instead of CKY) for mRNA design
    Following pseudocode (Fig. S3) of the LD paper, but using one nonterminal (slow).
    NEXT: USE S AND P nonterminals.

'''
__author__ = "Liang Huang (liang.huang.sh@gmail.com)"

import network
import sys
from collections import defaultdict

is_partition = False #True
match = {'CG': 3, 'GC': 3, 'AU': 2, 'UA': 2, 'GU': 1, 'UG': 1}
sharpturn = 3

import numpy as np

class Float(float): # log-space
    def __iadd__(self, other):
        self = Float(np.logaddexp(self, other)) # must return Float not float
        return self

float_class = Float # Float2 much slower

def nussinov(protein, aa_graphs):

    def update(i, i_node, j, j_node, value, backpointer): # back is either k or (i_nuc, j_nuc)
        if is_partition: # sum, logplus
            #print("update", i, i_node, j, j_node, best[i, i_node, j, j_node], value)
            best[i, i_node, j, j_node] += value # logplus
            #print("updated", i, i_node, j, j_node, "new", best[i, i_node, j, j_node])
        else: # max            
            if value > best[i, i_node, j, j_node]:
                best[i, i_node, j, j_node] = value
                back[i, i_node, j, j_node] = backpointer

    def backtrace(i, i_node, j, j_node):
        if i == j:
            return "", "" # (xxx) case: left side is empty
        backpointer = back[i, i_node, j, j_node]

        if type(backpointer) is str: # singleton
            return backpointer, "."
        else: # tuple: two cases: (k, k_node) or (i_nuc, iplus1_node, j_nuc, jminus1_node)
            if len(backpointer) == 2: # xxx .
                jminus1_node, j_nuc = backpointer
                seq, struct = backtrace(i, i_node, j-1, jminus1_node)
                return seq + j_nuc, struct + "."
            else: # xxx(xxx)
                k, k_node, k_nuc, kplus1_node, jminus1_node, j_nuc = backpointer                     
                seq1, struct1 = backtrace(i, i_node, k, k_node)
                seq2, struct2 = backtrace(k+1, kplus1_node, j-1, jminus1_node)
                return seq1 + k_nuc + seq2 + j_nuc, struct1 + "(" + struct2 + ")"            
        
    best = defaultdict(lambda : (float_class if is_partition else lambda x:x)(-np.inf)) # so that 0 might be answer
    back = {}
    #protein = graph.aa
    print(protein)
    m = len(protein) # protein lengthh
    n = m * 3 # mRNA length
    for i in range(n):  # between-nuc indices
        i3 = i % 3
        iplus13 = (i+1) % 3
        i_graph = aa_graphs[protein[i // 3]]
        for i_node in i_graph.nodes[i3]:
            best[i, i_node, i, i_node] = (float_class if is_partition else lambda x:x)(0)
            for iplus1_node, nuc in i_graph.right_edges[i_node]:
                if is_partition:
                    best[i, i_node, i+1, iplus1_node] += 0 # two choices => e^0 + e^0
                else:
                    best[i, i_node, i+1, iplus1_node] = 0
                    back[i, i_node, i+1, iplus1_node] = nuc # needed?
        
    for span in range(2, n+1):
        for i in range(n-span+1):
            j = i + span
            i3, j3 = i % 3, j % 3
            i_graph = aa_graphs[protein[i // 3]]
            j_graph = aa_graphs[protein[(j-1) // 3]]
            for i_node in i_graph.nodes[i3]:
                for j_node in j_graph.nodes[j3]:
                    for jminus1_node, j_nuc in j_graph.left_edges[j_node]:
                        # case 1: j unpaired: xxxx.
                        update(i, i_node, j, j_node,
                               best[i, i_node, j-1, jminus1_node],
                               (jminus1_node, j_nuc))
                            
                        # case 2: j paired with some k: xxx(xxx)
                        for k in range(i, j-sharpturn-1): # i+1...j-4
                            k3 = k % 3
                            k_graph = aa_graphs[protein[k // 3]] # it's fine for k_left
                            for k_node in k_graph.nodes[k3]:
                                for kplus1_node, k_nuc in k_graph.right_edges[k_node]:
                                    if k_nuc + j_nuc in match:
                                        #print("pair", k, j, k_nuc + j_nuc)
                                        update(i, i_node, j, j_node,
                                               best[i, i_node, k, k_node] + best[k+1, kplus1_node, j-1, jminus1_node] + match[k_nuc + j_nuc],
                                               (k, k_node, k_nuc, kplus1_node, jminus1_node, j_nuc))
                                        
    mfe = (best[0, (0,0), n, (0,0)])
    #print(best)
    if not is_partition:
        seq = ("\n".join(backtrace(0, (0,0), n, (0,0))))

    return mfe, seq
    
if __name__ == "__main__":
    aa_graphs, codon_table = network.read_wheel("coding_wheel.txt", old=True) # TODO: replace OLD with NEW
    for line in sys.stdin:
        protein = line.split()
        print(nussinov(protein, aa_graphs))
