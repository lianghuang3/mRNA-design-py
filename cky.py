#!/usr/bin/env python3

''' modified CKY for mRNA design'''
__author__ = "Liang Huang (liang.huang.sh@gmail.com)"

import network
import sys
from collections import defaultdict

match = {'CG': 3, 'GC': 3, 'AU': 2, 'UA': 2, 'GU': 1, 'UG': 1}

def cky(protein):

    def update(i, i_node, j, j_node, value, backpointer): # back is either k or (i_nuc, j_nuc)
        if value > best[i, i_node, j, j_node]:
            best[i, i_node, j, j_node] = value
            back[i, i_node, j, j_node] = backpointer

    def backtrace(i, i_node, j, j_node):
        if i == j:
            return "", ""
        backpointer = back[i, i_node, j, j_node]

        if type(backpointer) is str: # singleton
            return backpointer, "."
        else: # tuple: two cases: (k, k_node) or (i_nuc, iplus1_node, j_nuc, jminus1_node)
            if len(backpointer) == 4: # (...)
                i_nuc, iplus1_node, j_nuc, jminus1_node = backpointer
                seq, struct = backtrace(i+1, iplus1_node, j-1, jminus1_node)
                return "%s%s%s" % (i_nuc, seq, j_nuc), "(%s)" % struct
            else: # split
                k, k_node = backpointer
                seq1, struct1 = backtrace(i, i_node, k, k_node)
                seq2, struct2 = backtrace(k, k_node, j, j_node)
                return seq1+seq2, struct1+struct2
        
    best = defaultdict(lambda : -1) # so that 0 might be answer
    back = {}
    print(protein)
    m = len(protein) # protein lengthh
    n = m * 3 # mRNA length
    for i in range(n):  # between-nuc indices
        i3 = i % 3
        iplus13 = (i+1) % 3
        i_graph = aa_graphs[protein[i // 3]]
        for i_node in i_graph.nodes[i3]:
            best[i, i_node, i, i_node] = 0
            for iplus1_node, nuc in i_graph.right_edges[i_node]:
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
                    # optimize best[i, i_node][j, j_node]
                    # case 1: (...)
                    if j-i > 4: # i(...)j   (i-right-nuc and j-left-nuc match)
                        for iplus1_node, i_nuc in i_graph.right_edges[i_node]:
                            for jminus1_node, j_nuc in j_graph.left_edges[j_node]:
                                if i_nuc + j_nuc in match:
                                    update(i, i_node, j, j_node,
                                           best[i+1, iplus1_node, j-1, jminus1_node] + match[i_nuc + j_nuc],
                                           (i_nuc, iplus1_node, j_nuc, jminus1_node))
                    # case 2: X|Y
                    for k in range(i+1, j): # i+1...j-1
                        k3 = k % 3
                        k_graph = aa_graphs[protein[k // 3]] # it's fine for k_left
                        for k_node in k_graph.nodes[k3]:
                            update(i, i_node, j, j_node, 
                                   best[i, i_node, k, k_node] + best[k, k_node, j, j_node], 
                                   (k, k_node))
            
    print(best[0, (0,0), n, (0,0)])
    print("\n".join(backtrace(0, (0,0), n, (0,0))))
    
if __name__ == "__main__":
    aa_graphs = network.read_wheel("coding_wheel.txt")
    for line in sys.stdin:
        protein = line.split()
        cky(protein)
        
