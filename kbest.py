#!/usr/bin/env python3

''' modified CKY for mRNA design'''
__author__ = "Liang Huang (liang.huang.sh@gmail.com)"

import network
import sys
from collections import defaultdict
from heapq import heapify, heappop, heappush
from random import random

match = {'CG': 3, 'GC': 3, 'AU': 2, 'UA': 2, 'GU': 1, 'UG': 1}

class cell(list): # list of (value, backpointer)'s, with (best, back) pointing to the best pair
    
    def __init__(self):
        self.best = -float("inf")
        self.kbest = []

    def append(self, item):
        super().append(item)
        value, backpointer = item
        if value > self.best:
            self.best = value
            self.back = backpointer

def cky(protein):

    def update(i, i_node, j, j_node, value, backpointer): # back is either k or (i_nuc, j_nuc)
        best[i, i_node, j, j_node].append((value, backpointer))
        #if value > best[i, i_node, j, j_node].best:
        #    best[i, i_node, j, j_node] = value
        #    back[i, i_node, j, j_node] = backpointer

    def backtrace(i, i_node, j, j_node):
        if i == j:
            return "", ""
        backpointer = best[i, i_node, j, j_node].back

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

    def kbest(i, i_node, j, j_node, k): # get kth-best (0-indexed) from state (i, i_node, j, j_node)

        def make_cand(value, backpointer, s, indices, order=None):
            return (-value, s, order if order is not None else random(), backpointer, indices) # bad tie-breaking
            
        cell = best[i, i_node, j, j_node]
        if k < len(cell.kbest):
            return cell.kbest[k]
        if k == 0 or not hasattr(cell, "cands"): # first call to this state, build cands first
            cell.cands = [make_cand(v, b, backtrace(i, i_node, j, j_node), (0,0), order) for order, (v, b) in enumerate(cell)]
            print(cell.best, cell.back)            
            print(cell.cands)
            heapify(cell.cands)
        if cell.cands != []:
            value, (seq, struct), _, backpointer, indices = heappop(cell.cands)
            print(i, i_node, j, j_node, k, value, (seq, struct), backpointer, indices)
            value = -value
            cell.kbest.append((value, (seq, struct)))
            if type(backpointer) is tuple:
                if len(backpointer) == 4: # (...)
                    i_nuc, iplus1_node, j_nuc, jminus1_node = backpointer
                    k = indices[0] + 1 # next best
                    subv, (subseq, substruct) = kbest(i+1, iplus1_node, j-1, jminus1_node, k)
                    if subv is not None:
                        heappush(cell.cands, make_cand(subv + 1, backpointer, 
                                                       ("%s%s%s" % (i_nuc, subseq, j_nuc), "(%s)" % substruct), (k, 0)))
                else: # split
                    sp, sp_node = backpointer
                    k1, k2 = indices
                    l_v, (l_seq, l_struct) = kbest(i, i_node, sp, sp_node, k1) # TODO: improve
                    r_v, (r_seq, r_struct) = kbest(sp, sp_node, j, j_node, k2)
                    newl_v, (newl_seq, newl_struct) = kbest(i, i_node, sp, sp_node, k1+1)
                    if newl_v is not None: # TODO : check duplicate (k1,k2)
                        heappush(cell.cands, make_cand(newl_v + r_v, backpointer, 
                                                       (newl_seq + r_seq, newl_struct + r_struct), (k1+1, k2)))
                    newr_v, (newr_seq, newr_struct) = kbest(sp, sp_node, j, j_node, k2+1)
                    if newr_v is not None:
                        heappush(cell.cands, make_cand(l_v + newr_v, backpointer, 
                                                       (l_seq + newr_seq, l_struct + newr_struct), (k1, k2+1)))
                    
            return value, (seq, struct)
        else:
            return None, (None, None) # no more kbest
        
    best = defaultdict(cell) # so that 0 might be answer
    print(protein)
    m = len(protein) # protein lengthh
    n = m * 3 # mRNA length
    for i in range(n):  # between-nuc indices
        i3 = i % 3
        iplus13 = (i+1) % 3
        i_graph = aa_graphs[protein[i // 3]]
        for i_node in i_graph.nodes[i3]:
            best[i, i_node, i, i_node].append((0, None))
            for iplus1_node, nuc in i_graph.right_edges[i_node]:
                best[i, i_node, i+1, iplus1_node].append((0, nuc))
        
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
                                           best[i+1, iplus1_node, j-1, jminus1_node].best + match[i_nuc + j_nuc],
                                           (i_nuc, iplus1_node, j_nuc, jminus1_node))
                    # case 2: X|Y
                    for k in range(i+1, j): # i+1...j-1
                        k3 = k % 3
                        k_graph = aa_graphs[protein[k // 3]] # it's fine for k_left
                        for k_node in k_graph.nodes[k3]:
                            update(i, i_node, j, j_node, 
                                   best[i, i_node, k, k_node].best + best[k, k_node, j, j_node].best, 
                                   (k, k_node))
            
    print(best[0, (0,0), n, (0,0)].best)
    print("\n".join(backtrace(0, (0,0), n, (0,0))))
    
    print(kbest(0, (0,0), n, (0,0), 0))
    print(kbest(0, (0,0), n, (0,0), 1))
#    print(kbest(0, (0,0), n, (0,0), 2))
#    print(kbest(0, (0,0), n, (0,0), 3))

    
if __name__ == "__main__":
    aa_graphs = network.read_wheel("coding_wheel.txt")
    for line in sys.stdin:
        protein = line.split()
        cky(protein)
        
