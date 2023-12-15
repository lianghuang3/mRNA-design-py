#!/usr/bin/env python

import sys

def enum(seq, i=0):
    if i == len(seq):
        yield ""
    else:
        for codon in table[seq[i]]:
            for s in enum(seq, i+1):
                yield codon + s
        
if __name__ == "__main__":

    table = {}
    for line in open("coding_wheel.txt"):
        stuff = line.strip().split("\t") # Leu	U U AG	C U UCAG
        aa = stuff[0]
        codons = []
        for i, option in enumerate(stuff[1:]):
            first, second, thirds = option.split(" ")
            for third in thirds:
                codons.append(first + second + third)
        table[aa] = codons

    aa_seq = sys.stdin.readline().split()
    for s in enum(aa_seq):
        print(s)
    
