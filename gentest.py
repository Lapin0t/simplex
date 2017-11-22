#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import random
import numpy as np


def sample(n, r, d):
    return np.array([
        0 if random.random() >= d else random.randrange(*r)
        for _ in range(n)])

def output(n, m, a, b, c):
    print(n)
    print(m)
    print(' '.join(map(str, c)))
    print(' '.join(map(str, b)))
    for row in a:
        print(' '.join(map(str, row)))

def yolo(n, m, r, d):
    return (sample(n*m, (-r, r), d).reshape(m, n),
            sample(m, (-r, r), d),
            sample(n, (-r, r), d))

def feasible(n, m, r, d):
    a = sample(n*m, (-r, r), d).reshape(m, n)
    return (a,
            a @ sample(n, (-r, r), d) + sample(m, (0, r), d/2),
            sample(n, (-r, r), d))

def infeasible(n, m, r, d):
    a = sample(n*m, (-r, r), d).reshape(m, n)
    return (a,
            a @ sample(n, (-r, r), d) - sample(m, (0, r), d/2),
            sample(n, (-r, r), d))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='LP generator',
        epilog='THE COMMANDS ARE PROVIDED `AS IS` WITHOUT ANY WARRANTY, '
        'INCLUDING, BUT NOT LIMITED TO, THE LP BEING FEASIBLE WHEN BEING ASKED'
        ' TO!')
    parser.add_argument('-r', '--range', type=int, default=200,
        help='domain of the random numbers')
    parser.add_argument('-d', '--density', type=float, default=0.3,
        help='probability for a coefficient to be non-zero')
    parser.add_argument('type', choices=('yolo', 'feasible', 'infeasible'),
        help='it should do what it says', default='yolo', nargs='?')
    parser.add_argument('n', type=int, help='number of variables')
    parser.add_argument('m', type=int, help='number of constraints')
        
    args = parser.parse_args()

    n, m = args.n, args.m
    output(n, m, *globals()[args.type](n, m, args.range, args.density))
