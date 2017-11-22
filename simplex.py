#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from fractions import Fraction
import logging
import random


Frac = np.vectorize(Fraction)


def align_cols(tbl, min=0):
    """Pad a 2D array of strings such that each column has the same width."""
    
    ls = [max(min, max(map(len, c))) for c in zip(*tbl)]
    return [['{:>{}}'.format(*b) for b in zip(r, ls)] for r in tbl]


BOUNDED = 0
UNBOUNDED = 1
INFEASIBLE = 2


class LinProg:
    """Representation of a linear program in canonical form."""
    
    def __init__(self, a, b, c):
        n = len(c)  # variables
        m = len(b)  # constraints

        if len(a) != m or any(len(x) != n for x in a):
            raise ValueError('wrong constraint matrix shape')

        self.n = n
        self.m = m
        self.a = Frac(np.array(a))
        self.b = Frac(np.array(b))
        self.c = Frac(np.array(c))

    def __str__(self):
        def fmt(i, x):
            if x == 0:
                return ''
            elif x == 1:
                return '+x_{}'.format(i)
            elif x == -1:
                return '-x_{}'.format(i)
            else:
                return '{}{}x_{}'.format('' if x < 0 else '+', x, i)
        #fmt = lambda i, x: '{}x_{}'.format(x, i)
        tbl = align_cols([[fmt(*x) for x in enumerate(self.c)],
               *([fmt(*x) for x in enumerate(r)] for r in self.a)])
        return '\n'.join((
            'maximize:',
            '  {}'.format(' '.join(tbl[0])),
            'subject to:',
            *('  {} <= {}'.format(' '.join(r), self.b[i]) for (i, r) in enumerate(tbl[1:])),
            '  %s >= 0' % ', '.join('x_%s' % i for i in range(self.n+self.m))))

    @classmethod
    def parse(cls, stream):
        """Parse a linear program in canonical form from an io-stream."""
        
        n = int(stream.readline().strip())
        m = int(stream.readline().strip())
        c = stream.readline().split()
        b = stream.readline().split()
        a = [stream.readline().split() for _ in range(m)]
        return cls(a, b, c)

    def solve(self, rule='random'):
        """Solve the LP using phase I/II simplex alorithm."""

        for l in str(self).splitlines():
            logging.info(l)

        logging.info('number of variables: %s', self.n)
        logging.info('number of constraints: %s', self.m)

        steps = 0
        
        bad, = np.where(self.b < 0)
        n_bad = len(bad)

        if n_bad > 0:
            logging.info('PHASE 1 ({} bad constraints)'.format(n_bad))

            tb = Tableau(
                Frac(np.block([[self.a, np.eye(self.m, self.m+n_bad),
                                self.b.reshape(self.m, 1)],
                               [np.zeros(self.n+self.m), -np.ones(n_bad), 0]])),
                self.n+np.arange(self.m),
                rule=rule)

            tb.tb[bad,range(self.n+self.m, self.n+self.m+n_bad)] = Frac(-1)

            for i, j in enumerate(bad):
                tb.pivot(self.n+self.m+i, j)
            steps += len(bad)

            st, (tag, res) = tb.solve()
            assert tag != UNBOUNDED, 'phase 1 cannot be unbounded'
            steps += st

            if tag == INFEASIBLE or res[0] > 0:
                return steps, (INFEASIBLE, None)

            # artificial variables that are still basic
            degen, = np.where(tb.bs >= self.n + self.m)
            # non-basic variables that are not articifial
            nb = {i for i in range(self.n + self.m) if i not in tb.bs}
            for i in degen:
                for j in nb:
                    if tb.tb[i,j] != 0:
                        tb.pivot(j, i)
                        nb.remove(j)
                        break

            # patch the tableau
            tb.tb = np.delete(tb.tb, self.n+self.m+np.arange(n_bad), 1)
            tb.tb[-1,:self.n] = Frac(self.c)
            tb.tb[-1,self.n:] = Frac(0)
            tb.n -= n_bad

            for (i, x) in enumerate(tb.bs):
                tb.tb[-1] -= tb.tb[i]*tb.tb[-1,x]

        else:
            tb = Tableau(
                Frac(np.block([[self.a, np.identity(self.m),
                                self.b.reshape(self.m, 1)],
                               [self.c, np.zeros(self.m+1)]])),
                np.arange(self.n, self.n+self.m),
                rule=rule)

        logging.info('PHASE 2 (initial base: {})'.format(tb.bs))

        st, (tag, res) = tb.solve()
        steps += st

        if tag == BOUNDED:
            return steps, (BOUNDED, (res[0], res[1][:self.n]))
        else:
            return steps, (tag, None)

class Tableau:
    """Datastructure used to compute the optimal value of an LP from a
    feasible solution.

    :param tb: a numpy 2D array containing the built tableau (note that the
       objective row is the last row because it simplifies some indexing).
    :param bs: a numpy 1D array representing the basis, it's i-th element
       is the basic variable associated to the i-th constraint.
    """
    
    def __init__(self, tb, bs, rule='random'):

        self.m, self.n = tb.shape
        if len(bs) != self.m-1:
            raise ValueError('wrong basis length')

        self.tb = tb
        self.bs = bs

        try:
            self.rule = getattr(self, rule)
        except AttributeError:
            raise ValueError('unknown rule %s' % rule)

    def leaving_candidates(self, j):
        """Compute the rows candidates to be leaving, ie the minimal values
        according to the ratio test."""
        
        idx, = np.where(self.tb[:-1,j] > 0)
        if len(idx) == 0:
            return []
        ratio = self.tb[idx,-1] / self.tb[idx,j]
        cands = idx[ratio == ratio.min()]
        return cands

    def bland(self, e_cands):
        """Select the pivot by the Bland rule."""
        
        entering = e_cands[0]
        l_cands = self.leaving_candidates(entering)
        return (entering, min(l_cands, key=self.bs.__getitem__))

    def random(self, e_cands):
        """Select the pivot uniformly amongst the candidates."""
        
        entering = random.choice(e_cands)
        return (entering, random.choice(self.leaving_candidates(entering)))

    def max_coef(self, e_cands):
        """Select the pivot which has the largest coefficient in the objective."""

        entering = max(e_cands, key=lambda e: self.tb[-1,e])
        return (entering, random.choice(self.leaving_candidates(entering)))

    def greedy(self, e_cands):
        """Select the pivot which maximizes the objective gain."""

        l_cands = {e: self.leaving_candidates(e) for e in e_cands}
        ratios = {e: self.tb[l_cands[e][0],-1] / self.tb[l_cands[e][0],e]
                 for e in e_cands}
        entering = max(e_cands, key=lambda e: self.tb[-1,e]*ratios[e])
        return (entering, random.choice(l_cands[entering]))

    def pivot(self, j, i):
        """Pivot with entering variable `j` and leaving variable `self.bs[i]`."""
        
        logging.debug('pivoting with entering variable: x_%s and leaving '
                     'variable: x_%s', j, self.bs[i])
        self.bs[i] = j
        self.tb[i,:] /= self.tb[i,j]
        mask = np.full(self.m, True)
        mask[i] = False
        self.tb[mask,:] -= np.outer(self.tb[mask,j], self.tb[i,:])

    def solve(self):
        """Find the optimal solution by repeated pivoting."""
        
        steps = 0
        while True:
            logging.debug('current tableau:')
            for l in str(self).splitlines():
                logging.debug(l)
            logging.debug('current basis: %s', ', '.join(map('x_{}'.format, self.bs)))

            e_cands, = np.where(self.tb[-1,:-1] > 0)

            # optimality test
            if len(e_cands) == 0:
                # create a mapping {var_idx => value} (as list)
                attr = [0]*(self.n - 1)
                for (i, v) in enumerate(self.bs):
                    attr[v] = self.tb[i,-1]
                return steps, (BOUNDED, (-self.tb[-1,-1], attr, self.bs))

            # infinite value test
            if (self.tb[:-1,self.tb[-1,:] > 0] <= 0).all(axis=0).any():
                return steps, (UNBOUNDED, None)

            self.pivot(*self.rule(e_cands))
            steps += 1
            

    def __str__(self):
        aligned = align_cols(np.vectorize(str)(self.tb), min=2)
        return '\n'.join(map('  '.join, aligned))


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description='LP Solver -- OA 17-18')
    parser.add_argument('-v', '--verbose', type=int, metavar='LEVEL',
                        help='verbosity level (0-2)', default=-1)
    parser.add_argument('-r', '--rule',
                        choices=('random','bland','greedy','max_coef'),
                        help='pivot rule (default: greedy)', default='greedy')
    parser.add_argument('file', help='input file or `-` for stdin')
    args = parser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s',
                        level=10*(3-min(2, max(-1, args.verbose))))

    if args.file == '-':
        stream = sys.stdin
    else:
        try:
            stream = open(args.file)
        except Exception as err:
            logging.error('could not open file {}: {}'.format(args.file, err))
            sys.exit(1)

    try:
        lp = LinProg.parse(stream)
    except Exception as err:
        logging.error('could not parse the file {}: {}'.format(args.file, err))
        sys.exit(1)

    steps, (tag, res) = lp.solve(rule=args.rule)

    logging.info('number of pivot steps (both phases): %s', steps)
    logging.info('rule used: %s', args.rule)

    if tag == BOUNDED:
        print('BOUNDED')
        print('OPTIMAL VALUE: ', res[0])
        print('SOLUTION: ', ', '.join('x_{} = {}'.format(*x)
                                      for x in enumerate(res[1])))
    elif tag == UNBOUNDED:
        print('UNBOUNDED')
    else:
        print('INFEASIBLE')

    if stream is not sys.stdin:
        stream.close()
