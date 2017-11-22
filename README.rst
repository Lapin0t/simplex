=====================
LP Solver -- OA 17-18
=====================

This is a solver for linear programming that uses the simplex algorithm
(phase i/ii).

Dependencies
------------

- python3
- numpy >= 1.13

Usage
-----

The main script is `simplex.py`. One can get the full help with::

   ./simplex.py --help

The options are:

`-v LEVEL`
   The verbosity level: by default (0) you only get the raw result, with 1 you
   get a few more infos and with 2 the full debug.
`-r RULE`
   The choice of rule, by default i choose `greedy` as it seems to be the most
   efficient and it shouldn't really loop. The possible options are `greedy`,
   `random`, `max_coef` and `bland`.

The only argument is the input file. If you pass `-` as input it will read from
standard input.

There is also a test generator `gentest.py` to which one can specify the size
parameters (n and m), the domain for random variables and the percentage of
zero values. I tried to implement generators that create feasible or infeasible
problems but they are mostly boguous. It might have been interesting to provide
a switch to make the domain not centered on zero (and different for A and b).


Possible optimizations
----------------------

I tried to run it on pypy+numpypy but there are a few quirks in numpypy
(around ufuncs, `np.vectorize` and also because the development somewhat
stopped i should factor out `np.block` which requires >=1.13) so it doesn't
really work. Anyway, numpy is used almost exclusively to facilitate indexing
and a few basic operations so to optimize it on pypy the best way would be to
rewrite the whole operations with pure python arrays.

Actually, the most valuable optimization would be to switch to the revised
simplex algorithm which doesn't really use a tableau explicitely and can
beneficiate from the sparsity of the involved arrays.
