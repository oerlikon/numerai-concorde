#!/usr/bin/env python2 -B

import random
import sys

from math import log, log1p, sqrt

import numpy as np
import pandas as pd

from sklearn.cluster import MiniBatchKMeans


def noize(p):
    if p == 0.5:
        return 0
    if p < 0.5:
        if p < 0.25:
            q = p
        else:
            q = 0.5 - p
    else:
        if p < 0.75:
            q = p - 0.5
        else:
            q = 1 - p
    z = random.normalvariate(0, 0.005 * sqrt(q))
    asterisk = ''
    if (p > 0.5 and p + z < 0.5) or (p < 0.5 and p + z > 0.5):
        asterisk = ' *'
    if abs(z) > 0.01 or (asterisk and abs(z) > 0.0001):
        print >>sys.stderr, '# %+.7f %+.7f %+.7f %+.7f%s' % (p, q, z, abs(z / max(p-0.5, 0.5-p)), asterisk)
    return z


def pmerge(row):
    for q in row[1:]:
        if q != 0.5:
            return round(q + noize(q), 5)
    return 0.5


# Main function.

def main(argv):
    if len(argv) < 3:
        usage = 'Usage: %s <tournament_data.csv> <predictions.csv> [predictions.csv...]' % argv[0]
        print >>sys.stderr, usage
        return 1

    rng = random.Random(tuple(argv))

    try:
        tournament = pd.read_csv(argv[1])
        predictions = pd.read_csv(argv[2])
        for n, arg in enumerate(argv[3:]):
            pp = pd.read_csv(arg)
            predictions = predictions.join(pp.probability, rsuffix='%d' % (2 + n))
    except IOError as err:
        print >>sys.stderr, err
        return 2

    predictions['probability'] = predictions.apply(pmerge, axis=1)

    X = tournament.loc[:, 'feature1':'feature50']
    kmeans = MiniBatchKMeans(n_clusters=108, random_state=rng.getrandbits(32))
    kmeans.fit(X)
    tournament['C'] = pd.Series(kmeans.predict(X))

    nt = len(tournament)
    tournament = tournament.merge(predictions, how='outer', on='id')
    if len(tournament) != nt:
        print >>sys.stderr, 'wtf?', len(tournament), nt
        return 3

    concorde = pd.DataFrame(tournament.loc[:-1, ['id', 'probability', 'target']])

    for k, v in tournament.groupby('C', sort=True):
        gp = v.loc[np.isnan(v.probability) == False]

        c = pd.DataFrame(gp.loc[:, ['id', 'target']])
        pp = [ round(p + noize(p), 5) for p in gp.probability ]
        c['probability'] = pp
        concorde = concorde.append(c)

        lp = gp.probability.tolist()
        if len(lp) == 0:
            lp = [0.5]
        n = len(lp) - 1

        gx = v.loc[np.isnan(v.probability) & np.isnan(v.target)]

        c = pd.DataFrame(gx.loc[:, ['id', 'target']])
        pp = [ lp[rng.randint(0, n)] for _ in xrange(c.shape[0]) ]
        pp = [ round(p + noize(p), 5) for p in pp ]
        c['probability'] = pp
        concorde = concorde.append(c)

        gtx = v.loc[np.isnan(v.probability) & (np.isnan(v.target) == False)]
        c = pd.DataFrame(gtx.loc[:, ['id', 'target']])
        c.sort_values('target', inplace=True)
        pp = [ lp[rng.randint(0, n)] for _ in xrange(c.shape[0]) ]
        pp = [ round(p + noize(p), 5) for p in pp ]
        pp.sort()

        m = len(pp) - 1
        for i in xrange(m):
            j = rng.randint(i+1, m)
            p, q = pp[i], pp[j]
            if rng.random() < p*(1-q) / (p*(1-q) + q*(1-p)):
                pp[i], pp[j] = pp[j], pp[i]

        c['probability'] = pp
        concorde = concorde.append(c)

    concorde.sort_index(inplace=True)
    concorde.to_csv(sys.stdout, columns=['id', 'probability'], index=False)

    predictions = predictions.merge(concorde, how='left', on='id', suffixes=['', '_noize'])

    e, r = RunningStats(), RunningStats()
    for i, p in predictions.iterrows():
        e.push(logloss(p.probability, p.probability))
        if not np.isnan(p.target):
            r.push(logloss(p.probability, p.target))

    print >>sys.stderr, 'E %.9f %.9f R' % (e.mean(), r.mean())


# Running stats.

class RunningStats(object):
    def __init__(self, series=None):
        self.n, self.m1, self.m2 = 0, 0, 0
        if series is not None:
            for x in series:
                self.push(x)

    def push(self, x):
        self.n += 1
        dx = x - self.m1
        self.m1 += dx / self.n
        self.m2 += dx * (x - self.m1)

    def size(self):
        return self.n

    def mean(self):
        return self.m1

    def var(self):
        if self.n == 0:
            return 0
        return self.m2 / self.n


# Logloss function.

def logloss(p, t):
    return -t*log(p) - (1-t)*log1p(-p)


# Entry point.

if __name__ == '__main__':
    ret = main(sys.argv)
    exit(ret)
