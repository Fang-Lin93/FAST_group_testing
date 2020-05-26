import numpy as np
from utils import evaluate, flatten


class Algorithm(object):
    def __init__(self, **kwargs):
        self.flat_x = None
        self.PPT = None
        self.accuracy = None
        self.costs = None
        self.fnr = kwargs.get('fnr')
        self.fpr = kwargs.get('fpr')
        self.sensitivity = 1 - self.fnr
        self.specificity = 1 - self.fpr
        self.dilution_factor = kwargs.get('dilution_factor')

    def __call__(self, *args, **kwargs):
        """
        :param x: input list of states
        :param kwargs: paras of a single test
        :return:
        """
        self.costs = 0
        self.x = args[0]
        self.flat_x = flatten(self.x)  # record the true state
        self.N = len(self.flat_x)
        self.result = self.strategy(*args)
        if self.costs is not 0:
            self.PPT = self.N / self.costs
        if isinstance(self.result, list):
            self.accuracy = evaluate(self.flat_x, self.result)
        return self.result, self.PPT

    def strategy(self, *inputs):
        """
        Should be overridden by all subclasses.
        :param inputs: suggest inputting  states + pool size
        :return: tested results, same structure as inputs
        """
        return NotImplementedError

    def pcr(self, x, r=1, prop=1.0, op=False):
        """
        Example:
        x = generate_patient(rho=0.01, size=100000)
        y = list(map(lambda v: PCR(v, r = 1,one_active=True), x))
        evaluate(x, y)
        :param op: once-positive rule
        :param x: 0 is healthy, otherwise it's sick
        :param r: number of replicates, use majority rule
        :param prop: dilution factor, proportion of infected person
        :return: 1 or 0
        """

        if x == 0:
            return int(
                sum([np.random.uniform() > self.specificity for _ in range(r)]) > (0 if op else int(r / 2)))
        else:
            group_sensitivity = 1 - np.power((1 - self.sensitivity) / self.specificity,
                                             np.power(prop, self.dilution_factor)) * self.specificity
            return int(sum([np.random.uniform() <= group_sensitivity for _ in range(r)]) > (0 if op else int(r / 2)))

    def layer(self, local_x, r, op, positive=1):
        """
        it's an iterative algorithm, the leaves given positive are always tested one-by-one, like in reality;
        pooling structures are preserved, only testing results are added and sub-structure delivered

        :param op: 0 use majority rule, 1 use one active rule
        :param local_x: list of current pool, elements can be sub-pools or indivuals
        :param r: a list of number of replicates for the following sets
        :param positive: testing results of current pool, usually got by former testing
        :return: if its elements are sub-pools, return sub-pools testing results and deliver sub-pools to the sub-test;
                 if they are individuals,
                 return testing results for each person if necessary(current pool is/isn't positive)
        """
        replicates = r[0]
        one_rule = op[0]
        if isinstance(local_x[0], list):
            pool_ts = [0] * len(local_x)
            if positive:
                flat = list(map(lambda d: flatten(d), local_x))
                pool_ts = [self.pcr(sum(i), r=replicates,
                                    op=one_rule, prop=sum(i) / len(i)) for i in
                           flat]
                self.costs += len(local_x) * replicates
            return [self.layer(i, r[1:], op[1:], positive=j) for i, j in zip(local_x, pool_ts)]
        else:
            if positive:
                self.costs += len(local_x) * replicates
            return [self.pcr(i, r=replicates, op=one_rule) if positive != 0 else 0 for i in local_x]
