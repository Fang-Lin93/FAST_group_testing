import numpy as np


class SampleGenerator(object):
    """
    methods for generate simulation data
    """

    def __init__(self, **paras):
        self.size = paras.get('size')
        self.rho = paras.get('rho')
        self.sub_pool = paras.get('household')
        self.beta = paras.get('beta')
        self.mean_pool = sum(np.array(self.sub_pool[0]) * np.array(self.sub_pool[1]))
        self.rho_star = self.mean_pool / (1 + (self.mean_pool - 1) * self.beta) * self.rho

    def generate_fam_size(self):
        num = self.sub_pool[0]
        prob = self.sub_pool[1]
        return np.random.choice(num, p=prob)

    def generate_unit(self, rho=None):
        """
        Here I fixed the fraction of illness, generate fixed number of infections
        :param rho: prevalence
        :return: list with fraction of 1 is rho, others are 0
        """
        if rho is None:
            rho = self.rho
        res = np.zeros(self.size)
        res[np.random.choice(self.size, int(rho * self.size))] = 1
        return list(res)

    def generate_family(self, state):
        """
        randomly generates patients in a family
        :param state: infected=1
        :return:
        """
        size = self.generate_fam_size()
        if state == 0:
            return [0] * size
        else:
            return [1] + [int(np.random.uniform() < self.beta) for _ in range(size - 1)]

    def generate_fam_sample(self):
        """
        units: the states of families
        :return: expand each family to family members
        """
        units = self.generate_unit(self.rho_star)
        return list(map(self.generate_family, units))
