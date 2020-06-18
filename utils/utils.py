import numpy as np
from scipy.special import binom


def mean_ppt(k, rho, r=1):
    n_tests = (1 - np.power(1 - rho, k)) * k * r + r
    return k / n_tests

def two_ppt(k,m,rho,r=1):
    rho_star = 1-np.power(1 - rho, m)
    k_star = k/m
    n_tests = (1 - np.power(1 - rho_star, k_star)) * k_star * r + r + k_star*rho_star*m*r
    return k/n_tests


def theo_optk(rho):
    k = 1
    while mean_ppt(k,rho) < mean_ppt(k+1,rho):
        k += 1
    return k
# se= []
# tr = []
# rho = 0
# for r in np.linspace(0.00001,0.4,100000):
#       se += [int(round(1/np.sqrt(r)))]
#       tr += [theo_optk(r)]
#       if se[-1] == tr[-1] or (se[-1]+1)== tr[-1]:
#           rho+= 1
# print(rho)



def majority_rate(p, r=3):
    """
    :param p: Prob(happens) for single trial
    :param r: number of replicates
    :return: Prob(majority happens)
    """
    return sum([binom(r, i) * np.power(p, r - i) * np.power(1 - p, i) for i in range(int(r / 2) + 1)])


def pcr(x, sensitivity=0.98, specificity=0.999, r=1, prop=1.0, d=0, op=False):
    """
    Example:
    x = generate_patient(rho=0.01, size=100000)
    y = list(map(lambda v: PCR(v, r = 1,one_active=True), x))
    evaluate(x, y)
    :param x: 0 is healthy, otherwise it's sick
    :param sensitivity:
    :param specificity:
    :param r: number of replicates, use majority rule
    :param prop: dilution factor, proportion of infected person
    :param d: shrinkage factor>=0, the smaller, FNR is less sensitive to the dilution
    :param op: 1: use once-positive rule, 0: majority rule
    :return: 1 or 0
    """

    if x == 0:
        return int(sum([np.random.uniform() > specificity for _ in range(r)]) > (0 if op else int(r / 2)))
    else:
        group_sensi = 1 - np.power((1 - sensitivity) / specificity, np.power(prop, d)) * specificity
        return int(sum([np.random.uniform() <= group_sensi for _ in range(r)]) > (0 if op else int(r / 2)))



def sir(y, delta_t=1, beta=1 / 2, gamma=1 / 7):
    """
    Classical SIR model
    """
    res = y
    for t in range(delta_t):
        s = res[0]
        i = res[1]
        r = res[2]
        res[0] = -beta * s * i + s
        res[1] = beta * s * i - gamma * i + i
        res[2] = gamma * i + r
    return res


def flatten(nested):
    """
    :param nested: input nested list
    :return: flatten list which is 1-D
    """
    while isinstance(nested[0], list):
        tmp = []
        for i in nested:
            tmp += i
        nested = tmp
    return nested


def evaluate(x, y):
    """
    :param x: list of true states
    :param y: list of test results
    :return: FNR, FPR
    """
    x = flatten(x)
    y = flatten(y)
    p = x.count(1)
    n = x.count(0)
    compare = list(np.array(x) - np.array(y))
    fn = compare.count(1)
    fp = compare.count(-1)
    return fn / p, fp / n


def batch(x, k):
    """
    :param x:  list of patients
    :param k: number of people per batch
    :return: batch number of patients, number of people
    """
    num_g = int(len(x) / k) + (len(x) % k != 0)
    return [x[i * k:(i + 1) * k] for i in range(num_g)]
