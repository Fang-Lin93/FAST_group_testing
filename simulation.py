import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from tqdm import tqdm
from utils import batch, flatten
from utils.plots import plot_scatter_line, sg_filter
from simulation import SampleGenerator
from scipy import stats
from grouptesting import Algorithm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

np.random.seed(0)

paras_sample = {
    'size': 20000,
    'rho': 0.005,
    'household': ([1, 3, 5, 7], [0.02, 0.3, 0.46, 0.22]),  # ([3], [1]),
    'beta': 0.7,
    'replication': 50
}

test_ac = {
    'fnr': 0.02,
    'fpr': 0.001,
    'dilution_factor': 0.0
}

test_inac = {
    'fnr': 0.15,
    'fpr': 0.01,
    'dilution_factor': 0.0
}


class Individual(Algorithm):
    def __init__(self, **paras):
        super(Individual, self).__init__(**paras)
        self.name = 'Individual'

    def strategy(self, x, **kwargs):
        return self.layer(flatten(x), r=[1, 1], op=[0, 0])


class CG1(Algorithm):
    def __init__(self, **paras):
        super(CG1, self).__init__(**paras)

    def strategy(self, x, k_star):
        pooling = batch(x, k_star)
        pooling = [flatten(i) for i in pooling]
        return self.layer(pooling, r=[2, 2], op=[1, 1])


class Dorfman(Algorithm):
    def __init__(self, **paras):
        super(Dorfman, self).__init__(**paras)
        self.name = 'Dorfman'

    def strategy(self, x, **kwargs):
        flat_x = flatten(x)
        k = kwargs.get('k')
        # self.flat_x records the true position of shuffled data
        self.flat_x, flat_x = shuffle(self.flat_x, flat_x)
        pooling = batch(flat_x, k)
        return self.layer(pooling, r=[1, 1], op=[1, 1])


class RG2(Algorithm):
    def __init__(self, **paras):
        super(RG2, self).__init__(**paras)

    def strategy(self, x, k, m):
        flat_x = flatten(self.x)
        self.flat_x, flat_x = shuffle(self.flat_x, flat_x)
        pooling = batch(batch(flat_x, m), k)
        return self.layer(pooling, r=[2, 2, 2], op=[1, 1, 1])


class FeatCostly(Algorithm):
    def __init__(self, **paras):
        super(FeatCostly, self).__init__(**paras)
        self.name = 'FeatCostly'

    # k_star is the number of subgroups in a group, a subgroup contains k_s families
    def strategy(self, x, **kwargs):
        k_2 = kwargs.get('k_2')
        k_c = kwargs.get('k_c')
        x = [flatten(x[i:i + k_c]) for i in range(0, len(x), k_c)]
        pooling = batch(x, k_2)
        return self.layer(pooling, r=[1, 2, 2], op=[1, 1, 1])


class FeatCheap(Algorithm):
    def __init__(self, **paras):
        super(FeatCheap, self).__init__(**paras)
        self.name = 'FeatCheap'

    # k_star is the number of subgroups in a group, a subgroup contains k_s families
    def strategy(self, x, **kwargs):
        k_2 = kwargs.get('k_2')
        k_c = kwargs.get('k_c')
        x = [flatten(x[i:i + k_c]) for i in range(0, len(x), k_c)]
        pooling = batch(x, k_2)
        return self.layer(pooling, r=[2, 2, 2], op=[1, 1, 1])


class Optimal:
    def __init__(self, **kwargs):
        # kwargs are given by paras_sample
        self.rep = kwargs.get('replication')
        self.rho = kwargs.get('rho')
        self.beta = kwargs.get('beta')
        self.household = kwargs.get('household')
        self.m = sum(np.array(self.household[0]) * np.array(self.household[1]))
        self.sample = SampleGenerator(**kwargs)
        self.rho_c = self.m * self.rho / (1 + (self.m - 1) * self.beta)

        self.inputs = None
        self.method = None
        self.name = None
        self.result = None
        self.test_paras = None

    def performance(self, **kwargs):
        # kwargs are information of tests
        eff = []
        fnr = []
        fpr = []
        for _ in range(self.rep):
            x = self.sample.generate_fam_sample()
            gt = self.method(**kwargs)
            eff += [gt(x, **self.inputs)]
            fnr += [gt.gt_fnr]
            fpr += [gt.gt_fpr]
        self.result = {'name': self.name,
                       'efficiency': (np.mean(eff), np.std(eff)),
                       'fnr': (np.mean(fnr), np.std(fnr)),
                       'fpr': (np.mean(fpr), np.std(fpr))}

    def individual(self, **kwargs):
        self.result = None
        self.name = 'Individual'
        self.inputs = {'k': 1}
        self.method = Individual
        self.test_paras = kwargs
        self.performance(**kwargs)
        return self.result, self.inputs

    def dorfman(self, **kwargs):
        self.result = None
        k = int(1 / np.sqrt(self.rho)) + 1
        self.inputs = {'k': k}
        self.method = Dorfman
        self.name = 'Dorfman'
        if len(kwargs) != 0:
            self.test_paras = kwargs
            self.performance(**kwargs)
        return self.result, self.inputs

    def feat_costly(self, **kwargs):
        self.result = None
        k_2 = int(1 / np.sqrt(2 * self.rho_c)) + 1
        k = round(k_2 * self.m)
        self.inputs = {'k': k, 'k_2': k_2, 'k_c': 1}
        self.method = FeatCostly
        self.name = 'FeatCostly'
        if len(kwargs) != 0:
            self.test_paras = kwargs
            self.performance(**kwargs)
        return self.result, self.inputs

    def feat_cheap(self, **kwargs):
        self.result = None
        k_2 = int(1 / np.sqrt(self.rho_c)) + 1
        k = round(k_2 * self.m)
        self.inputs = {'k': k, 'k_2': k_2, 'k_c': 1}
        self.method = FeatCheap
        self.name = 'FeatCheap'
        if len(kwargs) != 0:
            self.test_paras = kwargs
            self.performance(**kwargs)
        return self.result, self.inputs

    def feat_multi_cheap(self, **kwargs):
        self.result = None
        k_2 = int(np.power(self.rho_c / self.m, -1 / 3)) + 1
        k_c = int(np.power(self.rho_c * self.m ** 2, -1 / 3)) + 1
        k = round(k_2 * self.m * k_c)
        self.inputs = {'k': k, 'k_2': k_2, 'k_c': k_c}
        self.method = FeatCheap
        self.name = 'FEAT_multi_cheap'
        if len(kwargs) != 0:
            self.test_paras = kwargs
            self.performance(**kwargs)
        return self.result, self.inputs

    def feat_multi_costly(self, **kwargs):
        self.result = None
        k_2 = int(np.power(2 * self.rho_c / self.m, -1 / 3)) + 1
        k_c = int(np.power(2 * self.rho_c * self.m ** 2, -1 / 3)) + 1
        k = round(k_2 * self.m * k_c)
        self.inputs = {'k': k, 'k_2': k_2, 'k_c': k_c}
        self.method = FeatCostly
        self.name = 'feat_multi_costly'
        if len(kwargs) != 0:
            self.test_paras = kwargs
            self.performance(**kwargs)
        return self.result, self.inputs

    def get_opt(self, name, **kwargs):
        if name == 'Individual':
            self.individual(**kwargs)
        if name == 'Dorfman':
            self.dorfman(**kwargs)
        if name == 'FeatCostly':
            self.feat_costly(**kwargs)
        if name == 'FeatCheap':
            self.feat_cheap(**kwargs)

        return self.result, self.inputs


def compare_data(accurate='ac'):
    methods_cls = [Individual, Dorfman, FeatCostly, FeatCheap]
    if accurate == 'ac':
        tests = [test_ac, test_ac, test_ac, test_ac]
    else:
        accurate = 'inac'
        tests = [test_inac, test_inac, test_inac, test_inac]
    methods = [i(**j) for i, j in zip(methods_cls, tests)]
    columns = ['pool_size']
    for i in methods:
        columns += ['fnr' + '_' + i.name, 'fpr' + '_' + i.name, 'eff' + '_' + i.name]

    replicates = paras_sample['replication']
    mean_n = sum(np.array(paras_sample['household'][0]) * np.array(paras_sample['household'][1]))
    pool_size = 101
    n_families = int(pool_size / mean_n)
    k_set = np.round(np.arange(1, n_families) * mean_n).astype(int)
    simulations = pd.DataFrame(columns=columns)

    sample = SampleGenerator(**paras_sample)

    for k in tqdm(k_set):
        k_2 = int(max(round(k / mean_n), 1))  # number of families
        paras = {'k': k, 'k_2': k_2, 'k_c': 1}
        for rep in range(replicates):
            new_row = {'pool_size': k}
            x = sample.generate_fam_sample()
            for gt in methods:
                new_row['eff' + '_' + gt.name] = gt(x, **paras)
                new_row['fnr' + '_' + gt.name], new_row['fpr' + '_' + gt.name] = gt.gt_fnr, gt.gt_fpr
            simulations = simulations.append(new_row, ignore_index=True)
    simulations.to_csv(accurate + 'sim_rho{}_beta{}.csv'.format(paras_sample['rho'], paras_sample['beta']), index=False)


def compare_plots(ac='ac'):
    simulations = pd.read_csv(ac + 'sim_rho{}_beta{}.csv'.format(paras_sample['rho'], paras_sample['beta']))
    num_methods = int((len(simulations.columns) - 1) / 3)
    data_eff = pd.concat([simulations['pool_size'], simulations.loc[:, simulations.columns.str.startswith('eff')]],
                         axis=1)
    data_fnr = pd.concat([simulations['pool_size'], simulations.loc[:, simulations.columns.str.startswith('fnr')]],
                         axis=1)
    data_fpr = pd.concat([simulations['pool_size'], simulations.loc[:, simulations.columns.str.startswith('fpr')]],
                         axis=1)

    # efficiency
    eff_mean = data_eff.groupby(['pool_size']).mean()
    eff_std = data_eff.groupby(['pool_size']).std()
    colors = cm.Dark2_r(np.linspace(0, 1, num_methods))
    fig, axes = plt.subplots(1, 3, figsize=(30, 8))
    for i in range(len(eff_mean.columns)):
        # data_smooth = sg_filter(eff_mean.index, eff_mean.iloc[:, i], wind=11, degree=3)
        axes[0].plot(eff_mean.index, eff_mean.iloc[:, i], color=colors[i], label=eff_mean.columns[i], linewidth=2, )
        axes[0].fill_between(eff_mean.index, eff_mean.iloc[:, i] - eff_std.iloc[:, i],
                             eff_mean.iloc[:, i] + eff_std.iloc[:, i],
                             alpha=0.2, facecolor=colors[i])
    axes[0].legend()

    for col in eff_mean.columns:
        max_eff = round(eff_mean[col].max(), 1)
        arg_eff = eff_mean[col].idxmax()
        if col != 'eff_Individual':
            axes[0].annotate('{}: Efficiency={},k={}'.format(col, max_eff, arg_eff), xy=(arg_eff, max_eff))
    axes[0].set_xlabel('Pool size')
    axes[0].set_ylabel('Efficiency')
    axes[0].legend()
    axes[0].set_title('Efficiency( prevalence={}, infection rate={})'.format(paras_sample['rho'], paras_sample['beta']))

    # plot fnr
    for i in range(len(data_fnr.columns) - 1):
        sns.regplot(data_fnr['pool_size'], data_fnr.iloc[:, i + 1], color=colors[i], label=data_fnr.columns[i + 1],
                    ax=axes[1], x_bins=10, x_ci=75, truncate=True)
    axes[1].set_xlabel('Pool size')
    axes[1].set_ylabel('FNR')
    axes[1].legend()
    axes[1].set_title('FNR( prevalence={}, infection rate={})'.format(paras_sample['rho'], paras_sample['beta']))

    # plot fpr
    for i in range(len(data_fpr.columns) - 1):
        sns.regplot(data_fpr['pool_size'], data_fpr.iloc[:, i + 1], color=colors[i], label=data_fpr.columns[i + 1],
                    ax=axes[2], x_bins=10, x_ci=75, truncate=True)
    axes[2].set_xlabel('Pool size')
    axes[2].set_ylabel('FPR')
    axes[2].legend()
    axes[2].set_title('FPR( prevalence={}, infection rate={})'.format(paras_sample['rho'], paras_sample['beta']))
    fig.savefig('Compare_rho{}_beta{}.jpg'.format(paras_sample['rho'], paras_sample['beta']))
    fig.show()


def beta_rho_data(var):
    var_sample = paras_sample.copy()
    var_sample['replication'] = 30
    methods_cls = [Dorfman, FeatCostly, FeatCheap]
    # change the parameters of the tests here
    tests = [test_ac, test_ac, test_ac, test_ac]
    methods = [i(**j) for i, j in zip(methods_cls, tests)]

    if var == 'beta':
        columns = ['beta']
        var_set = np.arange(0, 1.1, 0.1)
    if var == 'rho':
        columns = ['rho']
        var_set = np.power(10, np.arange(-4, -1.8, 0.2))

    for i in methods:
        columns += ['eff_' + i.name, 'std_' + i.name, 'pool_' + i.name]
    data = pd.DataFrame(columns=columns)

    for value in tqdm(var_set):
        if var == 'beta':
            var_sample['beta'] = value
            new_row = {'beta': value}
        if var == 'rho':
            var_sample['rho'] = value
            new_row = {'rho': value}
        opt = Optimal(**var_sample)
        for gt in methods:
            res = opt.get_opt(gt.name, **gt.test)
            new_row['eff_' + gt.name] = res[0]['efficiency'][0]
            new_row['std_' + gt.name] = res[0]['efficiency'][1]
            new_row['pool_' + gt.name] = res[1]['k']
        data = data.append(new_row, ignore_index=True)

    if var == 'beta':
        data.to_csv('beta_compare.csv', index=False)
    if var == 'rho':
        data.to_csv('rho_compare.csv', index=False)


def beta_rho_plot(var):
    if var == 'beta':
        data = pd.read_csv('beta_compare.csv', index_col='beta')
    if var == 'rho':
        data = pd.read_csv('rho_compare.csv', index_col='rho')
    data_eff = data.loc[:, data.columns.str.startswith('eff')]
    data_std = data.loc[:, data.columns.str.startswith('std')]
    data_pool = data.loc[:, data.columns.str.startswith('pool')]
    num_c = len(data_eff.columns)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    colors = cm.Dark2_r(np.linspace(0, 1, num_c))
    for i in range(len(data_eff.columns)):
        axes[0].plot(data_pool.index, data_eff.iloc[:, i], color=colors[i], label=data_pool.columns[i], linewidth=2, )
        axes[1].plot(data_eff.index, data_eff.iloc[:, i], color=colors[i], label=data_eff.columns[i], linewidth=2, )
        axes[1].fill_between(data_pool.index, data_eff.iloc[:, i] - data_std.iloc[:, i],
                             data_eff.iloc[:, i] + data_std.iloc[:, i],
                             alpha=0.2, facecolor=colors[i])

    axes[0].legend()
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['top'].set_visible(False)
    axes[1].legend()
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)
    if var == 'beta':
        axes[0].set_xlabel('Infection rate')
        axes[0].set_ylabel('Pool size')
        axes[1].set_xlabel('Infection rate')
        axes[1].set_ylabel('Efficiency')
        fig.savefig('beta_compare.jpg')
    if var == 'rho':
        axes[0].set_xlabel('Prevalence')
        axes[0].set_ylabel('Pool size')
        axes[1].set_xlabel('Prevalence')
        axes[1].set_ylabel('Efficiency')
        axes[0].set_xscale('log')
        axes[1].set_xscale('log')
        fig.savefig('rho_compare.jpg')
    fig.show()


if __name__ == '__main__':
    new_data = False
    if new_data:
        compare_data('ac')
        compare_data('inac')
        beta_rho_data('beta')
        beta_rho_data('rho')

    compare_plots()
    beta_rho_plot('beta')
    beta_rho_plot('rho')

    example = Optimal(**paras_sample)
    res = example.feat_cheap(**test_inac)
    print(res)