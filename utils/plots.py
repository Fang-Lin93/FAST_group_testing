import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.cm as cm
from scipy.interpolate import splrep, splev
import matplotlib.pyplot as plt
from .utils import sir, mean_ppt
from mpl_toolkits.mplot3d import Axes3D

fig_size = (8, 5)
line_width = 2.0
scatter_width = 1


def opt_k():
    """
    largest rho = 25%
    :return:
    """
    log_rhos = np.linspace(-4, -0.6, 1000)
    rhos = np.power(10, log_rhos)
    ks = np.arange(1, 200)
    x = np.array(range(1, 101))
    optimal_k = []
    max_ppt = []
    for rho in rhos:
        ppt = mean_ppt(ks, rho, r=2)
        optimal_k += [min(ppt.argmax() + 1, 200)]
        max_ppt += [ppt.max()]

    fig, ax = plt.subplots()
    ax.plot(log_rhos, optimal_k, label='optimal_k')
    ax.plot(log_rhos, max_ppt, label='PPT')
    plt.xlabel('log_rho')
    plt.ylabel('k')
    plt.legend()
    plt.show()

    choice = [0.001, 0.005, 0.1, 0.2, 0.3]
    for rho in choice:
        ppt = np.array([mean_ppt(i, rho, r=2) for i in x])
        plt.plot(x, ppt, label='rho={}'.format(rho))
    plt.legend()
    plt.xlabel('k')
    plt.ylabel('PPT')
    plt.title('PPT_curve(low risk)')
    plt.show()
    for rho in np.round(np.arange(0.05, 0.3, 0.05), 2):
        ppt = np.array([mean_ppt(i, rho, r=2) for i in x])
        plt.plot(x, ppt, label='rho={}'.format(rho))
    plt.plot(x, [1] * len(x), label='benchmark', c='gray')
    plt.legend()
    plt.xlabel('k')
    plt.ylabel('PPT')
    plt.title('PPT_curve(high risk)')
    plt.savefig('high_risk.jpg')
    plt.show()


def ppt_surface():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    gaps = np.arange(-3, -0.5, 0.01)
    x_grid, y_grid = np.meshgrid(np.array(range(1, 101)), np.power(10.0, gaps))
    z = np.array([mean_ppt(x, y, r=2) for x, y in zip(np.ravel(x_grid), np.ravel(y_grid))])
    z = z.reshape(x_grid.shape)

    choice = [0.001, 0.0025, 0.05, 0.01, 0.1, 0.2, 0.3]
    for i in choice:
        line1x, line1y = np.meshgrid(np.array(range(1, 101)), np.array([i]))
        line1z = np.array([mean_ppt(x, y, r=2) for x, y in zip(np.ravel(line1x), np.ravel(line1y))])
        line1z = line1z.reshape(line1x.shape)
        ax.plot_wireframe(line1x, line1y, line1z)
    ax.plot_surface(x_grid, y_grid, z, cmap='viridis', alpha=0.5)
    ax.set_xlabel('K')
    ax.set_ylabel('rho')
    ax.set_zlabel('mean PPT')
    plt.title('Expected PPT Surface')
    plt.savefig('surface.jpg')
    plt.show()


def theoretical_approx(rho):
    x = np.arange(1, 100)
    y = rho * x + 1 / x
    z = 1 - np.power(1 - rho, x) + 1 / x
    plt.plot(x, y)
    plt.plot(x, z, c='g')
    plt.show()


def plot_sir():
    sir_ts = [1, 0.000001, 0]
    s, i, r = [sir_ts[0]], [sir_ts[1]], [sir_ts[2]]
    days = list(range(101))
    for i in days[1:]:
        sir_ts = sir(sir_ts)
        s += [sir_ts[0]]
        i += [sir_ts[1]]
        r += [sir_ts[2]]
    sir_data = pd.DataFrame({'Days': days, 'S': s, 'I': i, 'R': r})
    sir_data.plot(x='Days')
    plt.show()


def plot_dilution():
    prop = pd.DataFrame({'sample_proportion': [1 - n * 0.001 for n in range(0, 1001)]})
    for i in np.arange(0, 1.1, 0.1):
        prop['d={}'.format(round(i, 2))] = prop['sample_proportion'].apply(lambda x: np.power(0.02, np.power(x, i)))

    prop.plot(x='sample_proportion')
    plt.ylabel('FNR')
    plt.show()


def lowess(x, y, frac=0.05):
    return sm.nonparametric.lowess(y, x, frac=frac)[:, 1]


def spline(x, y, k=None, s=None):
    if s is None:
        s = 0.5
    if k is None:
        k = 3
    return splev(x, splrep(x, y, k=k, s=s))


def plot_scatter_line(data, frac, title=None, x_axis='pool_size', scatters=False, linestyle=None, cubicspl=False,
                      s=None, k=None):
    """
    Plot smooth interpolation of scatter data, the default method is lowess
    :param data: pd.Dataframe with x, y in columns
    :param frac: frac for lowess
    :param title: title of the output figure
    :param x_axis: column name of independent variables
    :param scatters: whether show scatters
    :param linestyle: linestyle of smooth lines
    :param cubicspl: whether use cubic spline
    :param s: smooth factor for cubic spline
    :param k: k=3 gives cubic spline, k=2 gives quadratic spline, k =1 gives linear
    :return: after run this function, use 'plt' to continue edite the figure.
    """
    smooth = pd.DataFrame(data[x_axis])
    for col in data.columns[1:]:
        if not cubicspl:
            smooth[col] = lowess(data[x_axis], data[col], frac=frac)
        else:
            smooth[col] = spline(data[x_axis], data[col], s=s, k=k)
    colors = cm.Dark2_r(np.linspace(0, 1, len(data.columns[1:])))
    plt.figure(figsize=fig_size)
    for i in range(1, len(data.columns)):
        plt.plot(data[x_axis], smooth.iloc[:, i], color=colors[i - 1], label=data.columns[i],
                 linewidth=line_width, linestyle=linestyle[i - 1] if linestyle else 'solid')
        if scatters:
            plt.scatter(data[x_axis], data.iloc[:, i], color=colors[i - 1], s=scatter_width)
    plt.legend()
    plt.title(title)
