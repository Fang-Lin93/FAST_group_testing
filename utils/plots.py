import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.cm as cm
from scipy.interpolate import splrep, splev
import matplotlib.pyplot as plt
import seaborn as sns
from .utils import sir, mean_ppt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
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


def sg_filter(x, y, wind=11, degree=3):
    itp = interp1d(x, y, kind='linear')
    x_bal = np.linspace(x.min(), x.max(), len(x))
    return savgol_filter(itp(x_bal), wind, degree)


def spline(x, y, k=None, s=None):
    if s is None:
        s = 0.5
    if k is None:
        k = 3
    return splev(x, splrep(x, y, k=k, s=s))

#
# def plot_scatter_line(data, frac, title=None, x_axis=None, scatters=False, line_style=None, cubicspl=False, sgf=False,
#                       s=None, k=None, split=False, colors=None, mark=False, mark_s=False, wind=11, degree=3):
#     """
#     Plot smooth interpolation of scatter data, the default method is lowess
#     :param colors:
#     :param split: where to perform subplots
#     :param data: pd.Dataframe with x, y in columns
#     :param frac: frac for lowess
#     :param title: title of the output figure
#     :param x_axis: column name of independent variables
#     :param scatters: whether show scatters
#     :param line_style: linestyle of smooth lines
#     :param cubicspl: whether use cubic spline
#     :param s: smooth factor for cubic spline
#     :param k: k=3 gives cubic spline, k=2 gives quadratic spline, k =1 gives linear
#     :return: after run this function, use 'plt' to continue edite the figure.
#     """
#     if x_axis is None:
#         x_axis = data.columns[0]
#     if split is False:
#         plt.figure(figsize=fig_size)
#     smooth = pd.DataFrame(data[x_axis])
#     for col in data.columns[1:]:
#         if cubicspl:
#             smooth[col] = spline(data[x_axis], data[col], s=s, k=k)
#         elif sgf:
#             smooth[col] = sg_filter(data[x_axis], data[col], wind=wind, degree=degree)
#         else:
#             smooth[col] = lowess(data[x_axis], data[col], frac=frac)
#     if colors is None:
#         colors = cm.Dark2_r(np.linspace(0, 1, len(data.columns[1:])))
#     for i in range(1, len(data.columns)):
#         if split is True:
#             plt.subplot(1, len(data.columns) - 1, i)
#         plt.plot(data[x_axis], smooth.iloc[:, i], color=colors[i - 1], label=data.columns[i],
#                  linewidth=line_width, linestyle=line_style[i - 1] if line_style else 'solid')
#         if scatters:
#             plt.scatter(data[x_axis], data.iloc[:, i], color=colors[i - 1], s=scatter_width)
#
#         # marker
#         if mark_s is True:
#             plt.vlines(data[x_axis], 0, smooth.iloc[:, i], linestyle="dashed")
#             plt.hlines(smooth.iloc[:, i], 0, data[x_axis], linestyle="dashed")
#         if mark is True:
#             plt.vlines(data[x_axis], 0, data.iloc[:, i], linestyle="dashed")
#             plt.hlines(data.iloc[:, i], 0, data[x_axis], linestyle="dashed")
#         plt.xlabel(x_axis)
#         plt.legend()
#     plt.title(title)


def plot_scatter_line( data, frac, axes=None, scatters=False, line_style=None, smooth = None,
                      s=None, k=None, colors=None, wind=11, degree=3):
    """
    Plot smooth interpolation of scatter data, the default method is lowess
    :param colors:
    :param split: where to perform subplots
    :param data: pd.Dataframe with x, y in columns
    :param frac: frac for lowess
    :param title: title of the output figure
    :param x_axis: column name of independent variables
    :param scatters: whether show scatters
    :param line_style: linestyle of smooth lines
    :param cubicspl: whether use cubic spline
    :param s: smooth factor for cubic spline
    :param k: k=3 gives cubic spline, k=2 gives quadratic spline, k =1 gives linear
    :return: after run this function, use 'plt' to continue edite the figure.
    """
    if axes is None:
        fig, axes = plt.subplots(figsize=fig_size)
    x_axis = data.columns[0]

    smoo = pd.DataFrame(data[x_axis])
    for col in data.columns[1:]:
        if smooth is 'cubicspline':
            smoo[col] = spline(data[x_axis], data[col], s=s, k=k)
        elif smooth is 'sgf':
            smoo[col] = sg_filter(data[x_axis], data[col], wind=wind, degree=degree)
        elif smooth is 'lowess':
            smoo[col] = lowess(data[x_axis], data[col], frac=frac)
        else:
            pass
    if colors is None:
        colors = cm.Dark2_r(np.linspace(0, 1, len(data.columns[1:])))
    if smooth is not None:
        for i in range(1, len(data.columns)):
            axes.plot(data[x_axis], smoo.iloc[:, i], color=colors[i - 1], label=data.columns[i],
                      linewidth=line_width, linestyle=line_style[i - 1] if line_style else 'solid')
        if scatters:
            axes.scatter(data[x_axis], data.iloc[:, i], color=colors[i - 1], s=scatter_width)
    else:
        for i in range(1, len(data.columns)):
            sns.regplot(x=data[x_axis], y=data.iloc[:, i], color=colors[i - 1],
                        line_kws={'linewidth': line_width}, ax=axes, x_estimator=np.mean)


    plt.xlabel(x_axis)
    plt.legend()
    return fig, axes

