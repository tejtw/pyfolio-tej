#
# Copyright 2018 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import datetime
from collections import OrderedDict
from functools import wraps
import sys
import empyrical as ep
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import scipy as sp
from matplotlib import figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.ticker import FuncFormatter

import seaborn as sns

from . import capacity
from . import pos
from . import timeseries
from . import txn
from . import utils
from .utils import APPROX_BDAYS_PER_MONTH, MM_DISPLAY_UNIT


def customize(func):
    """
    Decorator to set plotting context and axes style during function call.
    """

    @wraps(func)
    def call_w_context(*args, **kwargs):
        set_context = kwargs.pop("set_context", True)
        if set_context:
            with plotting_context(), axes_style():
                plt.rcParams['axes.unicode_minus'] = False                 # 20230823 (by MRC) 負號顯示問題
                #plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']     # 20230823 (by MRC) 中文顯示問題

                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    return call_w_context


def plotting_context(context="notebook", font_scale=1.5, rc=None):
    """
    Create pyfolio default plotting style context.

    Under the hood, calls and returns seaborn.plotting_context() with
    some custom settings. Usually you would use in a with-context.

    Parameters
    ----------
    context : str, optional
        Name of seaborn context.
    font_scale : float, optional
        Scale font by factor font_scale.
    rc : dict, optional
        Config flags.
        By default, {'lines.linewidth': 1.5}
        is being used and will be added to any
        rc passed in, unless explicitly overriden.

    Returns
    -------
    seaborn plotting context

    Example
    -------
    >>> with pyfolio.plotting.plotting_context(font_scale=2):
    >>>    pyfolio.create_full_tear_sheet(..., set_context=False)

    See also
    --------
    For more information, see seaborn.plotting_context().

    """
    if rc is None:
        rc = {}

    rc_default = {"lines.linewidth": 1.5}

    # Add defaults if they do not exist
    for name, val in rc_default.items():
        rc.setdefault(name, val)

    return sns.plotting_context(context=context, font_scale=font_scale, rc=rc)


def axes_style(style="darkgrid", rc=None):
    """
    Create pyfolio default axes style context.

    Under the hood, calls and returns seaborn.axes_style() with
    some custom settings. Usually you would use in a with-context.

    Parameters
    ----------
    style : str, optional
        Name of seaborn style.
    rc : dict, optional
        Config flags.

    Returns
    -------
    seaborn plotting context

    Example
    -------
    >>> with pyfolio.plotting.axes_style(style='whitegrid'):
    >>>    pyfolio.create_full_tear_sheet(..., set_context=False)

    See also
    --------
    For more information, see seaborn.plotting_context().

    """
    if rc is None:
        rc = {}

    # 20230823 (by MRC) 中文顯示問題
    # plt.rcParams預設是['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
    # windows可以修改為Microsoft JhengHei、mingliu、'Arial Unicode MS'、或 DFKai-SB
    # Mac可設為'Arial Unicode MS'
    rc_default = {"font.sans-serif": ['Arial Unicode MS']}

    # Add defaults if they do not exist
    for name, val in rc_default.items():
        rc.setdefault(name, val)

    return sns.axes_style(style=style, rc=rc)

# 20231101 modify by yo
def transfer_chinese(cnm):
    """
    Decorator to set transfer_chinese during function call.
    """
    def call_set_language(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            set_context = kwargs.pop("set_chinese_name", False)
            if set_context:                  
                return func(cname=cnm,*args,**kwargs)
            else:  
                return func(*args, **kwargs)
        return wrapper
    return call_set_language


def plot_monthly_returns_heatmap(returns, ax=None, cname=None, **kwargs):
    """
    Plots a heatmap of returns by month.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to seaborn plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    monthly_ret_table = ep.aggregate_returns(returns, "monthly")
    monthly_ret_table = monthly_ret_table.unstack().round(3)

    sns.heatmap(
        monthly_ret_table.fillna(0) * 100.0,
        annot=True,
        annot_kws={"size": 9},
        alpha=1.0,
        center=0.0,
        cbar=False,
        #cmap=matplotlib.cm.RdYlGn,
        cmap=matplotlib.cm.RdYlGn_r,  # 20230822 (by MRC) 配合台股習慣，紅色上漲，綠色下跌
        ax=ax,
        **kwargs,
    )
    ax.set_ylabel("Year")
    ax.set_xlabel("Month")
    ax.set_title("Monthly returns (%)")

    # 202311101 modify to show chinese name(yo)
    if cname:
        _name = sys._getframe(0).f_code.co_name
        ax.set_ylabel(cname[_name]['ylabel'])
        ax.set_xlabel(cname[_name]['xlabel'])
        ax.set_title(cname[_name]['title'])

    return ax


def plot_annual_returns(returns, ax=None, cname=None, **kwargs):
    """
    Plots a bar graph of returns by year.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    x_axis_formatter = FuncFormatter(utils.percentage)
    ax.xaxis.set_major_formatter(FuncFormatter(x_axis_formatter))
    ax.tick_params(axis="x", which="major")

    ann_ret_df = pd.DataFrame(ep.aggregate_returns(returns, "yearly"))

    ax.axvline(
        100 * ann_ret_df.values.mean(),
        color="steelblue",
        linestyle="--",
        lw=4,
        alpha=0.7,
    )
    (100 * ann_ret_df.sort_index(ascending=False)).plot(
        ax=ax, kind="barh", alpha=0.70, **kwargs
    )
    ax.axvline(0.0, color="black", linestyle="-", lw=3)

    ax.set_ylabel("Year")
    ax.set_xlabel("Returns")
    ax.set_title("Annual returns")
    ax.legend(["Mean"], frameon=True, framealpha=0.5)

    # 20231010 modify to show chinese name(yo)
    if cname:
        _name = sys._getframe(0).f_code.co_name
        ax.set_ylabel(cname[_name]['ylabel'])
        ax.set_xlabel(cname[_name]['xlabel'])
        ax.set_title(cname[_name]['title'])

    return ax


def plot_monthly_returns_dist(returns, ax=None, cname=None, **kwargs):
    """
    Plots a distribution of monthly returns.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    x_axis_formatter = FuncFormatter(utils.percentage)
    ax.xaxis.set_major_formatter(FuncFormatter(x_axis_formatter))
    ax.tick_params(axis="x", which="major")

    monthly_ret_table = ep.aggregate_returns(returns, "monthly")

    ax.hist(
        100 * monthly_ret_table,
        color="orangered",
        alpha=0.80,
        bins=20,
        **kwargs,
    )

    ax.axvline(
        100 * monthly_ret_table.mean(),
        color="gold",
        linestyle="--",
        lw=4,
        alpha=1.0,
    )

    ax.axvline(0.0, color="black", linestyle="-", lw=3, alpha=0.75)
    ax.legend(["Mean"], frameon=True, framealpha=0.5)
    ax.set_ylabel("Number of months")
    ax.set_xlabel("Returns")
    ax.set_title("Distribution of monthly returns")

    # 20231101 modify to show chinese name(yo)
    if cname:
        _name = sys._getframe(0).f_code.co_name
        ax.set_ylabel(cname[_name]['ylabel'])
        ax.set_xlabel(cname[_name]['xlabel'])
        ax.set_title(cname[_name]['title'])

    return ax


def plot_holdings(returns, positions, legend_loc="best", ax=None, cname=None, **kwargs):
    """
    Plots total amount of stocks with an active position, either short
    or long. Displays daily total, daily average per month, and
    all-time daily average.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    positions : pd.DataFrame, optional
        Daily net position values.
         - See full explanation in tears.create_full_tear_sheet.
    legend_loc : matplotlib.loc, optional
        The location of the legend on the plot.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    positions = positions.copy().drop("cash", axis="columns")
    df_holdings = positions.replace(0, np.nan).count(axis=1)
    df_holdings_by_month = df_holdings.resample("1M").mean()
    df_holdings.plot(color="steelblue", alpha=0.6, lw=0.5, ax=ax, **kwargs)
    df_holdings_by_month.plot(color="orangered", lw=2, ax=ax, **kwargs)
    ax.axhline(df_holdings.values.mean(), color="steelblue", ls="--", lw=3)

    ax.set_xlim((returns.index[0], returns.index[-1]))

    leg = ax.legend(
        [
            "Daily holdings",
            "Average daily holdings, by month",
            "Average daily holdings, overall",
        ],
        loc=legend_loc,
        frameon=True,
        framealpha=0.5,
    )
    leg.get_frame().set_edgecolor("black")

    ax.set_title("Total holdings")
    ax.set_ylabel("Holdings")
    ax.set_xlabel("")

    # 20231101 modify to show chinese name(yo)
    if cname:
        _name = sys._getframe(0).f_code.co_name
        ax.set_ylabel(cname[_name]['ylabel'])
        ax.set_xlabel(cname[_name]['xlabel'])
        ax.set_title(cname[_name]['title'])

    return ax


def plot_long_short_holdings(
    returns, positions, legend_loc="upper left", ax=None, cname=None,  **kwargs
):
    """
    Plots total amount of stocks with an active position, breaking out
    short and long into transparent filled regions.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    positions : pd.DataFrame, optional
        Daily net position values.
         - See full explanation in tears.create_full_tear_sheet.
    legend_loc : matplotlib.loc, optional
        The location of the legend on the plot.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.

    """

    if ax is None:
        ax = plt.gca()

    positions = positions.drop("cash", axis="columns")
    positions = positions.replace(0, np.nan)
    df_longs = positions[positions > 0].count(axis=1)
    df_shorts = positions[positions < 0].count(axis=1)
    lf = ax.fill_between(
        df_longs.index, 0, df_longs.values, color="r", alpha=0.5, lw=2.0
    )                                                                     # 20230822 (by MRC) 配合台股習慣 color="g"->"r"
    sf = ax.fill_between(
        df_shorts.index, 0, df_shorts.values, color="g", alpha=0.5, lw=2.0
    )                                                                     # 20230822 (by MRC) 配合台股習慣 color="r"->"g"

    bf = patches.Rectangle([0, 0], 1, 1, color="darkgoldenrod")
    leg = ax.legend(
        [lf, sf, bf],
        [
            "Long (max: %s, min: %s)" % (df_longs.max(), df_longs.min()),
            "Short (max: %s, min: %s)" % (df_shorts.max(), df_shorts.min()),
            "Overlap",
        ],
        loc=legend_loc,
        frameon=True,
        framealpha=0.5,
    )
    leg.get_frame().set_edgecolor("black")

    ax.set_xlim((returns.index[0], returns.index[-1]))
    ax.set_title("Long and short holdings")
    ax.set_ylabel("Holdings")
    ax.set_xlabel("")

    # 20231101 modify to show chinese name(yo)
    if cname:
        _name = sys._getframe(0).f_code.co_name
        ax.set_ylabel(cname[_name]['ylabel'])
        ax.set_xlabel(cname[_name]['xlabel'])
        ax.set_title(cname[_name]['title'])

    return ax


def plot_drawdown_periods(returns, top=10, ax=None, cname=None, **kwargs):
    """
    Plots cumulative returns highlighting top drawdown periods.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    top : int, optional
        Amount of top drawdowns periods to plot (default 10).
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    y_axis_formatter = FuncFormatter(utils.two_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    df_cum_rets = ep.cum_returns(returns, starting_value=1.0)
    df_drawdowns = (timeseries.gen_drawdown_table(returns, top=top)).dropna(how = 'all')
    df_cum_rets.plot(ax=ax, **kwargs)

    lim = ax.get_ylim()
    colors = sns.cubehelix_palette(len(df_drawdowns))[::-1]
    for i, (peak, recovery) in df_drawdowns[
        ["Peak date", "Recovery date"]
    ].iterrows():
        if pd.isnull(recovery):
            recovery = returns.index[-1]
        ax.fill_between(
            (peak, recovery), lim[0], lim[1], alpha=0.4, color=colors[i]
        )
    ax.set_ylim(lim)
    ax.set_title("Top %i drawdown periods" % top)
    ax.set_ylabel("Cumulative returns")
    ax.legend(["Portfolio"], loc="upper left", frameon=True, framealpha=0.5)
    ax.set_xlabel("")

    # 20231101 modify to show chinese name(yo)
    if cname:
        _name = sys._getframe(0).f_code.co_name
        cname[_name]['title'] = cname[_name]['title'].format(top=top)
        ax.set_ylabel(cname[_name]['ylabel'])
        ax.set_xlabel(cname[_name]['xlabel'])
        ax.set_title(cname[_name]['title'])
    
    return ax


def plot_drawdown_underwater(returns, ax=None, cname=None, **kwargs):
    """
    Plots how far underwaterr returns are over time, or plots current
    drawdown vs. date.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    y_axis_formatter = FuncFormatter(utils.percentage)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    df_cum_rets = ep.cum_returns(returns, starting_value=1.0)
    running_max = np.maximum.accumulate(df_cum_rets)
    underwater = -100 * ((running_max - df_cum_rets) / running_max)
    (underwater).plot(ax=ax, kind="area", color="coral", alpha=0.7, **kwargs)
    ax.set_ylabel("Drawdown")
    ax.set_title("Underwater plot")
    ax.set_xlabel("")

    # 20231101 modify to show chinese name(yo)
    if cname:
        _name = sys._getframe(0).f_code.co_name
        ax.set_ylabel(cname[_name]['ylabel'])
        ax.set_xlabel(cname[_name]['xlabel'])
        ax.set_title(cname[_name]['title'])

    return ax


def plot_perf_stats(returns, factor_returns, cname=None, ax=None):
    """
    Create box plot of some performance metrics of the strategy.
    The width of the box whiskers is determined by a bootstrap.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    factor_returns : pd.Series
        Daily noncumulative returns of the benchmark factor to which betas are
        computed. Usually a benchmark such as market returns.
         - This is in the same style as returns.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    bootstrap_values = timeseries.perf_stats_bootstrap(
        returns, factor_returns, return_stats=False
    )
    bootstrap_values = bootstrap_values.drop("Kurtosis", axis="columns")

    sns.boxplot(data=bootstrap_values, orient="h", ax=ax)
    
    # 20231010 modify to show chinese name(yo)
    if cname:
        _name = sys._getframe(0).f_code.co_name
        ax.set_title(cname[_name]['title'])
    return ax


STAT_FUNCS_PCT = [
    "Annual return",
    "Cumulative returns",
    "Annual volatility",
    "Max drawdown",
    "Daily value at risk",
    "Daily turnover",
]


def show_perf_stats(
    returns,
    factor_returns=None,
    positions=None,
    transactions=None,
    turnover_denom="AGB",
    live_start_date=None,
    bootstrap=False,
    header_rows=None,
):
    """
    Prints some performance metrics of the strategy.

    - Shows amount of time the strategy has been run in backtest and
      out-of-sample (in live trading).

    - Shows Omega ratio, max drawdown, Calmar ratio, annual return,
      stability, Sharpe ratio, annual volatility, alpha, and beta.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    factor_returns : pd.Series, optional
        Daily noncumulative returns of the benchmark factor to which betas are
        computed. Usually a benchmark such as market returns.
         - This is in the same style as returns.
    positions : pd.DataFrame, optional
        Daily net position values.
         - See full explanation in create_full_tear_sheet.
    transactions : pd.DataFrame, optional
        Prices and amounts of executed trades. One row per trade.
        - See full explanation in tears.create_full_tear_sheet
    turnover_denom : str, optional
        Either AGB or portfolio_value, default AGB.
        - See full explanation in txn.get_turnover.
    live_start_date : datetime, optional
        The point in time when the strategy began live trading, after
        its backtest period.
    bootstrap : boolean, optional
        Whether to perform bootstrap analysis for the performance
        metrics.
         - For more information, see timeseries.perf_stats_bootstrap
    header_rows : dict or OrderedDict, optional
        Extra rows to display at the top of the displayed table.
    """

    if bootstrap:
        perf_func = timeseries.perf_stats_bootstrap
    else:
        perf_func = timeseries.perf_stats

    perf_stats_all = perf_func(
        returns,
        factor_returns=factor_returns,
        positions=positions,
        transactions=transactions,
        turnover_denom=turnover_denom,
    )

    date_rows = OrderedDict()
    if len(returns.index) > 0:
        date_rows["Start date"] = returns.index[0].strftime("%Y-%m-%d")
        date_rows["End date"] = returns.index[-1].strftime("%Y-%m-%d")

    if live_start_date is not None:
        live_start_date = ep.utils.get_utc_timestamp(live_start_date)
        returns_is = returns[returns.index < live_start_date]
        returns_oos = returns[returns.index >= live_start_date]

        positions_is = None
        positions_oos = None
        transactions_is = None
        transactions_oos = None

        if positions is not None:
            positions_is = positions[positions.index < live_start_date]
            positions_oos = positions[positions.index >= live_start_date]
            if transactions is not None:
                transactions_is = transactions[
                    (transactions.index < live_start_date)
                ]
                transactions_oos = transactions[
                    (transactions.index > live_start_date)
                ]

        perf_stats_is = perf_func(
            returns_is,
            factor_returns=factor_returns,
            positions=positions_is,
            transactions=transactions_is,
            turnover_denom=turnover_denom,
        )

        perf_stats_oos = perf_func(
            returns_oos,
            factor_returns=factor_returns,
            positions=positions_oos,
            transactions=transactions_oos,
            turnover_denom=turnover_denom,
        )
        if len(returns.index) > 0:
            date_rows["In-sample months"] = int(
                len(returns_is) / APPROX_BDAYS_PER_MONTH
            )
            date_rows["Out-of-sample months"] = int(
                len(returns_oos) / APPROX_BDAYS_PER_MONTH
            )

        perf_stats = pd.concat(
            OrderedDict(
                [
                    ("In-sample", perf_stats_is),
                    ("Out-of-sample", perf_stats_oos),
                    ("All", perf_stats_all),
                ]
            ),
            axis=1,
        )
    else:
        if len(returns.index) > 0:
            date_rows["Total months"] = int(
                len(returns) / APPROX_BDAYS_PER_MONTH
            )
        perf_stats = pd.DataFrame(perf_stats_all, columns=["Backtest"])

    for column in perf_stats.columns:
        for stat, value in perf_stats[column].items():
            if stat in STAT_FUNCS_PCT:
                perf_stats.loc[stat, column] = (
                    str(np.round(value * 100, 3)) + "%"
                )
    if header_rows is None:
        header_rows = date_rows
    else:
        header_rows = OrderedDict(header_rows)
        header_rows.update(date_rows)

    utils.print_table(
        perf_stats,
        float_format="{0:.2f}".format,
        header_rows=header_rows,
    )
    return perf_stats

def plot_returns(returns, live_start_date=None, ax=None, cname=None):
    """
    Plots raw returns over time.

    Backtest returns are in green, and out-of-sample (live trading)
    returns are in red.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    live_start_date : datetime, optional
        The date when the strategy began live trading, after
        its backtest period. This date should be normalized.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    ax.set_label("")
    ax.set_ylabel("Returns")

    if live_start_date is not None:
        live_start_date = ep.utils.get_utc_timestamp(live_start_date)
        is_returns = returns.loc[returns.index < live_start_date]
        oos_returns = returns.loc[returns.index >= live_start_date]
        is_returns.plot(ax=ax, color="g")
        oos_returns.plot(ax=ax, color="r")

    else:
        returns.plot(ax=ax, color="g")
    
    # 20231010 modify to show chinese name(yo)
    if cname:
        _name = sys._getframe(0).f_code.co_name
        ax.set_ylabel(cname[_name]['ylabel'])
        ax.set_xlabel(cname[_name]['xlabel'])
        ax.set_title(cname[_name]['title'])

    return ax


def plot_rolling_returns(
    returns,
    factor_returns=None,
    live_start_date=None,
    logy=False,
    cone_std=None,
    legend_loc="best",
    volatility_match=False,
    cone_function=timeseries.forecast_cone_bootstrap,
    ax=None,
    cname=None,
    **kwargs,
):
    """
    Plots cumulative rolling returns versus some benchmarks'.

    Backtest returns are in green, and out-of-sample (live trading)
    returns are in red.

    Additionally, a non-parametric cone plot may be added to the
    out-of-sample returns region.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    factor_returns : pd.Series, optional
        Daily noncumulative returns of the benchmark factor to which betas are
        computed. Usually a benchmark such as market returns.
         - This is in the same style as returns.
    live_start_date : datetime, optional
        The date when the strategy began live trading, after
        its backtest period. This date should be normalized.
    logy : bool, optional
        Whether to log-scale the y-axis.
    cone_std : float, or tuple, optional
        If float, The standard deviation to use for the cone plots.
        If tuple, Tuple of standard deviation values to use for the cone plots
         - See timeseries.forecast_cone_bounds for more details.
    legend_loc : matplotlib.loc, optional
        The location of the legend on the plot.
    volatility_match : bool, optional
        Whether to normalize the volatility of the returns to those of the
        benchmark returns. This helps compare strategies with different
        volatilities. Requires passing of benchmark_rets.
    cone_function : function, optional
        Function to use when generating forecast probability cone.
        The function signiture must follow the form:
        def cone(in_sample_returns (pd.Series),
                 days_to_project_forward (int),
                 cone_std= (float, or tuple),
                 starting_value= (int, or float))
        See timeseries.forecast_cone_bootstrap for an example.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    ax.set_xlabel("")
    ax.set_ylabel("Cumulative returns")
    ax.set_yscale("log" if logy else "linear")

    if volatility_match and factor_returns is None:
        raise ValueError(
            "volatility_match requires passing of " "factor_returns."
        )
    elif volatility_match and factor_returns is not None:
        bmark_vol = factor_returns.loc[returns.index].std()
        returns = (returns / returns.std()) * bmark_vol

    cum_rets = ep.cum_returns(returns, 1.0)

    y_axis_formatter = FuncFormatter(utils.two_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    if factor_returns is not None:
        cum_factor_returns = ep.cum_returns(
            factor_returns[cum_rets.index], 1.0
        )
        cum_factor_returns.plot(
            lw=2,
            color="gray",
            label=factor_returns.name,
            alpha=0.60,
            ax=ax,
            **kwargs,
        )

    if live_start_date is not None:
        live_start_date = ep.utils.get_utc_timestamp(live_start_date)
        is_cum_returns = cum_rets.loc[cum_rets.index < live_start_date]
        oos_cum_returns = cum_rets.loc[cum_rets.index >= live_start_date]
    else:
        is_cum_returns = cum_rets
        oos_cum_returns = pd.Series([])

    is_cum_returns.plot(
        lw=3, color="forestgreen", alpha=0.6, label="Backtest", ax=ax, **kwargs
    )

    if len(oos_cum_returns) > 0:
        oos_cum_returns.plot(
            lw=4, color="red", alpha=0.6, label="Live", ax=ax, **kwargs
        )

        if cone_std is not None:
            if isinstance(cone_std, (float, int)):
                cone_std = [cone_std]

            is_returns = returns.loc[returns.index < live_start_date]
            cone_bounds = cone_function(
                is_returns,
                len(oos_cum_returns),
                cone_std=cone_std,
                starting_value=is_cum_returns[-1],
            )

            cone_bounds = cone_bounds.set_index(oos_cum_returns.index)
            for std in cone_std:
                ax.fill_between(
                    cone_bounds.index,
                    cone_bounds[float(std)],
                    cone_bounds[float(-std)],
                    color="steelblue",
                    alpha=0.5,
                )

    if legend_loc is not None:
        ax.legend(loc=legend_loc, frameon=True, framealpha=0.5)
    ax.axhline(1.0, linestyle="--", color="black", lw=2)
    # 20231101 modify to show chinese name(yo)
    if cname:
        _name = sys._getframe(0).f_code.co_name
        ax.set_ylabel(cname[_name]['ylabel'])
        ax.set_xlabel(cname[_name]['xlabel'])
        ax.set_title(cname[_name]['title'])

    return ax


def plot_rolling_beta(
    returns, factor_returns, legend_loc="best", ax=None, cname=None, **kwargs
):
    """
    Plots the rolling 6-month and 12-month beta versus date.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    factor_returns : pd.Series
        Daily noncumulative returns of the benchmark factor to which betas are
        computed. Usually a benchmark such as market returns.
         - This is in the same style as returns.
    legend_loc : matplotlib.loc, optional
        The location of the legend on the plot.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    y_axis_formatter = FuncFormatter(utils.two_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    ax.set_title("Rolling portfolio beta to " + str(factor_returns.name))
    ax.set_ylabel("Beta")
    rb_1 = timeseries.rolling_beta(
        returns, factor_returns, rolling_window=APPROX_BDAYS_PER_MONTH * 6
    )
    rb_2 = timeseries.rolling_beta(
        returns, factor_returns, rolling_window=APPROX_BDAYS_PER_MONTH * 12
    )

    # 20230822 (MRC) 避免線段超出顯示範圍，且把位置修改在這邊(原935)，
    # 當最大值>3時只顯示到3，當最大值<3時顯示到最大值。
    # 同時讓使用者可以透過**kwargs(ylim=(min, max))修改預設。
    rb_max = max(rb_2.max(),rb_1.max())
    rb_min = max(rb_2.min(),rb_1.min())
    top = 3 if rb_max > 3 else rb_max
    bottom = -3 if rb_min <-3 else rb_min

    # 20231107 避免回測期間太短無法產出beta值而導致計算Y軸上下限錯誤
    if pd.isna(top) & pd.notna(bottom):
        ax.set_ylim((bottom, 3))

    elif pd.isna(bottom) & pd.notna(top):
        ax.set_ylim((-3, top))

    elif pd.isna(top) & pd.isna(bottom):
        ax.set_ylim((-2, 2))

    else:
        ax.set_ylim((bottom, top))

    rb_1.plot(color="steelblue", lw=3, alpha=0.6, ax=ax, **kwargs)
    rb_2.plot(color="grey", lw=3, alpha=0.4, ax=ax, **kwargs)
    ax.axhline(rb_1.mean(), color="steelblue", linestyle="--", lw=3)
    ax.axhline(0.0, color="black", linestyle="-", lw=2)

    ax.set_xlabel("")
    ax.legend(["6-mo", "12-mo"], loc=legend_loc, frameon=True, framealpha=0.5)

    #ax.set_ylim((-1.0, 1.0)) # 20230822 (MRC) 避免線段超出顯示範圍

    # 20231101 modify to show chinese name(yo)
    if cname:
        _name = sys._getframe(0).f_code.co_name
        ax.set_ylabel(cname[_name]['ylabel'])
        ax.set_xlabel(cname[_name]['xlabel'])
        ax.set_title(cname[_name]['title'])

    return ax


def plot_rolling_volatility(
    returns,
    factor_returns=None,
    rolling_window=APPROX_BDAYS_PER_MONTH * 6,
    legend_loc="best",
    ax=None,
    cname=None,
    **kwargs,
):
    """
    Plots the rolling volatility versus date.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    factor_returns : pd.Series, optional
        Daily noncumulative returns of the benchmark factor for which the
        benchmark rolling volatility is computed. Usually a benchmark such
        as market returns.
         - This is in the same style as returns.
    rolling_window : int, optional
        The days window over which to compute the volatility.
    legend_loc : matplotlib.loc, optional
        The location of the legend on the plot.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    y_axis_formatter = FuncFormatter(utils.two_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    rolling_vol_ts = timeseries.rolling_volatility(returns, rolling_window)
    rolling_vol_ts.plot(alpha=0.7, lw=3, color="orangered", ax=ax, **kwargs)
    if factor_returns is not None:
        rolling_vol_ts_factor = timeseries.rolling_volatility(
            factor_returns, rolling_window
        )
        rolling_vol_ts_factor.plot(
            alpha=0.7, lw=3, color="grey", ax=ax, **kwargs
        )

    ax.set_title("Rolling volatility (6-month)")
    ax.axhline(rolling_vol_ts.mean(), color="steelblue", linestyle="--", lw=3)

    ax.axhline(0.0, color="black", linestyle="-", lw=2)

    ax.set_ylabel("Volatility")
    ax.set_xlabel("")
    if factor_returns is None:
        ax.legend(
            ["Volatility", "Average volatility"],
            loc=legend_loc,
            frameon=True,
            framealpha=0.5,
        )
    else:
        ax.legend(
            ["Volatility", "Benchmark volatility", "Average volatility"],
            loc=legend_loc,
            frameon=True,
            framealpha=0.5,
        )

    # 20231101 modify to show chinese name(yo)
    if cname:
        _name = sys._getframe(0).f_code.co_name
        ax.set_ylabel(cname[_name]['ylabel'])
        ax.set_xlabel(cname[_name]['xlabel'])
        ax.set_title(cname[_name]['title'].format(round(rolling_window/APPROX_BDAYS_PER_MONTH)))

    return ax


def plot_rolling_sharpe(
    returns,
    factor_returns=None,
    rolling_window=APPROX_BDAYS_PER_MONTH * 6,
    legend_loc="best",
    ax=None,
    cname=None,
    **kwargs,
):
    """
    Plots the rolling Sharpe ratio versus date.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    factor_returns : pd.Series, optional
        Daily noncumulative returns of the benchmark factor for
        which the benchmark rolling Sharpe is computed. Usually
        a benchmark such as market returns.
         - This is in the same style as returns.
    rolling_window : int, optional
        The days window over which to compute the sharpe ratio.
    legend_loc : matplotlib.loc, optional
        The location of the legend on the plot.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    y_axis_formatter = FuncFormatter(utils.two_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    rolling_sharpe_ts = timeseries.rolling_sharpe(returns, rolling_window)
    rolling_sharpe_ts.plot(alpha=0.7, lw=3, color="orangered", ax=ax, **kwargs)

    if factor_returns is not None:
        rolling_sharpe_ts_factor = timeseries.rolling_sharpe(
            factor_returns, rolling_window
        )
        rolling_sharpe_ts_factor.plot(
            alpha=0.7, lw=3, color="grey", ax=ax, **kwargs
        )

    ax.set_title("Rolling Sharpe ratio (6-month)")
    ax.axhline(
        rolling_sharpe_ts.mean(), color="steelblue", linestyle="--", lw=3
    )
    ax.axhline(0.0, color="black", linestyle="-", lw=3)

    ax.set_ylabel("Sharpe ratio")
    ax.set_xlabel("")
    if factor_returns is None:
        ax.legend(
            ["Sharpe", "Average"], loc=legend_loc, frameon=True, framealpha=0.5
        )
    else:
        ax.legend(
            ["Sharpe", "Benchmark Sharpe", "Average"],
            loc=legend_loc,
            frameon=True,
            framealpha=0.5,
        )

    # 20231101 modify to show chinese name(yo)
    if cname:
        _name = sys._getframe(0).f_code.co_name
        ax.set_ylabel(cname[_name]['ylabel'])
        ax.set_xlabel(cname[_name]['xlabel'])
        ax.set_title(cname[_name]['title'].format(round(rolling_window/APPROX_BDAYS_PER_MONTH)))

    return ax


def plot_gross_leverage(returns, positions, ax=None, cname=None,  **kwargs):
    """
    Plots gross leverage versus date.

    Gross leverage is the sum of long and short exposure per share
    divided by net asset value.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    positions : pd.DataFrame
        Daily net position values.
         - See full explanation in create_full_tear_sheet.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()
    gl = timeseries.gross_lev(positions)
    gl.plot(lw=0.5, color="limegreen", legend=False, ax=ax, **kwargs)

    ax.axhline(gl.mean(), color="g", linestyle="--", lw=3)

    ax.set_title("Gross leverage")
    ax.set_ylabel("Gross leverage")
    ax.set_xlabel("")

    # 20231101 modify to show chinese name(yo)
    if cname:
        _name = sys._getframe(0).f_code.co_name
        ax.set_ylabel(cname[_name]['ylabel'])
        ax.set_xlabel(cname[_name]['xlabel'])
        ax.set_title(cname[_name]['title'])
    
    return ax


def plot_exposures(returns, positions, ax=None, cname=None, **kwargs):
    """
    Plots a cake chart of the long and short exposure.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    positions_alloc : pd.DataFrame
        Portfolio allocation of positions. See
        pos.get_percent_alloc.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    pos_no_cash = positions.drop("cash", axis=1)
    l_exp = pos_no_cash[pos_no_cash > 0].sum(axis=1) / positions.sum(axis=1)
    s_exp = pos_no_cash[pos_no_cash < 0].sum(axis=1) / positions.sum(axis=1)
    net_exp = pos_no_cash.sum(axis=1) / positions.sum(axis=1)

    ax.fill_between(
        l_exp.index, 0, l_exp.values, label="Long", color="red", alpha=0.5
    )                                                                         # 20230822 (by MRC) 配合台股習慣 color="green"->"red"
    ax.fill_between(
        s_exp.index, 0, s_exp.values, label="Short", color="green", alpha=0.5
    )                                                                         # 20230822 (by MRC) 配合台股習慣 color="red"->"green"
    ax.plot(
        net_exp.index,
        net_exp.values,
        label="Net",
        color="black",
        linestyle="dotted",
    )

    ax.set_xlim((returns.index[0], returns.index[-1]))
    ax.set_title("Exposure")
    ax.set_ylabel("Exposure")
    ax.legend(loc="lower left", frameon=True, framealpha=0.5)
    ax.set_xlabel("")

    # 20231101 modify to show chinese name(yo)
    if cname:
        _name = sys._getframe(0).f_code.co_name
        ax.set_ylabel(cname[_name]['ylabel'])
        ax.set_xlabel(cname[_name]['xlabel'])
        ax.set_title(cname[_name]['title'])

    return ax


def show_and_plot_top_positions(
    returns,
    positions_alloc,
    show_and_plot=2,
    hide_positions=False,
    legend_loc="real_best",
    ax=None,
    cname=None,
    **kwargs,
):
    """
    Prints and/or plots the exposures of the top 10 held positions of
    all time.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    positions_alloc : pd.DataFrame
        Portfolio allocation of positions. See pos.get_percent_alloc.
    show_and_plot : int, optional
        By default, this is 2, and both prints and plots.
        If this is 0, it will only plot; if 1, it will only print.
    hide_positions : bool, optional
        If True, will not output any symbol names.
    legend_loc : matplotlib.loc, optional
        The location of the legend on the plot.
        By default, the legend will display below the plot.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes, conditional
        The axes that were plotted on.

    """
    positions_alloc = positions_alloc.copy()
    positions_alloc.columns = positions_alloc.columns.map(utils.format_asset)

    df_top_long, df_top_short, df_top_abs = pos.get_top_long_short_abs(
        positions_alloc
    )

    if show_and_plot == 1 or show_and_plot == 2:
        utils.print_table(
            pd.DataFrame(df_top_long * 100, columns=["max"]),
            float_format="{0:.2f}%".format,
            name="Top 10 long positions of all time",
        )

        utils.print_table(
            pd.DataFrame(df_top_short * 100, columns=["max"]),
            float_format="{0:.2f}%".format,
            name="Top 10 short positions of all time",
        )

        utils.print_table(
            pd.DataFrame(df_top_abs * 100, columns=["max"]),
            float_format="{0:.2f}%".format,
            name="Top 10 positions of all time",
        )

    if show_and_plot == 0 or show_and_plot == 2:

        if ax is None:
            ax = plt.gca()

        positions_alloc[df_top_abs.index].plot(
            title="Portfolio allocation over time, only top 10 holdings",
            alpha=0.5,
            ax=ax,
            **kwargs,
        )

        # Place legend below plot, shrink plot by 20%
        if legend_loc == "real_best":
            box = ax.get_position()
            ax.set_position(
                [
                    box.x0,
                    box.y0 + box.height * 0.1,
                    box.width,
                    box.height * 0.9,
                ]
            )

            # Put a legend below current axis
            ax.legend(
                loc="upper center",
                frameon=True,
                framealpha=0.5,
                bbox_to_anchor=(0.5, -0.14),
                ncol=5,
            )
        else:
            ax.legend(loc=legend_loc)

        ax.set_xlim((returns.index[0], returns.index[-1]))
        ax.set_ylabel("Exposure by holding")

        if hide_positions:
            ax.legend_.remove()
        # 20231101 modify to show chinese name(yo)
        if cname:
            _name = sys._getframe(0).f_code.co_name
            ax.set_ylabel(cname[_name]['ylabel'])
            ax.set_xlabel(cname[_name]['xlabel'])
            ax.set_title(cname[_name]['title'])
        return ax


def plot_max_median_position_concentration(positions, ax=None, cname=None, **kwargs):
    """
    Plots the max and median of long and short position concentrations
    over the time.

    Parameters
    ----------
    positions : pd.DataFrame
        The positions that the strategy takes over time.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    alloc_summary = pos.get_max_median_position_concentration(positions)
    #colors = ["mediumblue", "steelblue", "tomato", "firebrick"] # 20230822 (by MRC) 配合台股習慣
    colors = ["firebrick", "tomato", "steelblue", "mediumblue"]  
    alloc_summary.plot(linewidth=1, color=colors, alpha=0.6, ax=ax)

    ax.legend(loc="center left", frameon=True, framealpha=0.5)
    ax.set_ylabel("Exposure")
    ax.set_title("Long/short max and median position concentration")
    
    # 20231101 modify to show chinese name(yo)
    if cname:
        _name = sys._getframe(0).f_code.co_name
        ax.set_ylabel(cname[_name]['ylabel'])
        ax.set_xlabel(cname[_name]['xlabel'])
        ax.set_title(cname[_name]['title'])

    return ax


def plot_sector_allocations(returns, sector_alloc, ax=None, cname=None, **kwargs):
    """
    Plots the sector exposures of the portfolio over time.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    sector_alloc : pd.DataFrame
        Portfolio allocation of positions. See pos.get_sector_alloc.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    sector_alloc.plot(
        title="Sector allocation over time", alpha=0.5, ax=ax, **kwargs
    )

    box = ax.get_position()
    ax.set_position(
        [box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9]
    )

    # Put a legend below current axis
    ax.legend(
        loc="upper center",
        frameon=True,
        framealpha=0.5,
        bbox_to_anchor=(0.5, -0.14),
        ncol=5,
        prop={"family" : plt.rcParams['font.sans-serif']} 
        # 20230823 (by MRC) 新增prop參數，在set_context=True下或自行設定字體下，應可取到能支援中文的字體。
    )

    ax.set_xlim((sector_alloc.index[0], sector_alloc.index[-1]))
    ax.set_ylabel("Exposure by sector")
    ax.set_xlabel("")

    # 20231101 modify to show chinese name(yo)
    if cname:
        _name = sys._getframe(0).f_code.co_name
        ax.set_ylabel(cname[_name]['ylabel'])
        ax.set_xlabel(cname[_name]['xlabel'])
        ax.set_title(cname[_name]['title'])

    return ax


def plot_return_quantiles(returns, live_start_date=None, ax=None, cname=None, **kwargs):
    """
    Creates a box plot of daily, weekly, and monthly return
    distributions.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    live_start_date : datetime, optional
        The point in time when the strategy began live trading, after
        its backtest period.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to seaborn plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    is_returns = (
        returns
        if live_start_date is None
        else returns.loc[returns.index < live_start_date]
    )
    is_weekly = ep.aggregate_returns(is_returns, "weekly")
    is_monthly = ep.aggregate_returns(is_returns, "monthly")
    sns.boxplot(
        data=[is_returns.values, is_weekly.values, is_monthly.values], # 20231124 modify by yo
        palette=["#4c72B0", "#55A868", "#CCB974"],
        ax=ax,
        **kwargs,
    )

    if live_start_date is not None:
        oos_returns = returns.loc[returns.index >= live_start_date]
        oos_weekly = ep.aggregate_returns(oos_returns, "weekly")
        oos_monthly = ep.aggregate_returns(oos_returns, "monthly")

        sns.swarmplot(
            data=[oos_returns.values, oos_weekly.values, oos_monthly.values], # 20231124 modify by yo
            ax=ax,
            color="red",
            marker="d",
            **kwargs,
        )
        red_dots = matplotlib.lines.Line2D(
            [],
            [],
            color="red",
            marker="d",
            label="Out-of-sample data",
            linestyle="",
        )
        ax.legend(handles=[red_dots], frameon=True, framealpha=0.5)
    ax.set_xticklabels(["Daily", "Weekly", "Monthly"])
    ax.set_title("Return quantiles")
    # 20231101 modify to show chinese name(yo)
    if cname:
        _name = sys._getframe(0).f_code.co_name
        ax.set_ylabel(cname[_name]['ylabel'])
        ax.set_xlabel(cname[_name]['xlabel'])
        ax.set_title(cname[_name]['title'])
        ax.set_xticklabels(['日','周','月'])
    return ax


def plot_turnover(
    returns,
    transactions,
    positions,
    turnover_denom="AGB",
    legend_loc="best",
    ax=None,
    cname=None,
    **kwargs,
):
    """
    Plots turnover vs. date.

    Turnover is the number of shares traded for a period as a fraction
    of total shares.

    Displays daily total, daily average per month, and all-time daily
    average.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    transactions : pd.DataFrame
        Prices and amounts of executed trades. One row per trade.
         - See full explanation in tears.create_full_tear_sheet.
    positions : pd.DataFrame
        Daily net position values.
         - See full explanation in tears.create_full_tear_sheet.
    turnover_denom : str, optional
        Either AGB or portfolio_value, default AGB.
        - See full explanation in txn.get_turnover.
    legend_loc : matplotlib.loc, optional
        The location of the legend on the plot.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    y_axis_formatter = FuncFormatter(utils.two_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    df_turnover = txn.get_turnover(positions, transactions, turnover_denom)
    df_turnover_by_month = df_turnover.resample("M").mean()
    df_turnover.plot(color="steelblue", alpha=1.0, lw=0.5, ax=ax, **kwargs)
    df_turnover_by_month.plot(
        color="orangered", alpha=0.5, lw=2, ax=ax, **kwargs
    )
    ax.axhline(
        df_turnover.mean(), color="steelblue", linestyle="--", lw=3, alpha=1.0
    )
    ax.legend(
        [
            "Daily turnover",
            "Average daily turnover, by month",
            "Average daily turnover, net",
        ],
        loc=legend_loc,
        frameon=True,
        framealpha=0.5,
    )
    ax.set_title("Daily turnover")
    ax.set_xlim((returns.index[0], returns.index[-1]))
    ax.set_ylim((0, 2))
    ax.set_ylabel("Turnover")
    ax.set_xlabel("")

    # 20231010 modify to show chinese name(yo)
    if cname:
        _name = sys._getframe(0).f_code.co_name
        ax.set_ylabel(cname[_name]['ylabel'])
        ax.set_xlabel(cname[_name]['xlabel'])
        ax.set_title(cname[_name]['title'])

    return ax


def plot_slippage_sweep(
    returns,
    positions,
    transactions,
    slippage_params=(3, 8, 10, 12, 15, 20, 50),
    ax=None,
    cname=None,
    **kwargs,
):
    """
    Plots equity curves at different per-dollar slippage assumptions.

    Parameters
    ----------
    returns : pd.Series
        Timeseries of portfolio returns to be adjusted for various
        degrees of slippage.
    positions : pd.DataFrame
        Daily net position values.
         - See full explanation in tears.create_full_tear_sheet.
    transactions : pd.DataFrame
        Prices and amounts of executed trades. One row per trade.
         - See full explanation in tears.create_full_tear_sheet.
    slippage_params: tuple
        Slippage pameters to apply to the return time series (in
        basis points).
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to seaborn plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    slippage_sweep = pd.DataFrame()
    for bps in slippage_params:
        adj_returns = txn.adjust_returns_for_slippage(
            returns, positions, transactions, bps
        )
        label = str(bps) + " bps"
        slippage_sweep[label] = ep.cum_returns(adj_returns, 1)

    slippage_sweep.plot(alpha=1.0, lw=0.5, ax=ax)

    ax.set_title("Cumulative returns given additional per-dollar slippage")
    ax.set_ylabel("")

    ax.legend(loc="center left", frameon=True, framealpha=0.5)
    # 20231101 modify to show chinese name(yo)
    if cname:
        _name = sys._getframe(0).f_code.co_name
        ax.set_ylabel(cname[_name]['ylabel'])
        ax.set_xlabel(cname[_name]['xlabel'])
        ax.set_title(cname[_name]['title'])
    return ax


def plot_slippage_sensitivity(
    returns, positions, transactions, ax=None, cname=None, **kwargs
):
    """
    Plots curve relating per-dollar slippage to average annual returns.

    Parameters
    ----------
    returns : pd.Series
        Timeseries of portfolio returns to be adjusted for various
        degrees of slippage.
    positions : pd.DataFrame
        Daily net position values.
         - See full explanation in tears.create_full_tear_sheet.
    transactions : pd.DataFrame
        Prices and amounts of executed trades. One row per trade.
         - See full explanation in tears.create_full_tear_sheet.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to seaborn plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    avg_returns_given_slippage = pd.Series()
    for bps in range(1, 100):
        adj_returns = txn.adjust_returns_for_slippage(
            returns, positions, transactions, bps
        )
        avg_returns = ep.annual_return(adj_returns)
        avg_returns_given_slippage.loc[bps] = avg_returns

    avg_returns_given_slippage.plot(alpha=1.0, lw=2, ax=ax)

    ax.set_title("Average annual returns given additional per-dollar slippage")
    ax.set_xticks(np.arange(0, 100, 10))
    ax.set_ylabel("Average annual return")
    ax.set_xlabel("Per-dollar slippage (bps)")

    # 20231010 modify to show chinese name(yo)
    if cname:
        _name = sys._getframe(0).f_code.co_name
        ax.set_ylabel(cname[_name]['ylabel'])
        ax.set_xlabel(cname[_name]['xlabel'])
        ax.set_title(cname[_name]['title'])

    return ax


def plot_capacity_sweep(
    returns,
    transactions,
    market_data,
    bt_starting_capital,
    min_pv=100000,
    max_pv=300000000,
    step_size=1000000,
    cname=None,
    ax=None,
):
    txn_daily_w_bar = capacity.daily_txns_with_bar_data(
        transactions, market_data
    )

    captial_base_sweep = pd.Series()
    for start_pv in range(min_pv, max_pv, step_size):
        adj_ret = capacity.apply_slippage_penalty(
            returns, txn_daily_w_bar, start_pv, bt_starting_capital
        )
        sharpe = ep.sharpe_ratio(adj_ret)
        if sharpe < -1:
            break
        captial_base_sweep.loc[start_pv] = sharpe
    captial_base_sweep.index = captial_base_sweep.index / MM_DISPLAY_UNIT

    if ax is None:
        ax = plt.gca()

    captial_base_sweep.plot(ax=ax)
    ax.set_xlabel("Capital base ($mm)")
    ax.set_ylabel("Sharpe ratio")
    ax.set_title("Capital base performance sweep")
    # 20231101 modify to show chinese name(yo)
    if cname:
        _name = sys._getframe(0).f_code.co_name
        ax.set_ylabel(cname[_name]['ylabel'])
        ax.set_xlabel(cname[_name]['xlabel'])
        ax.set_title(cname[_name]['title'])

    return ax


def plot_daily_turnover_hist(
    transactions, positions, turnover_denom="AGB", ax=None, cname=None, **kwargs
):
    """
    Plots a histogram of daily turnover rates.

    Parameters
    ----------
    transactions : pd.DataFrame
        Prices and amounts of executed trades. One row per trade.
         - See full explanation in tears.create_full_tear_sheet.
    positions : pd.DataFrame
        Daily net position values.
         - See full explanation in tears.create_full_tear_sheet.
    turnover_denom : str, optional
        Either AGB or portfolio_value, default AGB.
        - See full explanation in txn.get_turnover.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to seaborn plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()
    turnover = txn.get_turnover(positions, transactions, turnover_denom)
    sns.histplot(turnover, ax=ax, **kwargs)
    ax.set_title("Distribution of daily turnover rates")
    ax.set_xlabel("Turnover rate")

    # 20231101 modify to show chinese name(yo)
    if cname:
        _name = sys._getframe(0).f_code.co_name
        ax.set_ylabel(cname[_name]['ylabel'])
        ax.set_xlabel(cname[_name]['xlabel'])
        ax.set_title(cname[_name]['title'])

    return ax


def plot_daily_volume(returns, transactions, ax=None, cname=None, **kwargs):
    """
    Plots trading volume per day vs. date.

    Also displays all-time daily average.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    transactions : pd.DataFrame
        Prices and amounts of executed trades. One row per trade.
         - See full explanation in tears.create_full_tear_sheet.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()
    daily_txn = txn.get_txn_vol(transactions)
    daily_txn.txn_shares.plot(alpha=1.0, lw=0.5, ax=ax, **kwargs)
    ax.axhline(
        daily_txn.txn_shares.mean(),
        color="steelblue",
        linestyle="--",
        lw=3,
        alpha=1.0,
    )
    ax.set_title("Daily trading volume")
    ax.set_xlim((returns.index[0], returns.index[-1]))
    ax.set_ylabel("Amount of shares traded")
    ax.set_xlabel("")

    # 20231101 modify to show chinese name(yo)
    if cname:
        _name = sys._getframe(0).f_code.co_name
        ax.set_ylabel(cname[_name]['ylabel'])
        ax.set_xlabel(cname[_name]['xlabel'])
        ax.set_title(cname[_name]['title'])
    
    return ax


def plot_txn_time_hist(
    transactions, bin_minutes=5, tz="Asia/Taipei", ax=None, cname=None, **kwargs   # 20230801 (by MRC)  "America/New_York"->Asia/Taipei 否則full tear sheet圖形出不來
):
    """
    Plots a histogram of transaction times, binning the times into
    buckets of a given duration.

    Parameters
    ----------
    transactions : pd.DataFrame
        Prices and amounts of executed trades. One row per trade.
         - See full explanation in tears.create_full_tear_sheet.
    bin_minutes : float, optional
        Sizes of the bins in minutes, defaults to 5 minutes.
    tz : str, optional
        Time zone to plot against. Note that if the specified
        zone does not apply daylight savings, the distribution
        may be partially offset.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    txn_time = transactions.copy()

    txn_time.index = txn_time.index.tz_convert(pytz.timezone(tz))
    txn_time.index = txn_time.index.map(lambda x: x.hour * 60 + x.minute)
    txn_time["trade_value"] = (txn_time.amount * txn_time.price).abs()
    txn_time = txn_time.groupby(level=0).sum( numeric_only = True ).reindex(index=range(570, 961))
    txn_time.index = (txn_time.index / bin_minutes).astype(int) * bin_minutes
    txn_time = txn_time.groupby(level=0).sum( numeric_only = True )

    txn_time["time_str"] = txn_time.index.map(
        lambda x: str(datetime.time(int(x / 60), x % 60))[:-3]
    )

    trade_value_sum = txn_time.trade_value.sum()
    txn_time.trade_value = txn_time.trade_value.fillna(0) / trade_value_sum

    ax.bar(txn_time.index, txn_time.trade_value, width=bin_minutes, **kwargs)

    ax.set_xlim(570, 960)
    ax.set_xticks(txn_time.index[:: int(30 / bin_minutes)])
    ax.set_xticklabels(txn_time.time_str[:: int(30 / bin_minutes)])
    ax.set_title("Transaction time distribution")
    ax.set_ylabel("Proportion")
    ax.set_xlabel("")

    # 20231010 modify to show chinese name(yo)
    if cname:
        _name = sys._getframe(0).f_code.co_name
        ax.set_ylabel(cname[_name]['ylabel'])
        ax.set_xlabel(cname[_name]['xlabel'])
        ax.set_title(cname[_name]['title'])

    return ax


def show_worst_drawdown_periods(returns, top=5):
    """
    Prints information about the worst drawdown periods.

    Prints peak dates, valley dates, recovery dates, and net
    drawdowns.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    top : int, optional
        Amount of top drawdowns periods to plot (default 5).
    """

    drawdown_df = timeseries.gen_drawdown_table(returns, top=top)
    utils.print_table(
        drawdown_df.sort_values("Net drawdown in %", ascending=False),
        name="Worst drawdown periods",
        float_format="{0:.2f}".format,
    )


def plot_monthly_returns_timeseries(returns, ax=None, cname=None, **kwargs):
    """
    Plots monthly returns as a timeseries.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to seaborn plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    def cumulate_returns(x):
        return ep.cum_returns(x)[-1]

    if ax is None:
        ax = plt.gca()

    monthly_rets = returns.resample("M").apply(lambda x: cumulate_returns(x))
    monthly_rets = monthly_rets.to_period()

    sns.barplot(x=monthly_rets.index, y=monthly_rets.values, color="steelblue")

    _, labels = plt.xticks()
    plt.setp(labels, rotation=90)

    # only show x-labels on year boundary
    xticks_coord = []
    xticks_label = []
    count = 0
    for i in monthly_rets.index:
        if i.month == 1:
            xticks_label.append(i)
            xticks_coord.append(count)
            # plot yearly boundary line
            ax.axvline(count, color="gray", ls="--", alpha=0.3)

        count += 1

    ax.axhline(0.0, color="darkgray", ls="-")
    ax.set_xticks(xticks_coord)
    ax.set_xticklabels(xticks_label)
    # 20231010 modify to show chinese name(yo)
    if cname:
        _name = sys._getframe(0).f_code.co_name
        ax.set_ylabel(cname[_name]['ylabel'])
        ax.set_xlabel(cname[_name]['xlabel'])
        ax.set_title(cname[_name]['title'])
    return ax


def plot_round_trip_lifetimes(round_trips, disp_amount=16, lsize=18, ax=None, cname=None):
    """
    Plots timespans and directions of a sample of round trip trades.

    Parameters
    ----------
    round_trips : pd.DataFrame
        DataFrame with one row per round trip trade.
        - See full explanation in round_trips.extract_round_trips
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.subplot()

    symbols_sample = round_trips.symbol.unique()
    np.random.seed(1)
    sample = np.random.choice(
        round_trips.symbol.unique(),
        replace=False,
        size=min(disp_amount, len(symbols_sample)),
    )
    sample_round_trips = round_trips[round_trips.symbol.isin(sample)]

    symbol_idx = pd.Series(np.arange(len(sample)), index=sample)

    for symbol, sym_round_trips in sample_round_trips.groupby("symbol"):
        for _, row in sym_round_trips.iterrows():
            c = "b" if row.long else "r"
            y_ix = symbol_idx[symbol] + 0.05
            ax.plot(
                [row["open_dt"], row["close_dt"]],
                [y_ix, y_ix],
                color=c,
                linewidth=lsize,
                solid_capstyle="butt",
            )

    ax.set_yticks(range(len(sample)))
    ax.set_yticklabels([utils.format_asset(s) for s in sample])

    ax.set_ylim((-0.5, min(len(sample), disp_amount) - 0.5))
    blue = patches.Rectangle([0, 0], 1, 1, color="b", label="Long")
    red = patches.Rectangle([0, 0], 1, 1, color="r", label="Short")
    leg = ax.legend(
        handles=[blue, red], loc="lower left", frameon=True, framealpha=0.5
    )
    leg.get_frame().set_edgecolor("black")
    ax.grid(False)
    
    # 20231124 modify to show chinese name(yo)
    if cname:
        _name = sys._getframe(0).f_code.co_name
        ax.set_ylabel(cname[_name]['ylabel'])
        ax.set_xlabel(cname[_name]['xlabel'])
        ax.set_title(cname[_name]['title'])
        
    return ax

def show_profit_attribution(round_trips):
    """
    Prints the share of total PnL contributed by each
    traded name.

    Parameters
    ----------
    round_trips : pd.DataFrame
        DataFrame with one row per round trip trade.
        - See full explanation in round_trips.extract_round_trips
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    total_pnl = round_trips["pnl"].sum()
    pnl_attribution = round_trips.groupby("symbol")["pnl"].sum() / total_pnl
    pnl_attribution.name = ""

    pnl_attribution.index = pnl_attribution.index.map(utils.format_asset)
    utils.print_table(
        pnl_attribution.sort_values(
            inplace=False,
            ascending=False,
        ),
        name="Profitability (PnL / PnL total) per name",
        float_format="{:.2%}".format,
    )


def plot_prob_profit_trade(round_trips, ax=None, cname=None):
    """
    Plots a probability distribution for the event of making
    a profitable trade.

    Parameters
    ----------
    round_trips : pd.DataFrame
        DataFrame with one row per round trip trade.
        - See full explanation in round_trips.extract_round_trips
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    x = np.linspace(0, 1.0, 500)

    round_trips["profitable"] = round_trips.pnl > 0

    dist = sp.stats.beta(
        round_trips.profitable.sum(), (~round_trips.profitable).sum()
    )
    y = dist.pdf(x)
    lower_perc = dist.ppf(0.025)
    upper_perc = dist.ppf(0.975)

    lower_plot = dist.ppf(0.001)
    upper_plot = dist.ppf(0.999)

    if ax is None:
        ax = plt.subplot()

    ax.plot(x, y)
    ax.axvline(lower_perc, color="0.5")
    ax.axvline(upper_perc, color="0.5")

    ax.set_xlabel("Probability of making a profitable decision")
    ax.set_ylabel("Belief")
    ax.set_xlim(lower_plot, upper_plot)
    ax.set_ylim((0, y.max() + 1.0))

    # 20231124 modify to show chinese name(yo)
    if cname:
        _name = sys._getframe(0).f_code.co_name
        ax.set_ylabel(cname[_name]['ylabel'])
        ax.set_xlabel(cname[_name]['xlabel'])
        ax.set_title(cname[_name]['title'])
        
    return ax

def plot_cones(
    name,
    bounds,
    oos_returns,
    num_samples=1000,
    ax=None,
    cone_std=(1.0, 1.5, 2.0),
    random_seed=None,
    num_strikes=3,
):
    """
    Plots the upper and lower bounds of an n standard deviation
    cone of forecasted cumulative returns. Redraws a new cone when
    cumulative returns fall outside of last cone drawn.

    Parameters
    ----------
    name : str
        Account name to be used as figure title.
    bounds : pandas.core.frame.DataFrame
        Contains upper and lower cone boundaries. Column names are
        strings corresponding to the number of standard devations
        above (positive) or below (negative) the projected mean
        cumulative returns.
    oos_returns : pandas.core.frame.DataFrame
        Non-cumulative out-of-sample returns.
    num_samples : int
        Number of samples to draw from the in-sample daily returns.
        Each sample will be an array with length num_days.
        A higher number of samples will generate a more accurate
        bootstrap cone.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    cone_std : list of int/float
        Number of standard devations to use in the boundaries of
        the cone. If multiple values are passed, cone bounds will
        be generated for each value.
    random_seed : int
        Seed for the pseudorandom number generator used by the pandas
        sample method.
    num_strikes : int
        Upper limit for number of cones drawn. Can be anything from 0 to 3.

    Returns
    -------
    Returns are either an ax or fig option, but not both. If a
    matplotlib.Axes instance is passed in as ax, then it will be modified
    and returned. This allows for users to plot interactively in jupyter
    notebook. When no ax object is passed in, a matplotlib.figure instance
    is generated and returned. This figure can then be used to save
    the plot as an image without viewing it.

    ax : matplotlib.Axes
        The axes that were plotted on.
    fig : matplotlib.figure
        The figure instance which contains all the plot elements.
    """

    if ax is None:
        fig = figure.Figure(figsize=(10, 8))
        FigureCanvasAgg(fig)
        axes = fig.add_subplot(111)
    else:
        axes = ax

    returns = ep.cum_returns(oos_returns, starting_value=1.0)
    bounds_tmp = bounds.copy()
    returns_tmp = returns.copy()
    cone_start = returns.index[0]
    colors = ["green", "orange", "orangered", "darkred"]

    for c in range(num_strikes + 1):
        if c > 0:
            tmp = returns.loc[cone_start:]
            bounds_tmp = bounds_tmp.iloc[0 : len(tmp)]
            bounds_tmp = bounds_tmp.set_index(tmp.index)
            crossing = tmp < bounds_tmp[float(-2.0)].iloc[: len(tmp)]
            if crossing.sum() <= 0:
                break
            cone_start = crossing.loc[crossing].index[0]
            returns_tmp = returns.loc[cone_start:]
            bounds_tmp = bounds - (1 - returns.loc[cone_start])
        for std in cone_std:
            x = returns_tmp.index
            y1 = bounds_tmp[float(std)].iloc[: len(returns_tmp)]
            y2 = bounds_tmp[float(-std)].iloc[: len(returns_tmp)]
            axes.fill_between(x, y1, y2, color=colors[c], alpha=0.5)

    # Plot returns line graph
    label = "Cumulative returns = {:.2f}%".format((returns.iloc[-1] - 1) * 100)
    axes.plot(
        returns.index, returns.values, color="black", lw=3.0, label=label
    )

    if name is not None:
        axes.set_title(name)
    axes.axhline(1, color="black", alpha=0.2)
    axes.legend(frameon=True, framealpha=0.5)

    if ax is None:
        return fig
    else:
        return axes

def plot_mae_mfe(st:str, et:str, stock:str):
    import matplotlib as mat
    mat.rcParams['axes.unicode_minus'] = False

    import matplotlib
    matplotlib.style.use('ggplot')
    """20230920 用來呈現股票在特定時間內的MAE和MFE，紅線為有利方向，綠線為不利方向　Terry"""
    '''=================================================載入資料區================================================='''
    from zipline.data import bundles
    bundle = bundles.load('tquant')
    adj_price = get_prices(start_date=pd.Timestamp(st, tz='utc'), 
                        end_date=pd.Timestamp(et, tz='utc'),
                        field='close',
                        assets=[bundle.asset_finder.lookup_symbol(stock, None)])
    stock = bundle.asset_finder.lookup_symbol(stock, None)
    mae = np.where(adj_price.loc[st: et, stock].pct_change().add(1).cumprod().fillna(1).gt(1) & adj_price.loc[st: et, stock].diff().fillna(0).gt(0), 1,
            np.where(adj_price.loc[st: et, stock].pct_change().add(1).cumprod().fillna(1).lt(1) & adj_price.loc[st: et, stock].diff().fillna(0).lt(0), -1, 0))

    y = pd.DataFrame(data=adj_price.loc[st: et, stock].values, columns=['price'], index=adj_price.loc[st: et, stock].index)

    y['mae'] = mae

    fig, ax = plt.subplots(figsize=(12, 8))

    '''=================================================畫線區================================================='''
    ax = y.price.plot(label='調整後收盤價')

    for index, row in y.iterrows():
        if row.mae > 0:
            ax.axvline(x=index, ymax=(row.price - ax.get_ylim()[0]) / -(ax.get_ylim()[0] - ax.get_ylim()[1]), 
                    ymin=(y.iloc[0].price - ax.get_ylim()[0]) / -(ax.get_ylim()[0] - ax.get_ylim()[1]), color='red', alpha=0.5)

        elif row.mae < 0:
            ax.axvline(x=index, ymax=(y.iloc[0].price - ax.get_ylim()[0]) / -(ax.get_ylim()[0] - ax.get_ylim()[1]), 
                    ymin=(row.price - ax.get_ylim()[0]) / -(ax.get_ylim()[0] - ax.get_ylim()[1]), color='green', alpha=0.5)
            
    ax.axvline(x=index, ymax=(row.price - ax.get_ylim()[0]) / -(ax.get_ylim()[0] - ax.get_ylim()[1]), 
                    ymin=(y.iloc[0].price - ax.get_ylim()[0]) / -(ax.get_ylim()[0] - ax.get_ylim()[1]), color='red', alpha=0.3, label='有利方向')

    ax.axvline(x=index, ymax=(y.iloc[0].price - ax.get_ylim()[0]) / -(ax.get_ylim()[0] - ax.get_ylim()[1]), 
                    ymin=(row.price - ax.get_ylim()[0]) / -(ax.get_ylim()[0] - ax.get_ylim()[1]), color='green', alpha=0.3, label='不利方向')

    '''=================================================畫點區================================================='''
    ax.scatter(x = [y.price.idxmax()], y=[y.price.max()], marker='*', s=100, color=['red'], label='MFE')

    #標記MFE時間與數值
    ax.annotate(text=f"{str(y.price.idxmax().date())}:{adj_price.max().div(adj_price.iloc[0]).sub(1).mul(100).values[0]:.2f}%", xy=[y.price.idxmax(), y.price.max()], xytext=(y.price.idxmax(), y.price.max()*1.0005))

    ax.scatter(x = [y.price.idxmin()], y=[y.price.min()], marker='*', s=100, color=['green'], label='MAE')

    #標記MAE時間與數值
    ax.annotate(text=f"{str(y.price.idxmin().date())}:{adj_price.min().div(adj_price.iloc[0]).sub(1).mul(100).values[0]:.2f}%", xy=[y.price.idxmin(), y.price.min()], xytext=(y.price.idxmin(), y.price.min()*0.9995))

    ax.axhline(y.price[0], color='black', linestyle='--')

    ax.set_title(f'{stock.symbol}在{st}~{et}期間 \n 有利方向(FE)與不利方向(AE)')

    ax.grid(linestyle='--', alpha=0.5)

    ax.legend()

    plt.show()

def ret_count(mae:pd.DataFrame, ax:None):    
    import matplotlib as mat
    mat.rcParams['axes.unicode_minus'] = False

    import matplotlib
    matplotlib.style.use('ggplot')
    """" 20230928 視覺化報酬率累積分布與計算勝率 Terry"""
    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize=(6,6))

    plus_bins = int(len(mae[mae['returns_tot'].gt(0)])/3) if len(mae[mae['returns_tot'].gt(0)]) > 3 else len(mae[mae['returns_tot'].gt(0)])
    minus_bins = int(len(mae[mae['returns_tot'].le(0)])/3) if len(mae[mae['returns_tot'].le(0)]) > 3 else len(mae[mae['returns_tot'].le(0)])

    ax.hist(x=mae[mae['returns_tot'].gt(0)]['returns_tot'], bins=plus_bins, color='r', label='Profit', width=2)
    ax.hist(x=mae[mae['returns_tot'].le(0)]['returns_tot'], bins=minus_bins, color='g', label='Loss', width=2)

    min_value = mae['returns_tot'].min()
    max_value = mae['returns_tot'].max()

    mean = mae['returns_tot'].mean()

    plt.xlim(min_value*1.1, max_value*1.1)

    ax.axvline(x = mean, color='black', linestyle='--', alpha=0.8, label=3)

    ax.set_xlabel('Returns(%)')
    ax.set_ylabel('Count')
    ax.set_title(f'Returns Distribution(Win ratio:{len(mae[mae["returns_tot"].gt(0)])/len(mae)*100:.2f}%)', fontsize=15)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=[handles[0], handles[1], handles[2]], labels=['Profit',  'Loss', f"Avg.R:{mae['returns_tot'].mean():.2f}%"])

def mae_returns(mae:pd.DataFrame, ax:None):    
    import matplotlib as mat
    mat.rcParams['axes.unicode_minus'] = False

    import matplotlib
    matplotlib.style.use('ggplot')
    """" 20230928 視覺化報酬率與MAE的分布圖 Terry"""
    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize=(6,6))

    scatter_size = 10 if mae.returns_tot.abs().max() <= 100 else 1
    ax.scatter(x=mae[mae.returns_tot.gt(0)].returns_tot, 
                y=mae[mae.returns_tot.gt(0)].MAE.abs(), 
                s=mae[mae.returns_tot.gt(0)].returns_tot.mul(scatter_size).abs(),
                color='r', alpha=0.9, edgecolor='white', marker='*', zorder=2, label='Profit')
    
    ax.scatter(x=mae[mae.returns_tot.le(0)].returns_tot, 
                y=mae[mae.returns_tot.le(0)].MAE.abs(), 
                s=mae[mae.returns_tot.le(0)].returns_tot.mul(scatter_size).abs(),
                color='g', alpha=0.9, edgecolor='white', marker='X', zorder=2, label='Loss')


    ax.axhline(mae[mae.returns_tot.le(0)].MAE.abs().quantile(0.75), linestyle='--', color='g', alpha=0.5, label='3')

    ax.axhline(mae[mae.returns_tot.gt(0)].MAE.abs().quantile(0.75), linestyle='--', color='r', alpha=0.5, label='4')


    plt.xlim(mae.returns_tot.min()*1.1, mae.returns_tot.max()*1.1)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=[handles[0], handles[3], handles[1], handles[2]], labels=['Profit',  
                                                                                f"P.Q3:{mae[mae.returns_tot.gt(0)].MAE.abs().quantile(0.75):.2f}%",
                                                                                'Loss',
                                                                                f"L.Q3:{mae[mae.returns_tot.le(0)].MAE.abs().quantile(0.75):.2f}%" 
                                                                                ])

    ax.set_title('MAE & Returns Distribution', fontsize=15)
    ax.set_xlabel('Returns(%)')
    ax.set_ylabel('MAE(%)')

def mdd_profit(mae:pd.DataFrame, ax:None):    
    import matplotlib as mat
    mat.rcParams['axes.unicode_minus'] = False

    import matplotlib
    matplotlib.style.use('ggplot')
    """" 20230928 視覺化報酬率與MDD的分布圖 Terry"""
    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize=(6,6))
        
    scatter_size = 10 if mae.returns_tot.abs().max() <= 100 else 1
    ax.scatter(x=mae[mae.returns_tot.gt(0)].MDD.abs(), 
                y=mae[mae.returns_tot.gt(0)].returns_tot, 
                s=mae[mae.returns_tot.gt(0)].returns_tot.mul(scatter_size).abs(),
                color='r', alpha=0.9, edgecolor='white', marker='*', zorder=2, label='Profit')
    
    ax.scatter(x=mae[mae.returns_tot.le(0)].MDD.abs(), 
                y=mae[mae.returns_tot.le(0)].returns_tot.abs(), 
                s=mae[mae.returns_tot.le(0)].returns_tot.mul(scatter_size).abs(),
                color='g', alpha=0.9, edgecolor='white', marker='X', zorder=2, label='Loss')

    ax.set_xlabel('MDD(%)')

    ax.set_ylabel('Returns(%)')

    xlimit = int(abs(mae.returns_tot.max()) if abs(mae.returns_tot.max()) > abs(mae.MDD.min()) else abs(mae.MDD.min())*1.2)

    plt.xlim(0, xlimit if xlimit <= 100 else 100)
    plt.ylim(0, xlimit*1.1)

    ax.plot([0, xlimit if xlimit <= 100 else 100], [0, xlimit if xlimit <= 100 else 100], color = 'black', linewidth = 2, linestyle='--', zorder=-1)

    ax.set_title('MDD & Returns Distribution')

    ax.legend()

def mae_gmfe(mae:pd.DataFrame, ax:None):    
    import matplotlib as mat
    mat.rcParams['axes.unicode_minus'] = False

    import matplotlib
    matplotlib.style.use('ggplot')
    """" 20230928 視覺化MAE與GMFE的分布圖 Terry"""
    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize=(6,6))
        
    scatter_size = 10 if mae.returns_tot.abs().max() <= 100 else 1
    ax.scatter(x=mae[mae.returns_tot.gt(0)].MAE.abs(), 
            y=mae[mae.returns_tot.gt(0)].GMFE, 
            s=mae[mae.returns_tot.gt(0)].returns_tot.mul(scatter_size).abs(),
            color='r', alpha=0.9, edgecolor='white', marker='*', zorder=2, label='Profit')
        
    ax.scatter(x=mae[mae.returns_tot.le(0)].MAE.abs(), 
                y=mae[mae.returns_tot.le(0)].GMFE, 
                s=mae[mae.returns_tot.le(0)].returns_tot.mul(scatter_size).abs(),
                color='g', alpha=0.9, edgecolor='white', marker='X', zorder=2, label='Loss')



    plt.xlim(mae.MAE.abs().min()*1.1, mae.MAE.abs().max()*1.1)

    ax.legend()
    ax.set_title('MAE & GMFE Distribution', fontsize=15)
    ax.set_xlabel('MAE(%)')
    ax.set_ylabel('GMFE(%)')

def mae_bmfe(mae:pd.DataFrame, ax:None):    
    import matplotlib as mat
    mat.rcParams['axes.unicode_minus'] = False

    import matplotlib
    matplotlib.style.use('ggplot')
    """" 20230928 視覺化MAE與BMFE的分布圖 Terry"""
    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize=(6,6))
        
    scatter_size = 10 if mae.returns_tot.abs().max() <= 100 else 1
    ax.scatter(x=mae[mae.returns_tot.gt(0)].MAE.abs(), 
                y=mae[mae.returns_tot.gt(0)].BMFE, 
                s=mae[mae.returns_tot.gt(0)].returns_tot.mul(scatter_size).abs(),
                color='r', alpha=0.9, edgecolor='white', marker='*', zorder=2, label='Profit')
    
    ax.scatter(x=mae[mae.returns_tot.le(0)].MAE.abs(), 
            y=mae[mae.returns_tot.le(0)].BMFE, 
            s=mae[mae.returns_tot.le(0)].returns_tot.mul(scatter_size).abs(),
            color='g', alpha=0.9, edgecolor='white', marker='X', zorder=2, label='Loss')

    plt.xlim(mae.MAE.abs().min()*1.1, mae.MAE.abs().max()*1.1)

    ax.legend()
    ax.set_title('MAE & BMFE Distribution', fontsize=15)
    ax.set_xlabel('MAE(%)')
    ax.set_ylabel('BMFE(%)')

def plot_edge(mae:pd.DataFrame, ax:None):
    import matplotlib as mat
    mat.rcParams['axes.unicode_minus'] = False

    import matplotlib
    matplotlib.style.use('ggplot')

    """" 20230928 視覺化優勢比率趨勢圖 Terry"""
    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize=(6,6))
        
    x = mae['prices'].apply(lambda x: pd.Series(x)).T

    x.cummax().div(x.iloc[0]).sum(axis=1).div(x.cummin().div(x.iloc[0]).sum(axis=1)).plot(
        title='Edge Ratio Trend',
        xlabel='Days',
        ylabel='Edge ratio',
        color='deepskyblue',
        ax=ax)

    ax.set_xlim(1, len(x))
    ax.set_xticks(labels=[i for i in range(1, len(x)+1, int(len(x)/10))], ticks=[i for i in range(1, len(x)+1, int(len(x)/10))])

def dist_mae_gmfe_bmfe(mae:pd.DataFrame, ax1:None, ax2:None, ax3:None):    
    """" 20230928 視覺化MAE, GMFE, BMFE等累積分布圖 Terry"""
    import matplotlib as mat
    mat.rcParams['axes.unicode_minus'] = False

    import matplotlib
    matplotlib.style.use('ggplot')

    if ax1 == None or ax2 == None or ax3 == None:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    mae['profit'] = np.where(mae['returns_tot'] > 0, 'Profit', 'Loss')
    mae_q3_p = mae.query('profit=="Profit"')['MAE'].abs().quantile(0.75)
    mae_q3_l = mae.query('profit=="Loss"')['MAE'].abs().quantile(0.75)
    gmfe_q3_p = mae.query('profit=="Profit"')['GMFE'].abs().quantile(0.75)
    gmfe_q3_l = mae.query('profit=="Loss"')['GMFE'].abs().quantile(0.75)
    bmfe_q3_p = mae.query('profit=="Profit"')['BMFE'].abs().quantile(0.75)
    bmfe_q3_l = mae.query('profit=="Loss"')['BMFE'].abs().quantile(0.75)

    sns.histplot(mae, x=mae['MAE'].abs(), hue='profit', palette={'Profit': 'r', 'Loss': 'g'}, multiple='dodge', ax=ax1, label='0', alpha=0.8)
    ax1.axvline(x=mae_q3_p, linestyle='--', color='coral', linewidth=2, label=[3])
    ax1.axvline(x=mae_q3_l, linestyle='--', color='dodgerblue', linewidth=2, label=[4])

    ax1.set_xlim([mae['MAE'].abs().min()*1.1, mae['MAE'].abs().max()*1.1])
    ax1.set_xlabel('MAE(%)')
    ax1.set_title('MAE(%) Distribution')

    #===============================================================================================================================================================
    sns.histplot(mae, x=mae['GMFE'].abs(), hue='profit', palette={'Profit': 'r', 'Loss': 'g'}, multiple='dodge', ax=ax2, label='0', alpha=0.8)
    ax2.axvline(x=gmfe_q3_p, linestyle='--', color='coral', linewidth=2, label=[3])
    ax2.axvline(x=gmfe_q3_l, linestyle='--', color='dodgerblue', linewidth=2, label=[4])

    ax2.set_xlim([mae['GMFE'].abs().min()*1.1, mae['GMFE'].abs().max()*1.1])
    ax2.set_xlabel('GMFE(%)')
    ax2.set_title('GMFE(%) Distribution')

    #===============================================================================================================================================================
    sns.histplot(mae, x=mae['BMFE'].abs(), hue='profit', palette={'Profit': 'r', 'Loss': 'g'}, multiple='dodge', ax=ax3, label='0', alpha=0.8)
    ax3.axvline(x=bmfe_q3_p, linestyle='--', color='coral', linewidth=2, label=[3])
    ax3.axvline(x=bmfe_q3_l, linestyle='--', color='dodgerblue', linewidth=2, label=[4])

    ax3.set_xlim([mae['BMFE'].abs().min()*1.1, mae['BMFE'].abs().max()*1.1])
    ax3.set_xlabel('BMFE(%)')
    ax3.set_title('BMFE(%) Distribution')

    #===============================================================================================================================================================
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=[handles[2], handles[0], handles[3], handles[1]], labels=['Profit',  f"P.Q3:{mae_q3_p:.2f}%", 'Loss', f"L.Q3:{mae_q3_l:.2f}%"])

    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles=[handles[2], handles[0], handles[3], handles[1]], labels=['Profit', f"P.Q3:{gmfe_q3_p:.2f}%", 'Loss', f"L.Q3:{gmfe_q3_l:.2f}%"])

    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(handles=[handles[2], handles[0], handles[3], handles[1]], labels=['Profit', f"P.Q3:{bmfe_q3_p:.2f}%", 'Loss', f"L.Q3:{bmfe_q3_l:.2f}%"])

def plot_all_mae(mae):  
    """" 20230928 呼叫前述所有圖表一次呈現 Terry"""
    import matplotlib as mat
    mat.rcParams['axes.unicode_minus'] = False

    import matplotlib
    matplotlib.style.use('ggplot')

    fig, ax = plt.subplots(3, 3, figsize=(12, 12))

    ret_count(mae, ax=ax[0, 0])
    plot_edge(mae, ax=ax[0, 1])
    mae_returns(mae, ax=ax[0, 2])
    mae_gmfe(mae, ax=ax[1, 0])
    mae_bmfe(mae, ax=ax[1, 1])
    mdd_profit(mae, ax=ax[1, 2])
    dist_mae_gmfe_bmfe(mae, ax1=ax[2, 0], ax2=ax[2, 1], ax3=ax[2, 2])

    fig.tight_layout()