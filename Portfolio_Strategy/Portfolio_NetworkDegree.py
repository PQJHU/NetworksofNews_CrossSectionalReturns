import os
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.formula import api as sma
import calendar
import pickle

full_figsize = (16, 7)
tick_size = 16
label_size = 18


def portfolio_characters_overtime(portfolios_overtime, character, num_port, shift_period=1, avg_weight=False):
    """
    Given the constructed portfolios and their characteristics over time,
    return dataframe of num_port columns
    @param portfolios_overtime:
    @param character:
    @return:
    """
    timestamp = [t[0].index[0] for t in portfolios_overtime]
    if avg_weight:
        weighted_values_overtime = [[p[character].mean(axis=0) for p in periodic_portfolios] for
                                    periodic_portfolios in portfolios_overtime]
    else:
        weighted_values_overtime = [[(p[character] * p['weight']).sum(axis=0) for p in periodic_portfolios] for
                                    periodic_portfolios in portfolios_overtime]
    # strategic_returns_overtime = [np.multiply(periodic_returns, positions) for periodic_returns in weighted_values_overtime]
    cols = [f'p_{i}' for i in range(1, num_port + 1)]
    portfolios_weighted_values = pd.DataFrame(data=weighted_values_overtime, index=timestamp, columns=cols)
    # shift the periods
    if shift_period is not None:
        portfolios_weighted_values = portfolios_weighted_values.shift(periods=shift_period).dropna(axis=0)
    return portfolios_weighted_values


def plot_3portfolios_cumreturns(portfolios_cumsum, benchmark_cumsum, lb='(EW)', ifbenchmark=True):
    cols = portfolios_cumsum.columns
    plt.plot(portfolios_cumsum[f'{cols[0]}'], color='red', label='Portfolio Low ' + lb)
    plt.plot(portfolios_cumsum[f'{cols[1]}'], color='blue', label='Portfolio Medium ' + lb)
    plt.plot(portfolios_cumsum[f'{cols[2]}'], color='green', label='Portfolio High ' + lb)
    if ifbenchmark:
        plt.plot(benchmark_cumsum, color='black', linestyle='--', label='Market Portfolio ' + lb)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.xlabel('Time', fontsize=label_size)
    plt.ylabel('Cumulative Return', fontsize=label_size)
    plt.legend(fontsize='large', frameon=False)
    plt.xlim([portfolios_cumsum.index[0], portfolios_cumsum.index[-1]])
    plt.tight_layout()


def convert_month_int_index_2_datetime(df):
    _df = df.copy(deep=True)
    _df['Date'] = [dt.datetime.strptime(str(date), "%Y%m") for date in _df.index]
    _df['Date'] = [date.replace(day=calendar.monthrange(date.year, date.month)[1]) for date in _df['Date']]
    _df.set_index('Date', drop=True, inplace=True)
    return _df


def read_ff_rf(freq='daily'):
    # Read FF factors
    ff3 = pd.read_csv(os.getcwd() + f'/Portfolio_Strategy/ff3_{freq}.csv', index_col=0, parse_dates=True)
    ff5 = pd.read_csv(os.getcwd() + f'/Portfolio_Strategy/ff5_{freq}.csv', index_col=0, parse_dates=True)
    ff3.columns = ['ex_mkt', 'SMB', 'HML', 'RF']
    ff5.columns = ['ex_mkt', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
    rf = ff3['RF'] / 100  # Risk free return
    if freq == 'monthly':
        ff3 = convert_month_int_index_2_datetime(ff3)
        ff5 = convert_month_int_index_2_datetime(ff5)
    ff3 = ff3.loc[(ff3.index >= dt.datetime(2016, 1, 1)) & (ff3.index < dt.datetime(2021, 1, 1))]
    ff5 = ff5.loc[(ff5.index >= dt.datetime(2016, 1, 1)) & (ff5.index < dt.datetime(2021, 1, 1))]
    ff3 = ff3 / 100
    ff5 = ff5 / 100
    return ff3, ff5, rf


def portfolio_ts_analysis(portfolio_returns_ts, newey_west=False, ifstd=False, ff_freq='daily', adj_r2=False):
    # portfolio_returns_ts = portfolio_returns_ew_m1['p_3'].to_frame()
    ff3, ff5, rf = read_ff_rf(freq=ff_freq)
    p_col = portfolio_returns_ts.columns
    mean = portfolio_returns_ts.mean().values[0]
    std = portfolio_returns_ts.std().values[0]
    _sr_p = portfolio_returns_ts.join(ff3['RF'])
    sr = (_sr_p[p_col[0]] - _sr_p['RF']).mean() / std

    # FF3 regression
    p_ff3 = portfolio_returns_ts.join(ff3)
    # p_ff3[['p_1', 'ex_mkt', 'SMB', 'HML']].cumsum(axis=0).plot()
    lr_p_ff3 = sma.ols(formula=f"{p_col[0]} ~ ex_mkt + SMB + HML", data=p_ff3)
    if newey_west:
        reg_ff3_fit = lr_p_ff3.fit(cov_type='HAC', cov_kwds={'maxlags': 1})
    else:
        reg_ff3_fit = lr_p_ff3.fit()
    alpha_ff3 = reg_ff3_fit.params['Intercept']
    alpha_ff3_t = reg_ff3_fit.tvalues['Intercept']
    if adj_r2:
        ff3_rsqr = reg_ff3_fit.rsquared_adj
    else:
        ff3_rsqr = reg_ff3_fit.rsquared

    # FF5 regression
    p_ff5 = portfolio_returns_ts.join(ff5)
    # p_ff5[['p_1', 'ex_mkt', 'SMB', 'HML', 'RMW', 'CMA']].cumsum(axis=0).plot()
    lr_p_ff5 = sma.ols(formula=f"{p_col[0]} ~ ex_mkt + SMB + HML + RMW + CMA", data=p_ff5)
    if newey_west:
        reg_ff5_fit = lr_p_ff5.fit(cov_type='HAC', cov_kwds={'maxlags': 1})
    else:
        reg_ff5_fit = lr_p_ff5.fit()
    alpha_ff5 = reg_ff5_fit.params['Intercept']
    alpha_ff5_t = reg_ff5_fit.tvalues['Intercept']
    ff5_rsqr = reg_ff5_fit.rsquared
    # ff5_rsqr = reg_ff5_fit.rsquared_adj

    adjust_coef = 100
    sr_annualized_factor = np.sqrt(12)
    # mean and alphas are in bps
    mean = (mean * adjust_coef).round(2)
    std = (std * adjust_coef).round(2)
    # mean = (mean * 100 * np.sqrt(255)).round(2)
    alpha_ff3 = (alpha_ff3 * adjust_coef).round(2)
    alpha_ff3_t = alpha_ff3_t.round(2)
    alpha_ff5 = (alpha_ff5 * adjust_coef).round(2)
    alpha_ff5_t = alpha_ff5_t.round(2)
    # percentage
    ff3_rsqr = (ff3_rsqr * 100).round(2)
    ff5_rsqr = (ff5_rsqr * 100).round(2)
    sr = (sr * sr_annualized_factor).round(2)  # annualized sharpe ratio
    if ifstd:
        return mean, sr, alpha_ff3, alpha_ff3_t, ff3_rsqr, alpha_ff5, alpha_ff5_t, ff5_rsqr, std
    else:
        return mean, sr, alpha_ff3, alpha_ff3_t, ff3_rsqr, alpha_ff5, alpha_ff5_t, ff5_rsqr


basic_cols = ['mean', 'sr', 'alpha_ff3', 'alpha_ff3_t', 'ff3_rsqr', 'alpha_ff5', 'alpha_ff5_t', 'ff5_rsqr', 'std']
portfolios_stats = pd.DataFrame(columns=basic_cols)
longshort_cusum_ts = pd.DataFrame()

param = ('MixSector_attention', 'ew', 1)
factor2sort = param[0]
weighting = param[1]
h = param[2]
ff_freq = 'monthly'
dropzero = True
nport_long = 5
portfolio_label = f'{factor2sort}_{weighting}'

mkt_portfolio_rts = pd.read_csv(os.getcwd() + '/Portfolio_Strategy/MarketPortfolioReturns.csv', index_col=0,
                                parse_dates=True)
portfolios_returns = pd.read_csv(os.getcwd() + '/Portfolio_Strategy/NetworkDegree_PortfolioReturn.csv', index_col=0,
                                 parse_dates=True)
with open(os.getcwd() + '/Portfolio_Strategy/FactorPortfolios.pkl', 'rb') as port_rick:
    factor_portfolios = pickle.load(port_rick)

ew_mkt_portfolio_stats = portfolio_ts_analysis(portfolio_returns_ts=mkt_portfolio_rts[f'ew_mkt_r_t{h}'].to_frame(),
                                               newey_west=True, ff_freq=ff_freq, adj_r2=False, ifstd=True)
for portfolio in portfolios_returns.columns:
    portfolio_stats = portfolio_ts_analysis(portfolio_returns_ts=portfolios_returns[portfolio].to_frame(),
                                            newey_west=True, adj_r2=False, ifstd=True, ff_freq=ff_freq)
    portfolios_stats.loc[f'{portfolio}_{portfolio_label}', basic_cols] = portfolio_stats

lhsl_returns = portfolios_returns[f'p_{nport_long}'] - portfolios_returns[f'p_1']
lhsl_stats = portfolio_ts_analysis(portfolio_returns_ts=lhsl_returns.to_frame('longshort'), newey_west=True,
                                   ifstd=True, ff_freq=ff_freq, adj_r2=False)
portfolios_stats.loc[f'lhsl_{portfolio_label}', basic_cols] = lhsl_stats

portfolio_asset_num = [[len(p) for p in periodic_portfolios] for periodic_portfolios in factor_portfolios]
portfolios_asset_num = pd.DataFrame(data=portfolio_asset_num,
                                    index=[t[0].index[0] for t in factor_portfolios],
                                    columns=[f'p_{i}' for i in range(1, nport_long + 1)])
portfolios_asset_num_avg = round(portfolios_asset_num.mean(axis=0), 0)

# factor2sort stats
portfolio_factor2sort = portfolio_characters_overtime(portfolios_overtime=factor_portfolios, character=factor2sort,
                                                      num_port=nport_long, shift_period=0, avg_weight=False)
portfolio_factor2sort_avg = round(portfolio_factor2sort.mean(axis=0), 2)

# MV Pct stats
portfolios_mv = portfolio_characters_overtime(portfolios_overtime=factor_portfolios, character='mv_daily',
                                              num_port=nport_long, shift_period=0, avg_weight=False)
portfolios_mv['sum'] = portfolios_mv.sum(axis=1)
portfolios_mv_percnt = pd.concat(
    [(portfolios_mv[col] / portfolios_mv['sum']).to_frame(col) for col in portfolios_mv.columns[0:-1]], axis=1)
portfolios_mv_percnt_avg = portfolios_mv_percnt.mean(axis=0)
portfolios_mv_percnt_avg = round(portfolios_mv_percnt_avg * 100, 2)
# B/M stats
porffolios_bm = portfolio_characters_overtime(portfolios_overtime=factor_portfolios, character='BM_daily',
                                              num_port=nport_long, shift_period=0, avg_weight=False)
portfolios_bm_avg = porffolios_bm.mean(axis=0)
portfolios_bm_avg = round(portfolios_bm_avg * 100, 2)
# Liquidity stats
porffolios_liquidity = portfolio_characters_overtime(portfolios_overtime=factor_portfolios,
                                                     character='turnover_daily', num_port=nport_long,
                                                     shift_period=0, avg_weight=False)
portfolios_liquidity_avg = round(porffolios_liquidity.mean(axis=0) * 100, 2)
index_names = [f'p_{num}_{portfolio_label}' for num in range(1, nport_long + 1)]
portfolios_stats.loc[portfolios_stats.index.isin(index_names), 'mv_pct'] = portfolios_mv_percnt_avg.values.tolist()
portfolios_stats.loc[portfolios_stats.index.isin(index_names), 'bm'] = portfolios_bm_avg.values.tolist()
portfolios_stats.loc[portfolios_stats.index.isin(index_names), 'turnover'] = portfolios_liquidity_avg.values.tolist()
portfolios_stats.loc[
    portfolios_stats.index.isin(index_names), 'factor2sort'] = portfolio_factor2sort_avg.values.tolist()
portfolios_stats.loc[portfolios_stats.index.isin(index_names), 'asset_num'] = portfolios_asset_num_avg.values.tolist()

long_portfolios_cumsum = portfolios_returns.cumsum(axis=0)
plt.figure(figsize=full_figsize)
lb = '(EW)' if weighting == 'ew' else '(MW)'
ifbenchmark = True

plot_3portfolios_cumreturns(
    long_portfolios_cumsum[[f'p_1', f'p_{int(np.ceil(nport_long / 2))}', f'p_{nport_long}']],
    mkt_portfolio_rts[f'{weighting}_mkt_r_t{h}'].cumsum(axis=0), lb, ifbenchmark)
plt.savefig(
    os.getcwd() + f'/Portfolio_Strategy/nport{nport_long}_{weighting}_{factor2sort}_dz{dropzero}_freq{ff_freq}.png',
    dpi=300)
plt.close()

portfolios_stats.to_csv(os.getcwd() + f'/Portfolio_Strategy/nport{nport_long}_dz{dropzero}_freq{ff_freq}.csv')
