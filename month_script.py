import akshare as ak
import pandas as pd
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional


plt.rcParams['font.sans-serif'] = ['AR PL UKai CN','Noto Sans Kaithi']
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def merge_fund_data_by_date(*dataframes):
    merged_df = dataframes[0][['date', 'price']].copy()
    merged_df.columns = ['date', 'price_0']
    for i, df in enumerate(dataframes[1:], 1):
        temp_df = df[['date', 'price']].copy()
        temp_df.columns = ['date', f'price_{i}']
        merged_df = merged_df.merge(temp_df, on='date', how='inner')
    price_columns = [col for col in merged_df.columns if col.startswith('price_')]
    merged_df['price'] = merged_df[price_columns].sum(axis=1)/len(price_columns)
    return merged_df[['date', 'price']]

def quit_zero(stock):
    stock_zero_mask = stock['price'] == 0
    zero_indices = stock[stock_zero_mask].index
    for idx in zero_indices:
        prev_idx = idx - 1
        while prev_idx >= 0 and stock.loc[prev_idx, 'price'] == 0:
            prev_idx -= 1
        next_idx = idx + 1
        while next_idx < len(stock) and stock.loc[next_idx, 'price'] == 0:
            next_idx += 1
        prev_value = stock.loc[prev_idx, 'price'] if prev_idx >= 0 else 0
        next_value = stock.loc[next_idx, 'price'] if next_idx < len(stock) else prev_value
        if prev_value == 0 and next_value != 0:
            stock.loc[idx, 'price'] = next_value
        elif next_value == 0 and prev_value != 0:
            stock.loc[idx, 'price'] = prev_value
        elif prev_value != 0 and next_value != 0:
            stock.loc[idx, 'price'] = (prev_value + next_value) / 2

# 黄金 000218
# 短融 000128, 008448
# 长债  003376
# 30长债 010309
# 美债 004998
# 红利 161907, 510880, 009051
# 红利低波 005561, 512890, 021482
# 标普 161125
# 纳斯达克 160213
# 创业板 110026
# 上证50 001051
# 沪深300 460300
def get_data():
    d1 = ak.fund_open_fund_info_em(symbol="000218", indicator="累计净值走势").rename(columns={'净值日期': 'date', '累计净值': 'price'})
    cash = ak.fund_open_fund_info_em(symbol="008448", indicator="累计净值走势").rename(columns={'净值日期': 'date', '累计净值': 'price'}) #短融现金储备不能改
    d2 = ak.fund_open_fund_info_em(symbol="010309", indicator="累计净值走势").rename(columns={'净值日期': 'date', '累计净值': 'price'})
    #股票
    d3 = ak.fund_open_fund_info_em(symbol="009051", indicator="累计净值走势").rename(columns={'净值日期': 'date', '累计净值': 'price'})
    d4 = ak.fund_open_fund_info_em(symbol="021482", indicator="累计净值走势").rename(columns={'净值日期': 'date', '累计净值': 'price'})
    stock = merge_fund_data_by_date(d3, d4)
    quit_zero(stock)
    dataframes = [d1, cash, d2, stock]
    merged_data = reduce(lambda left, right: left.merge(right, on='date', how='inner', suffixes=('', '_right')), dataframes)
    merged_data.columns = ['date', '黄金', '短融', '长债', '股票']
    merged_data['date'] = pd.to_datetime(merged_data['date'])
    return merged_data

def _annualize_vol(returns: pd.Series, periods_per_year: int = 12) -> float:
    return returns.std(ddof=1) * np.sqrt(periods_per_year)


def inverse_vol_weights(vols: np.ndarray) -> np.ndarray:
    inv = 1.0 / np.where(vols <= 0, 1e-8, vols)
    w = inv / np.sum(inv)
    return w

def compute_monthly_prices(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy()
    if 'date' in df.columns:
        df = df.set_index(pd.to_datetime(df['date'])).drop(columns=['date'])
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError('prices must have a DatetimeIndex or a date column')
    monthly = df.resample('ME').last()
    return monthly


def allocate_today_instant(
    prices: pd.DataFrame,
    amount: float,
    cash_col: str = '短融',
    longbond_col: str = '长债',
    gold_col: str = '黄金',
    equity_col: str = '股票',
    alpha: float = 0.5,
    cash_weight: float = 0.25,
    vol_lookback: int = 6,
    date: Optional[pd.Timestamp] = None,
) -> Dict[str, any]:
    """
    给定今天的可投资金额 amount（元），计算如何按回测中使用的规则配置仓位。

    核心规则（与回测一致）
    - 短融（cash_col）固定占 cash_weight（例如 0.25），不参与风险平价
    - 以最近 vol_lookback 月的波动/协方差计算长债、黄金、股票的风险平价权重
    - 三资产权重按 alpha 混合：blended3 = alpha * rp + (1-alpha) * pp3
    - 三资产占比乘以 (1 - cash_weight)，最终加上短融的 cash_weight

    返回字典包含：
    - 'target_weights': 每个资产的目标权重
    - 'amounts': 每个资产应投入的金额（元）
    - 'prices': 用于计算的最新价格
    - 'shares': 以给定价格可买的份额（如果 allow_fractional_shares=False，则向下取整）
    - 'cash_remaining': 如果不允许分数份额，因取整产生的剩余现金

    参数说明：与 backtest_blended_portfolio 大致对应，date 指定为 None 时使用 prices 中最后可用观测日
    """
    prices_monthly = compute_monthly_prices(prices)
    if date is None:
        date = prices_monthly.index[-1]
    else:
        date = pd.to_datetime(date)
        if date not in prices_monthly.index:
            date = prices_monthly.index[prices_monthly.index.get_indexer([date], method='ffill')[0]]

    needed = [cash_col, longbond_col, gold_col, equity_col]
    for c in needed:
        if c not in prices_monthly.columns:
            raise KeyError(f"prices missing required column: {c}")

    idx = prices_monthly.index.get_loc(date)
    lookback_start = max(0, idx - vol_lookback)
    window_returns = prices_monthly.pct_change().iloc[lookback_start:idx+1].dropna(how='all')

    if window_returns.shape[0] < 2:
        vols = np.array([
            _annualize_vol(prices_monthly[longbond_col].pct_change().dropna()) if prices_monthly[longbond_col].dropna().shape[0] > 1 else 1e-8,
            _annualize_vol(prices_monthly[gold_col].pct_change().dropna()) if prices_monthly[gold_col].dropna().shape[0] > 1 else 1e-8,
            _annualize_vol(prices_monthly[equity_col].pct_change().dropna()) if prices_monthly[equity_col].dropna().shape[0] > 1 else 1e-8,
        ])
    else:
        vols = np.array([
            _annualize_vol(window_returns[longbond_col].dropna()),
            _annualize_vol(window_returns[gold_col].dropna()),
            _annualize_vol(window_returns[equity_col].dropna()),
        ])

    try:
        rp3 = inverse_vol_weights(vols)
    except Exception:
        rp3 = inverse_vol_weights(vols)

    pp3 = np.array([0.25, 0.25, 0.25])
    pp3 = pp3 / pp3.sum()

    blended3 = alpha * rp3 + (1.0 - alpha) * pp3
    allocated3 = blended3 * (1.0 - cash_weight)

    target_weights = {
        longbond_col: float(allocated3[0]),
        cash_col: float(cash_weight),
        gold_col: float(allocated3[1]),
        equity_col: float(allocated3[2])
    }

    return {c: target_weights[c] * amount for c in needed}

print(allocate_today_instant(get_data(), 158000, alpha=0.5, cash_weight=0.25, vol_lookback=3))
