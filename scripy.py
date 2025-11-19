import akshare as ak
import pandas as pd
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
from datetime import datetime
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
        # 查找前一个非零值
        prev_idx = idx - 1
        while prev_idx >= 0 and stock.loc[prev_idx, 'price'] == 0:
            prev_idx -= 1
        
        # 查找后一个非零值
        next_idx = idx + 1
        while next_idx < len(stock) and stock.loc[next_idx, 'price'] == 0:
            next_idx += 1
        
        # 计算前后非零值的平均值
        prev_value = stock.loc[prev_idx, 'price'] if prev_idx >= 0 else 0
        next_value = stock.loc[next_idx, 'price'] if next_idx < len(stock) else prev_value
        
        # 如果前值为0但后值不为0，则使用后值
        if prev_value == 0 and next_value != 0:
            stock.loc[idx, 'price'] = next_value
        # 如果后值为0但前值不为0，则使用前值
        elif next_value == 0 and prev_value != 0:
            stock.loc[idx, 'price'] = prev_value
        # 如果前后值都不为0，则取平均值
        elif prev_value != 0 and next_value != 0:
            stock.loc[idx, 'price'] = (prev_value + next_value) / 2
        # 如果都为0，则保持原值（这种情况很少见）

# 黄金 000218
# 短融 000128
# 长债  003376
# 30长债 010309
# 美债 004998
# 红利 161907, 510880, 009051
# 红利低波 005561, 512890, 021482
# 标普 161125
# 纳斯达克 160213
# 创业板 110026
def get_data():
    d1 = ak.fund_open_fund_info_em(symbol="000218", indicator="累计净值走势").rename(columns={'净值日期': 'date', '累计净值': 'price'})
    cash = ak.fund_open_fund_info_em(symbol="000128", indicator="累计净值走势").rename(columns={'净值日期': 'date', '累计净值': 'price'}) #短融现金储备不能改
    d2 = ak.fund_open_fund_info_em(symbol="010309", indicator="累计净值走势").rename(columns={'净值日期': 'date', '累计净值': 'price'})
    #股票
    d3 = ak.fund_open_fund_info_em(symbol="009051", indicator="累计净值走势").rename(columns={'净值日期': 'date', '累计净值': 'price'})
    d4 = ak.fund_open_fund_info_em(symbol="021482", indicator="累计净值走势").rename(columns={'净值日期': 'date', '累计净值': 'price'})
    d5 = ak.fund_open_fund_info_em(symbol="160213", indicator="累计净值走势").rename(columns={'净值日期': 'date', '累计净值': 'price'})
    d6 = ak.fund_open_fund_info_em(symbol="161125", indicator="累计净值走势").rename(columns={'净值日期': 'date', '累计净值': 'price'})
    stock = merge_fund_data_by_date(d3, d4, d5, d6)
    quit_zero(stock)
    dataframes = [d1, cash, d2, stock]
    merged_data = reduce(lambda left, right: left.merge(right, on='date', how='inner', suffixes=('', '_right')), dataframes)
    merged_data.columns = ['date', '黄金', '短融', '长债', '股票']
    merged_data['date'] = pd.to_datetime(merged_data['date'])
    return merged_data


import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

# -----------------------
# 辅助函数
# -----------------------
def _get_last_price_on_or_before(df: pd.DataFrame, asset: str, date: pd.Timestamp) -> float:
    """
    返回 asset 在 date 当天或之前最近一天的 price。若没有可用数据抛出 KeyError。
    df 必须含 'date' 列，已排序或未排序均可。
    """
    series = df[['date', asset]].dropna().sort_values('date')
    idx = series[series['date'] <= date].last_valid_index()
    if idx is None:
        raise KeyError(f"No price for {asset} on or before {date}")
    return float(series.loc[idx, asset])

def _round_shares(shares: float, ndigits: int = 4) -> float:
    """份额保留 ndigits 位（默认 4 位）"""
    return float(np.floor(shares * (10 ** ndigits)) / (10 ** ndigits))

# -----------------------
# 1) 建仓函数（和你之前一致，并稍作丰富）
# -----------------------
def build_portfolio(
        df: pd.DataFrame,
        total_money: float = 100000.0,
        months: int = 12,
        fixed_cash_weight: float = 0.25,
        alpha: float = 0.3
    ) -> Dict[str, pd.Series]:
    """
    建仓：将 永久组合 (PP) 与 风险平价 (排除短融) 按 alpha 混合得到最终权重，再按 total_money 分配。
    返回 dict 含：
      - "永久组合权重"
      - "风险平价权重（排除短融）"
      - "融合后最终建仓权重"
      - "投资金额"
      - "初始持仓份额"（按最近 price 计算）
    注意：df 需包含列 ['date','黄金','短融','长债','股票']
    """
    df = df.sort_values('date').reset_index(drop=True)
    assets = ["黄金", "短融", "长债", "股票"]

    # 永久组合权重（短融固定）
    pp_weights = pd.Series({
        "黄金": (1 - fixed_cash_weight) / 3,
        "长债": (1 - fixed_cash_weight) / 3,
        "股票": (1 - fixed_cash_weight) / 3,
        "短融": fixed_cash_weight
    })

    # 计算最近 M 月风险平价（仅三风险资产）
    end_date = df['date'].max()
    start_date = end_date - pd.DateOffset(months=months)
    df_m = df[df['date'] >= start_date].copy()
    risk_assets = ["黄金", "长债", "股票"]
    rets = df_m[risk_assets].pct_change().dropna()
    vol = rets.std() * np.sqrt(250)
    vol = vol.replace(0, 1e-8)
    inv_risk = 1 / vol
    rp_weights = (inv_risk / inv_risk.sum()).astype(float)

    # 将风险平价扩展到所有资产（短融保持固定）
    rp_full = pd.Series(index=assets, dtype=float)
    for a in risk_assets:
        rp_full[a] = rp_weights[a]
    rp_full['短融'] = fixed_cash_weight
    # rp_weights currently sums to 1 for risk assets; we need to scale them so that risk_assets total = 1 - fixed_cash_weight
    rp_full[risk_assets] = rp_full[risk_assets] * (1 - fixed_cash_weight)
    # ensure sum==1
    rp_full = rp_full.fillna(0.0)
    rp_full = rp_full / rp_full.sum()

    # 融合最终权重
    final_weights = pp_weights * (1 - alpha) + rp_full * alpha
    final_weights = final_weights / final_weights.sum()

    # 投资金额
    money_alloc = final_weights * total_money

    # 计算初始份额（使用最近日期价格）
    latest_date = df['date'].max()
    shares = {}
    for a in assets:
        price = _get_last_price_on_or_before(df, a, latest_date)
        s = _round_shares(money_alloc[a] / price if price > 0 else 0.0)
        shares[a] = s

    shares = pd.Series(shares)

    return {
        "永久组合权重": pp_weights,
        "风险平价权重（排除短融）": rp_weights,
        "风险平价扩展权重（含短融）": rp_full,
        "融合后最终建仓权重": final_weights,
        "投资金额": money_alloc,
        "初始持仓份额": shares,
        "最新估值日期": latest_date
    }

# -----------------------
# 2) 定投函数（DCA）
# -----------------------
def dca_invest(
        df: pd.DataFrame,
        start_date: str,
        end_date: Optional[str],
        monthly_amount: float,
        weights: Optional[Dict[str, float]] = None,
        freq: str = 'MS',    # 'MS' 每月月初，'M' 月末（pandas freq）
        ndigits_shares: int = 4
    ) -> Tuple[pd.DataFrame, pd.Series]:
    """
    模拟定投过程（不考虑交易费），返回两样东西：
      - trades_df: DataFrame，逐次买入记录（date, asset, price, amount(元), shares）
      - final_holdings: Series，每个资产累计份额

    参数：
      - df: get_data() 返回的历史净值表
      - start_date: 'YYYY-MM-DD' 格式，定投开始（包含该月）
      - end_date: 'YYYY-MM-DD' 或 None（None 则到 df 的 max date）
      - monthly_amount: 每期投入总额（元）
      - weights: 字典映射资产->比例（若 None 则使用永久组合 3/4 分配 + 短融固定）
      - freq: pandas 频率字符串，默认 'MS'（每月月初）
    注意：定投在每个 period 的当天使用该资产最近可得净值（当日或之前最近一天）。
    """
    df = df.sort_values('date').reset_index(drop=True)
    assets = ["黄金", "短融", "长债", "股票"]
    if weights is None:
        # 默认按永久组合（短融固定按 25%），其余均分
        weights = {"短融": 0.25, "黄金": 0.25, "长债": 0.25, "股票": 0.25}
    w = pd.Series(weights)
    w = w.reindex(assets).fillna(0.0)
    if w.sum() <= 0:
        raise ValueError("weights sum must > 0")
    w = w / w.sum()

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date) if end_date is not None else df['date'].max()
    # build schedule
    schedule = pd.date_range(start=start, end=end, freq=freq)
    trades = []
    holdings = {a: 0.0 for a in assets}

    for dt in schedule:
        # 对每个资产，找到在 dt 或之前最近的价格
        for a in assets:
            try:
                price = _get_last_price_on_or_before(df, a, dt)
            except KeyError:
                # 若没有价格（太早），跳过该资产本期购买
                continue
            amount = monthly_amount * float(w[a])
            shares = _round_shares(amount / price, ndigits=ndigits_shares)
            if shares <= 0:
                continue
            holdings[a] += shares
            trades.append({
                "date": dt,
                "asset": a,
                "price": price,
                "amount": round(shares * price, 2),
                "shares": shares
            })

    trades_df = pd.DataFrame(trades)
    final_holdings = pd.Series(holdings)
    return trades_df, final_holdings

# -----------------------
# 3) 调仓指令生成（按目标权重）
# -----------------------
def rebalance_to_target(
        holdings: pd.Series,
        df: pd.DataFrame,
        target_weights: Dict[str, float],
        as_of_date: Optional[str] = None,
        cash_buffer: float = 0.0,
        ndigits_shares: int = 4
    ) -> Tuple[pd.DataFrame, pd.Series]:
    """
    生成调仓指令把当前持仓(份额) 调整到 target_weights（目标权重）。
    - holdings: Series, index: ["黄金","短融","长债","股票"] 值为份额
    - df: 历史净值表，用于取 as_of_date 的价格（或最近之前）
    - target_weights: dict 目标权重（按总市值）
    - as_of_date: 'YYYY-MM-DD' 字符串或 None（None 则使用 df 的 max date）
    - cash_buffer: 保留的现金比例（例如 0.01 表示保留 1% 现金，不用于买入）
    返回：
      - trades_df: DataFrame 每行一条交易指令 (asset, current_value, target_value, diff_value, price, shares_to_trade)
        diff_value >0 表示买入，<0 表示卖出
      - target_shares: Series 调仓后目标份额
    说明：
      - 假设卖出能得到的现金立即可用用于买入
      - 不考虑手续费与滑点；若需要可在外部对 amount 做调整
    """
    assets = ["黄金", "短融", "长债", "股票"]
    as_of = pd.to_datetime(as_of_date) if as_of_date else df['date'].max()

    # 当日价格
    prices = {}
    for a in assets:
        prices[a] = _get_last_price_on_or_before(df, a, as_of)
    prices = pd.Series(prices)

    # 当前市值
    cur_values = holdings.reindex(assets).fillna(0.0) * prices

    total_value = cur_values.sum()
    if total_value <= 0:
        raise ValueError("当前组合总市值 <= 0，无法调仓")

    # 调整目标权重（减去 cash_buffer）
    t_w = pd.Series(target_weights).reindex(assets).fillna(0.0)
    t_w = t_w / t_w.sum()
    if cash_buffer > 0:
        # 将 cash_buffer 视为现金不参与投资（从短融中扣除，或作为额外空余现金）
        # 这里实现：整体目标按 (1 - cash_buffer) 缩放
        t_w = t_w * (1 - cash_buffer)
        # 现金缓冲留在组合外，不分配给资产
    target_values = t_w * total_value
    # target shares
    target_shares = (target_values / prices).apply(lambda x: _round_shares(x, ndigits=ndigits_shares))

    # 交易指令 = target_shares - current_shares
    trades = []
    for a in assets:
        cur_sh = float(holdings.reindex(assets).fillna(0.0)[a])
        tgt_sh = float(target_shares[a])
        delta_sh = round(tgt_sh - cur_sh, ndigits_shares)
        trades.append({
            "asset": a,
            "price": prices[a],
            "current_shares": cur_sh,
            "target_shares": tgt_sh,
            "delta_shares": delta_sh,
            "current_value": round(cur_sh * prices[a], 2),
            "target_value": round(tgt_sh * prices[a], 2),
            "diff_value": round((tgt_sh - cur_sh) * prices[a], 2)
        })

    trades_df = pd.DataFrame(trades).set_index('asset')
    return trades_df, target_shares

# -----------------------
# 4) 执行交易（把 trades 应用到 holdings）
# -----------------------
def execute_trades(
        holdings: pd.Series,
        trades_df: pd.DataFrame,
        cash: float = 0.0,
        allow_negative_cash: bool = False
    ) -> Tuple[pd.Series, float]:
    """
    将 trades_df 应用到 holdings，返回 (new_holdings, remaining_cash)
    trades_df 需包含列 delta_shares 与 price
    - 正 delta_shares 表示买入（消耗现金），负表示卖出（获得现金）
    - cash 为当前可用现金（元）
    - allow_negative_cash: 是否允许现金为负（借入）
    """
    holdings = holdings.copy().astype(float)
    cash = float(cash)
    for asset, row in trades_df.iterrows():
        delta_sh = float(row['delta_shares'])
        price = float(row['price'])
        if delta_sh == 0:
            continue
        trade_value = delta_sh * price
        # 卖出：增加现金，买入：减少现金
        cash -= trade_value
        holdings[asset] = float(holdings.reindex(holdings.index).fillna(0.0)[asset]) + delta_sh

    if (not allow_negative_cash) and (cash < -1e-6):
        raise ValueError(f"现金不足以执行交易，剩余现金 {cash:.2f}. 设置 allow_negative_cash=True 可允许透支。")

    return holdings, cash

# -----------------------
# 高级：组合管理器（把上面功能串起来，做一次建仓 + 多期定投 + 定期调仓模拟）
# -----------------------
def portfolio_manager(
        df: pd.DataFrame,
        initial_capital: float,
        build_months: int = 12,
        fixed_cash_weight: float = 0.25,
        alpha: float = 0.3,
        dca_start: str = None,
        dca_end: str = None,
        monthly_amount: Optional[float] = None,
        dca_freq: str = 'MS',
        rebalance_every_months: Optional[int] = 3,
        rebalance_threshold: Optional[float] = None,
        cash_buffer: float = 0.0
    ) -> Dict:
    """
    一个方便的 orchestration 函数（仅作模拟/回测辅助）：
    - 第一步：按 build_portfolio 建仓（把 initial_capital 全部用完建仓）
    - 第二步（可选）：从 dca_start 到 dca_end 每期定投 monthly_amount
    - 第三步（可选）：每 rebalance_every_months 执行一次 rebalance_to_target（目标权重为 build_portfolio 的 融合后最终建仓权重）
    返回 dict 包含建仓信息、所有定投记录、每次调仓记录、最终持仓与现金等
    """
    # 1) 建仓（用 initial_capital）
    base = build_portfolio(df, total_money=initial_capital, months=build_months,
                           fixed_cash_weight=fixed_cash_weight, alpha=alpha)
    holdings = base['初始持仓份额'].reindex(["黄金","短融","长债","股票"]).fillna(0.0)
    cash = 0.0  # 建仓时假定把钱全部转成份额（没有剩余现金）
    target_weights = base['融合后最终建仓权重'].to_dict()

    logs = {
        "build": base,
        "dca_trades": pd.DataFrame(),
        "rebalance_logs": []
    }

    # 2) 定投（如果指定）
    if monthly_amount is not None and dca_start is not None:
        trades_df, dca_holdings = dca_invest(df, dca_start, dca_end, monthly_amount,
                                             weights=None, freq=dca_freq)
        # 将定投得到的份额并入 holdings（相当于把定投的钱来自外部现金，不影响 cash 变量）
        for a in dca_holdings.index:
            holdings[a] = float(holdings.get(a, 0.0)) + float(dca_holdings[a])
        logs['dca_trades'] = trades_df

    # 3) 定期调仓（按频率）
    if rebalance_every_months is not None and rebalance_every_months > 0:
        # 构建时间表，从建仓日期开始（建仓日期取 df 的 max date 当作“现在”）
        start = pd.to_datetime(df['date'].max())
        # 进行若干次 rebalance（向后模拟没有 new prices），这里我们只做一次基于当前最新价格的调仓
        # 如果你想按历史时间点回测，多次 rebalance 需提供历史时间序列触发点
        trades_df, targ_shares = rebalance_to_target(holdings, df, target_weights,
                                                     as_of_date=None,
                                                     cash_buffer=cash_buffer)
        logs['rebalance_logs'].append({
            "as_of": df['date'].max(),
            "trades": trades_df
        })
        # 执行交易（允许卖出获得现金用于买入）
        try:
            holdings, cash = execute_trades(holdings, trades_df, cash=cash, allow_negative_cash=True)
        except ValueError as e:
            # 若不允许负现金则会报错；这里我们允许透支以确保调仓指令能给出
            raise

    # 汇总最终估值
    prices_now = {a: _get_last_price_on_or_before(df, a, df['date'].max()) for a in ["黄金","短融","长债","股票"]}
    prices_now = pd.Series(prices_now)
    final_values = holdings * prices_now
    total_value = final_values.sum() + cash

    logs['final'] = {
        "holdings": holdings,
        "prices": prices_now,
        "market_values": final_values,
        "cash": cash,
        "total_value": total_value
    }

    return logs

# -----------------------
# 使用示例（伪代码）
# -----------------------
df = get_data()
# 1) 建仓
out = build_portfolio(df, total_money=158000, months=6, fixed_cash_weight=0.25, alpha=0.3)
print(out['融合后最终建仓权重'])
print(out['投资金额'])
print(out['初始持仓份额'])

# # 2) 定投模拟
trades_df, final_holdings = dca_invest(df, start_date='2024-1-01', end_date=None, monthly_amount=10000)
# print(trades_df.head())

# 3) 调仓指令
trades_df, target_shares = rebalance_to_target(final_holdings + out['初始持仓份额'], df, out['融合后最终建仓权重'])
print(trades_df)

# 4) 执行交易
# new_holdings, remaining_cash = execute_trades(final_holdings + out['初始持仓份额'], trades_df, cash=0.0, allow_negative_cash=True)
