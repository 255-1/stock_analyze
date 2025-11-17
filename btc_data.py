import ccxt
import pandas as pd
import time
import datetime

# 初始化交易所
exchange = ccxt.binance({
    "enableRateLimit": True
})

symbol = "BTC/USDT"
timeframe = "1d"

# 定义开始时间
since = int(pd.Timestamp("2016-01-01").timestamp() * 1000)

def fetch_all_ohlcv(symbol, timeframe="1d", since=None):
    all_data = []
    limit = 1000
    while True:
        try:
            data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        except ccxt.NetworkError as e:
            print("网络错误，重试中...", e)
            time.sleep(5)
            continue
        except ccxt.ExchangeError as e:
            print("交易所错误，退出", e)
            break

        if not data:
            break

        all_data.extend(data)
        since = data[-1][0] + 1  # 下一条时间戳
        print(f"已获取 {len(all_data)} 条数据，最新日期: {pd.to_datetime(data[-1][0], unit='ms')}")
        time.sleep(exchange.rateLimit / 1000)  # 限速

        if len(data) < limit:
            break

    return all_data

# 获取历史数据
ohlcv = fetch_all_ohlcv(symbol, timeframe, since)

# 转 DataFrame
df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
df["date"] = pd.to_datetime(df["timestamp"], unit="ms")

# 计算净值
initial_price = df["close"].iloc[0]
df["price"] = df["close"] / initial_price           # 累积净值
df["daily_return"] = df["close"].pct_change()           # 每日收益率

df = df[['date', 'price']]
# 保存
df.to_csv("btc_daily_nav_2016_to_now.csv", index=False)
print("完成！已保存 btc_daily_nav_2016_to_now.csv")
