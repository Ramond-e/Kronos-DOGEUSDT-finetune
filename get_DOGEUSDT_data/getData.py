from binance import Client
import time
import datetime
import csv
import sys

def ms2str(ms):
    return datetime.datetime.utcfromtimestamp(ms / 1000).strftime("%Y-%m-%d %H:%M:%S")

def get_klines_with_retry(client, params, max_retries=3):
    """带重试机制的USDⓈ-M合约K线数据获取"""
    for attempt in range(max_retries):
        try:
            return client.futures_klines(**params)
        except Exception as e:
            print(f"第 {attempt + 1} 次尝试失败: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                print("达到最大重试次数，退出程序")
                sys.exit(1)

client = Client()
symbol = "DOGEUSDT"
interval = "1h"
limit = 1500  # USDⓈ-M 合约接口单次最多可取 1500 条

all_klines = []
processed_times = set()  # 用于避免重复数据

# 从当前时间开始，按 endTime 向过去回溯，直到历史最早
end_time = int(time.time() * 1000)

print(f"开始获取 {symbol} {interval} 合约K线数据（USD-M）...")

while True:
    try:
        params = {"symbol": symbol, "interval": interval, "limit": limit, "endTime": end_time}
        
        klines = get_klines_with_retry(client, params)
        
        if not klines:
            print("没有更多数据")
            break
        
        # 过滤重复数据
        new_klines = []
        for kline in klines:
            if kline[0] not in processed_times:
                new_klines.append(kline)
                processed_times.add(kline[0])
        
        if not new_klines:
            print("所有数据都已处理过，退出")
            break
            
        # 由于是向过去抓取，这里暂存顺序为由近及远；最终写出前会按时间升序排序
        all_klines.extend(new_klines)
        print(f"累计 {len(all_klines)} 条，最早时间：{ms2str(new_klines[0][0])}，最新抓取块的最晚时间：{ms2str(new_klines[-1][0])}")
        
        # 下一轮设置 endTime 为当前块最早一根K线之前
        first_open_time = new_klines[0][0]
        end_time = first_open_time - 1
        
        # 若本次返回不满 limit 条，说明已到最早历史
        if len(klines) < limit:
            print("已到达最早历史数据")
            break
            
        # 添加延迟避免API限制
        time.sleep(0.5)
        
    except KeyboardInterrupt:
        print("\n用户中断程序")
        break
    except Exception as e:
        print(f"发生错误: {e}")
        break

if all_klines:
    # 写出前按 open_time 升序排序
    all_klines.sort(key=lambda x: x[0])
    # 导出为 CSV
    filename = f"{symbol.lower()}_{interval}_all_klines.csv"
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        writer.writerows(all_klines)

    print(f"全部获取完成，共 {len(all_klines)} 条数据")
    print(f"起始时间：{ms2str(all_klines[0][0])}，结束时间：{ms2str(all_klines[-1][0])}")
    print(f"数据已保存到: {filename}")
else:
    print("没有获取到任何数据")