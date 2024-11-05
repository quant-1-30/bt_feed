#!/usr/bin/env python3
# -*- coding

import pdb
import requests

if __name__ == "__main__":

    url = "http://19.push2his.eastmoney.com/api/qt/stock/kline/get?secid=1.000001&ut=fa5fd1943c7b386f172d6893dbfba10b&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5%2Cf6&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58%2Cf59%2Cf60%2Cf61&klt=101&fqt=1&beg=0&end=20500101&smplmt=460&lmt=1000000&_=1709308485782"
    data = requests.get(url=url)
    mapping = eval(data.content.decode("utf-8"))
    klines = mapping["data"]["klines"]
    trading = [item.split(",")[0] for item in klines]
    with open("trading_calendar.csv", "w") as f:
        # write header
        f.write("trading_date")
        for dt in trading:
            f.write("\n")
            dt = dt.replace("-", "")
            f.write(dt)
    pdb.set_trace()