#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pdb
import tushare as ts


if __name__ == "__main__":

    print(ts.__version__)
    pro = ts.pro_api(token="7325ca7b347c682eabdd9e9335f16526d01f6dff2de6ed80792cde25")
    exchange = " SSE"
    df = pro.trade_cal(exchange=exchange, start_date='20000101', end_date='20240301')
    pdb.set_trace()