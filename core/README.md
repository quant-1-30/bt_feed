simulate 
# 如果价格为接近涨跌幅（threshold=0.1%）则无法成交
# 无法构建orderbook, 只能以溢价模拟成交
# 控制冲击成本 交易量 * thres ( e.g. 70%)

1、tradeapi

    a. on_trade --- direction / symbol 
    b. on_event --- dividend / rights
    c. on_close --- close_position
    d. onQueryPosition
    e. onQueryAccount 
    f. simulate minute ticker downsample to 3s 
    g. policy + offset ---> order 
    h. record order and update ledger

# 关于卖出与买入策略, 关键为更新ledger cash 对应时间戳，在sim里面可以顺序执行买入与卖出, 实盘交易的时候内部处理好

2、quoteapi

    a. udp ---> grpc client

3、 bytes ---> int () int.from_bytes() | struct.unpack(), 大端(human)与小端数据


# sdk 整合 tradeapi / quoteapi

# 策略 以bayies 为主

workflow --- based on ray and big model sam to analyse ( indicator + pymc)

# # tradeapi --- ExecuPlan 执行计划 (risk / restricted)

# python 3.11 asyncio 对udp支持, 之前版本对于tcp stream 
