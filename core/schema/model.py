# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import pdb
import pickle
import datetime
import numpy as np
import pandas as pd
import uuid
import math
from dataclasses import dataclass
from typing import List, Any, Dict, Union
from enum import Enum


class OrderType(Enum):

    # lmt / fok / fak
    LIMIT = 1 << 1
    Market = 1 << 2


class OrderStatus(Enum):

    OPEN = 1
    FILLED = 2
    CANCELLED = 3
    REJECTED = 4
    HELD = 5
    SUBMITTING = 6


class BaseObject(object):

    # Descriptor
    # def __get__(self, instance, owner):
    #     """
    #     """

    # def __set__(self, instance, value):
    #     """
    #     """

    def __contains__(self, name):
        return name in self.__slots__
    
    def __getitem__(self, attr):
        return self.__getattribute__(attr)

    def __getstate__(self):
        state = {}
        """ 
            pickle -- __getstate__ , __setstate__
            __dict__ ---> __getattribute__ ---> getattr
        """
        for attr in self.__slots__:
            state[attr] = self.__getattribute__(attr)
        return state
    
    def __setstate__(self, state: dict[Any, Any]) -> None:
        # pickle loads 入参为__getstate__返回的结果
        # Adding 'sid' for backwards compatibility with downstream consumers.
        print("state", state)
        for k, v in state.items():
            self.__setattr__(k, v)
    
    def __missing__(self, key):
        return "NotFound"

    def __repr__(self):
        """
            String representation for this object.
        """
        state = {attr: self.__getattribute__(attr) for attr in self.__slots__ }
        return "{0}(**{1})".format(self.__class__.__name__, state)


class Asset(BaseObject):
    """
        前5个交易日,科创板科创板还设置了临时停牌制度, 当盘中股价较开盘价上涨或下跌幅度首次达到30%、60%时，都分别进行一次临时停牌
        单次盘中临时停牌的持续时间为10分钟。每个交易日单涨跌方向只能触发两次临时停牌, 最多可以触发四次共计40分钟临时停牌。
        如果跨越14:57则复盘, 之后交易日20%
        科创板盘后固定价格交易 15:00 --- 15:30
        若收盘价高于买入申报指令，则申报无效；若收盘价低于卖出申报指令同样无效

        A股主板, 中小板首日涨幅最大为44%而后10%波动，针对不存在价格笼子（科创板，创业板后期对照科创板改革）
        科创板在连续竞价机制 ---买入价格不能超过基准价格(卖一的102%, 卖出价格不得低于买入价格98%)
        设立市价委托必须设立最高价以及最低价
        科创板盘后固定价格交易 --- 以后15:00收盘价格进行交易 --- 15:00 -- 15:30(按照时间优先原则，逐步撮合成交）
    """
    __slots__ = ("sid", "first_trading", "delist")

    def __init__(self, sid: str, first_trading: int, delist: int=0):
        self.sid = sid
        self.first_trading = first_trading
        self.delist = delist

    @property
    def tick_size(self):
        """
            科创板 --- 申报最小200股, 递增可以以1股为单位 | 卖出不足200股一次性卖出
            创业板 --- 申报100 倍数 | 卖出不足100, 全部卖出
        """
        tick_size = 200 if self.sid.startswith("688") else 100
        return tick_size

    @property
    def increment(self):
        """
            multiplier / scatter 
        """
        incr = 200 if self.sid.startswith("688") else 100
        return incr 
    
    def on_special(self, dt):
        """
            st period 由于缺少st数据默认为False
        """
        return False

    def on_restricted(self, dt):
        """
            创业板2020年8月24日 20% 涨幅， 上市前五个交易日不设置涨跌幅, 第6个交易日开始
        """
        if self.sid.startswith("688"):
            limit = 0.2
        elif self.sid.startswith("3"):
            limit = 0.2 if datetime.datetime.strptime(str(dt), "%Y%m%d") >= datetime.date(2020, 8, 24) else 0.1
        else:
            limit = 0.1
    
        # special treatment
        if self.on_special(dt):
                limit = 0.2 if datetime.datetime.strptime(str(dt), "%Y%m%d") >= datetime.date(2020, 8, 24) else 0.05

        return limit

    def on_claim_restricted(self):
        """
            沪深主板单笔申报不超过100万股, 创业板单笔限价30万股;市价不超过15万股;定价申报100万股
            科创板单笔申报10万股;市价5万股;定价100万股
            风险警示板单笔买入申报50万股; 单笔卖出申报100万股, 退市整理期100万股
        """
        return np.Inf
    

class Order(BaseObject):

    # using __slots__ to save on memory usage.  Simulations can create many
    # Order objects and we keep them all in memory, so it's worthwhile trying
    # to cut down on the memory footprint of this object.
    
    __slots__ = ["created_dt", "sid", "amount", "volume", "price", "direction", "order_type", "_status"]

    # @expect_types(asset=Asset)
    def __init__(self, created_dt: int, sid: str, order_type: int, price: int=0, amount: int=0, volume: int=0):

        """
            @dt - datetime.datetime that the order was placed
            @asset - asset for the order.
            @amount - the number of shares to buy/sell
                    a positive sign indicates a buy
                    a negative sign indicates a sell
            @filled - how many shares of the order have been filled so far
            @order_type: market / price
            @direction: 0 (buy) / 1 (sell)
        """
        self.created_dt = created_dt
        self.sid = sid
        # 限价单 以等于或者低于限定价格买入 / 等于或者高于限定价格卖出订单 
        self.direction = math.copysign(1, volume)
        self.amount = amount
        self.volume = volume
        self.price = price
        self.order_type = OrderType.Market
        # get a string representation of the uuid.
        self.order_id = uuid.uuid4().hex
        self._status = OrderStatus.OPEN

    def on_status(self, status):
        self._status = status

    def on_measure(self, exec_price):
        """
            estimate volume threshold based on rate
        """
        if self.volume > 0:
            return self.volume
        else:
            increment =  200 if self.sid.startswith("688") else 100 
            per_value = increment * exec_price
            mul = np.floor(self.amount / per_value)
            return mul * increment


# @dataclass
class Transaction(BaseObject):

    __slots__ = ("sid", "created_dt", "price", "volume", "cost")
    
    def __init__(self, sid: str, created_dt: int, price: int, volume: int, cost: int) -> None:

        self.sid = sid
        self.created_dt = created_dt
        self.price = price
        self.volume = volume
        self.cost = cost


@dataclass(frozen=True)
class Commission:

    base_cost: int
    multiply: int

    def _on_benchmark(self, order: Order):
       # 印花税 1‰(卖的时候才收取，此为国家税收，全国统一)
        stamp_rate = 0 if order.direction== 1 else 1e-3
        # 过户费：深圳交易所无此项费用，上海交易所收费标准(按成交金额的0.02)
        transfer_rate = 2 * 1e-5 if order.sid.startswith('6') else 0
        # struct_date = datetime.datetime.fromtimestamp(transaction.created_dt).strftime("%Y%m%d")
        # 交易佣金：最高收费为3‰, 2015年之后万/3
        benchmark = 1e-4 if order.created_dt > pd.Timestamp('2015-06-09') else 1e-3
        # 完整的交易费率
        per_rate = stamp_rate + transfer_rate + benchmark * self.multiplier 
        return per_rate
    
    def on_rate(self, order):
        trade_rate = self._on_benchmark(order=order)
        return trade_rate
    
    def trade_calculate(self, rate, txn: Transaction):
        """
            :param order: Order object
            :return: cost for order
        """
        trading_cost = rate * txn.volume * txn.price
        validate_cost = trading_cost if trading_cost >= self.base_cost else self.base_cost
        return validate_cost
    

class Position(BaseObject):

    __slots__ = ("sid", "volume", "cost_basis", "freeze")

    def __init__(self, sid: str, volume: int, cost_basis: int, freeze: int=0):
        self.sid = sid
        self.volume = volume
        self.cost_basis = cost_basis
        self.freeze = freeze

    def _on_split(self, dividends):
        """
            股权登记日 ex_date
            股权除息日（为股权登记日下一个交易日）
            但是红股的到账时间不一致（制度是固定的）
            根据上海证券交易规则,对投资者享受的红股和股息实行自动划拨到账。股权息登记日为R日,除权息基准日为R+1日
            投资者的红股在R+1日自动到账,并可进行交易,股息在R+2日自动到帐
            其中对于分红的时间存在差异
            根据深圳证券交易所交易规则,投资者的红股在R+3日自动到账,并可进行交易,股息在R+5日自动到账
            持股超过1年税负5%;持股1个月至1年税负10%;持股1个月以内税负20%新政实施后,上市公司会先按照5%的最低税率代缴红利税

            update the postion by the split ratio and return the fractional share that will be converted into cash (除权）
        """
        volume_ratio = 1 + (dividends['bonus_share'] + dividends['transfer_share']) / 10
        self.volume = self.volume * volume_ratio
        self.cost_basis = self.cost_basis / volume_ratio
        cash_ratio = dividends['interest'] / 10
        bonus = self.volume * cash_ratio 
        return bonus
    
    def _on_right(self, rights):
        """
            register_date:登记日 ; ex_date:除权除息日 
            股权登记日后的下一个交易日就是除权日或除息日，这一天购入该公司股票的股东不再享有公司此次分红配股
            上交所证券的红股上市日为股权除权日的下一个交易日; 深交所证券的红股上市日为股权登记日后的第3个交易日
            price --- 配股价格 / ratio --- 配股比例
        """
        right_amount = self.volume * rights['ratio'] * rights["price"]
        right_volume = self.volume * (1 + rights["ratio"])
        self.cost_basis = (self.cost_basis * self.volume + right_amount) / right_volume
        self.volume = right_volume
        return 0

    def handle_event(self, event):
        if event.type == "split":
            left = self._on_split(event.data)
        else:
            left = self._on_right(event.data)
        return left

    def handle_transaction(self, txn: Transaction):
        """
            ZeroDivisionError异常和RuntimeWarning警告之间的区别 --- numpy.float64类型(RuntimeWarning) 0判断
            update position on transaction
        """
        p_volume = self.volume + self.freeze
        if np.sign(txn.volume) == -1:
            profit = (txn.price - self.cost_basis) * self.volume
            vol = p_volume + txn.volume
            self.cost_basis = self.cost_basis - profit / vol
            _to_cash = txn.price * txn.volume - txn.cost
        else:
            self.cost_basis = (p_volume * self.cost_basis + txn.volume * txn.price) / (p_volume + txn.volume)
            self.freeze = self.freeze + txn.volume
            _to_cash =  -txn.cost
        return _to_cash

    def sync_end_of_day(self, close):
        self.price = close
        # freeze to volume
        self.volume = self.volume + self.freeze
        self.freeze = 0


class Account(BaseObject):
    """
        The account object tracks information about the trading account. The
        values are updated as the algorithm runs and its keys remain unchanged.
        If connected to a broker, one can update these values with the trading
        account values as reported by the broker.
    """
    __slots__ = ("meta", "positions", "portfolio", "available")

    def __init__(self, meta, positions: Dict[str, Position] = {}, portfolio: int=0, available: int=0):
        # dataclass immutable
        self.positions = positions
        self.portfolio = portfolio
        self.available = available
        self.meta = meta

    def get_positions(self):
        # Position event push.
        # Position event of a specific vt_symbol is also pushed.
        p_obj = {p.sid for p in self.positions}
        return p_obj

    def process_trades(self, txns: List[Transaction]):
        for txn in txns:
            p = self.positions.get(txn.sid, Position(sid=txn.sid))
            _change = p.handle_transaction(txn)
            self.available = self.available + _change

    def process_events(self, events):
        """
            divdends and rights
            self.on_event(EVENT_POSITION, position)
        """
        reserve = 0
        for event in events:
            event_data = event.event_data
            reserve = reserve + self.positions[event_data["sid"]].handle_event(event=event_data)
        self.available = reserve + self.available

    def sync_end_of_day(self, sync_event):
        """
            sync close price at the end of session
        """
        prices = sync_event
        for p in self.positions.values():
            p.sync_end_of_day(sync_event.data[p.sid])
        # end_of_session
        portofolio = np.sum([sync_event.data[p.sid] * p.volume for p in self.positions])
        self.portfolio.loc[sync_event.session] = portofolio 
        self._cleanup_expired(sync_event.session)

    def _cleanup_expired(self, dt):
        """
            Clear out any assets that have expired before starting a new sim day.

            Finds all assets for which we have positions and generates
            close_position events for any assets that have reached their
            close_date.
        """
        # def past_close_date(asset):
        #     acd = asset.delist
        #     return acd is not None and acd == dt
    
        # # Remove positions in any sids that have reached their auto_close date.
        for p in self.positions.values():
            # if p.volume == 0 or past_close_date(p.sid):
            if p.volume == 0:
                self.positions.pop(p.sid)


if __name__ == "__main__":

    # asset
    asset = Asset("平安银行", 19910403, 0)
    print("asset ", asset)
    pct = asset.on_restricted(20080402)
    print("restricted pct ", pct)

    # # order
    order  = Order(created_dt=202404301130, sid=asset.sid, order_type="market", price=10, amount=2000, volume=1000)

    # transaction
    transaction = Transaction(sid=asset.sid, created_dt=202404301320, price=153, volume=1000, cost=134)

    # position
    position = Position(sid=asset.sid, volume=150, cost_basis=345, freeze=10) 

    # account
    meta = {"user_id": "test", "exp_id": "test_algo"}
    account = Account(meta=meta, positions={asset.sid: position}, portfolio=150, available=300)

    pdb.set_trace()
