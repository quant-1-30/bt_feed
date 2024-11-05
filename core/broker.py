
import datetime
import pickle
import itertools
import numpy as np
from numpy.random import default_rng
from textwrap import dedent
from typing import Any, List
from schema.model import Transaction, Commission, Order
from utils.dt_utilty import calc_distance, loc2ticker
# from six import with_metaclass
from schema.event import BrokerEvent
from meta import ParamBase


class Analog(ParamBase):
    """
        Analogy level 1 (3 s) tick stock data future 
        level1 250/ms
    """
    params = (
        ("level", 1),
    )
        
    @staticmethod
    def on_align(analogy, snapshot):
        analogy[0] = snapshot[0]
        analogy[-1] = snapshot[-1]
        m_idx = np.argmax(analogy)
        analogy[m_idx] = np.max(snapshot)
        n_idx = np.argmin(analogy)
        analogy[n_idx] = np.min(snapshot)
        return analogy 
    
    def downgrade(self, datas):
        raise NotImplementedError("240/m ticker to 4800 3/s ticker")
    

class BetaAnalog(Analog):

    params = (
        ("alias", "beta"),
        ("size", 20),
        ("prior", {"a":1, "b": 2})
    )

    def downgrade(self, tick_desc):
        """
            tick_desc: np.array[o,h,l,c] e.g. minute snapshot
        """
        rng = default_rng()
        probs = rng.beta(a=self.p.prior["a"], b=self.p.prior["b"], size=self.p.size)
        delta = np.max(tick_desc) - np.min(tick_desc)
        samples = delta * np.array(probs) + np.min(tick_desc)
        # align
        datas = self.on_align(samples, tick_desc)
        return datas


class BtBroker(ParamBase):
    """
        impact_factor:  measure sim_vol should less than real tick volume
        slippage_factor: sim_price should multiplier real tick price
        epsilon: measure whether price reaches on limit always on the trading_day
        bid_mechnism: execute on time or price
    """
    params = (
        ("level", 1),
        # eager --- True 最后接近收盘时候集中将为成交的订单成交按照市价撮合成交保持最大持仓
        ("eager", True),
        ("frequency", "minute"),
        ("delay", 2),
        ("impact_factor", 0.2),
        ("slippage_factor", 0.01),
        ("epsilon", 0.001),
        # commission
        ("base_cost", 5),
        ("multiply", 5),
    )

    def __init__(self, kwargs):
        # exchanges: List[Exchange] = []
        self.frequency = kwargs.pop("frequency", self.p.frequency)
        self.delay = kwargs.pop("delay", self.p.delay)
        self.impact_factor = kwargs.pop("impact_factor", self.p.impact_factor) 
        self.slippage_factor = kwargs.pop("slippage_factor", self.p.slippage_factor) 
        self.analog = kwargs.pop("analog", "BetaAnalog") 
        self.restricted = kwargs.pop("epsilon", self.p.restricted) 
        self.commission = Commission(self.p.base_cost, self.p.multiply)

    def on_restricted(self, daily):
        """
            订单设立可成交区间，涨跌幅有限制
            high - low < epison 由于st数据无法获取, 关于sid restricted 无法准确, 通过high / low 振幅判断
        """
        low = np.min(np.array([item["low"] for item in daily]))
        high = np.max(np.array([item["high"] for item in daily]))
        pct = (high - low)/low
        status = True if pct <= self.p.epsilon else False
        return status 

    def sample(self, lines):
        """
            analogy 3/s tick price with 1/m price
        """
        vols = [item["volume"] for item in lines]
        tick_prices = [self.analog.downgrade(tick, level=1)  for tick in lines]
        tick_vols = [self.analog.downgrade(vol, level=1)  for vol in vols]
        price_arrays = np.array(itertools.chain(*tick_prices))
        vol_arrays = np.array(itertools.chain(*tick_vols)) 
        return price_arrays, vol_arrays
    
    def _on_market(self, order, ticks):
        # locate index of created_dt
        seconds, format_dt = calc_distance(order.created_dt)
        loc = int(np.ceil(seconds/3))

        # slippage_price
        rate = self.commission.on_rate(order)
        sim_price = ticks[0][loc + self.delay] * ( 1 + order.direction * self.slippage_factor) * (1 + rate)
        exec_vol = order.on_measure(sim_price)

        # estimate volume account on market_impact
        sim_vol = ticks[1][loc + self.delay] * self.p.impact_factor
        filled, recursive = (exec_vol, True) if exec_vol <= sim_vol else (sim_vol, False)

        # create transaction
        created_dt = format_dt + datetime.timedelta(seconds=self.p.delay) 
        cost = self.commission.trade_calculate(rate, transaction)
        transaction = Transaction(sid=order.sid, 
                                  created_at=created_dt,
                                  price=sim_price, 
                                  volume=filled, 
                                  cost=cost)
        if recursive:
            recur_amount = order.amount - filled * sim_price
            dt = datetime.datetime(year=format_dt.year, month=format_dt.month, day=format_dt.day, hour=14, minute=57)
            recur_order = Order(created_dt==dt, sid=order.sid, direction=order.direction, order_type=4, amount=recur_amount)
            return transaction, recur_order
        return transaction, ""

    def _on_price(self, order, ticks, rate):
        # locate index according by price
        locs = np.argwhere(ticks[0] <= order.price) if order.direction ==1 else np.argwhere(ticks[0] >= order.price)
        if len(locs) and locs[0]:
            # commission rate
            rate = self.commission.on_rate(order)
            # locate price by loc
            sim_price = ticks[0][locs[0][0]] * (1 + rate)
            # estimate volume
            exec_vol = order.on_measure(sim_price)

            # within market_impact
            sim_vol = ticks[1][locs[0][0] + self.delay] * self.impact_factor
            filled, recursive = (exec_vol, False) if exec_vol <= sim_vol else (sim_vol, True)

            # create transaction
            ticker = loc2ticker(order.created_dt, locs[0][0])
            created_dt = ticker + datetime.timedelta(seconds=self.p.delay) 
            cost = self.commission.trade_calculate(rate, transaction)
            transaction = Transaction(sid=order.sid, 
                                      created_at=created_dt,
                                      price=sim_price,
                                      volume=filled,
                                      cost=cost) 
            if recursive:
                recur_vol = exec_vol - sim_vol
                dt = datetime.datetime.strptime(order.created_dt, "%Y%m%d") + datetime.timedelta(hour=14, minute=57)
                recur_order = Order(created_dt==dt, sid=order.sid, direction=order.direction, order_type=4, volume=recur_vol)
                return transaction, recur_order
            return transaction, ""
        return "", ""
        
    def on_impl(self, broker_event: BrokerEvent) -> None:
        """
        # orders to transactions
        Order event push.
        Order event of a specific vt_orderid is also pushed.
        """
        order = pickle.loads(broker_event.order)
        restricted = self.on_restricted(order)
        output = []
        if not restricted:
            _sample_tick = self.sample(BrokerEvent.line)
            # estimate order volume 
            if order.order_type == 4:
                txn, sub_order = self._on_market(order, _sample_tick)
            else:
                txn, sub_order = self._on_price(order, _sample_tick)
            if txn:
                output.append(txn) 
            # np.logical_and(txn.match_price>=restricted.channel[0], txn.match_price<=restricted.channel[1]):
            if self.p.eager and sub_order:
                eager_txn, _ = self._on_market(order, _sample_tick)
                output.append(eager_txn)
        return output

 
class Ledger(object):
    """
        the ledger tracks all orders and transactions as well as the current state of the portfolio and positions
        position_tracker ( process_execution ,handle_splits , handle_dividend ) 
    """
    __slots__ = ("account_obj",)

    def __init__(self):
        
        self.account_obj = object

    def restore(self, account_meta):

        self.account_obj = pickle.dumps(account_meta) 

    def on_events(self, events):

        self.account_obj.process_events(events)

    def on_trade(self, transactions):
        """
        Account event push.
        Account event of a specific vt_accountid is also pushed.
        """
        self.account_obj.process_trades(transactions)
    
    def on_sync(self, sync_event):
        """
            sync close price on positions
            Clear out any assets that have expired and positions which volume is zero before starting a new sim day.
            Finds all assets for which we have positions and generates
            close_position events for any assets that have reached their
            close_date.
        """
        self.account_obj.sync_end_of_day(sync_event)
