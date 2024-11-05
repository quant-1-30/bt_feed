# /usr/bin/env python3
# -*- coding: utf-8 -*-
import uuid
import pickle
from functools import lru_cache
from sim.meta import Ledger
from typing import List
from sim.schema.event import ReqEvent, BrokerEvent, Event, SyncEvent, LogEvent
from sim.meta import BtBroker
from sim.async_client import AsyncStreamClient


class TradeApi(object):
    """
    # How to implement a tradeApi:
    ---
    ## Basics
    A tradeApi should satisfies:
    * this class should be thread-safe:
        * all methods should be thread-safe
        * no mutable shared properties between objects.
    * all methods should be non-blocked
    * satisfies all requirements written in docstring for every method and callbacks.
    * automatically reconnect if connection lost.

    All the XxxData passed to callback should be constant, which means that
        the object should not be modified after passing to on_xxxx.
    So if you use a cache to store reference of data, use copy.copy to create a new object
    before passing that data into on_xxxx
    """
    __slots__ = ("fund",)

    def __init__(self, addr, broker_params={}, default_fund=200000):

        self.broker = BtBroker(broker_params)
        self.ledger = Ledger()
        self.async_client = AsyncStreamClient(addr)
        self.fund = default_fund
        
        # self.trade_username = "***"
        # self.trade_password = "***"
        # self.trade_token = ""

    def _on_persist(self, meta):
        req = ReqEvent(rpc_header="persist", meta=meta)
        status = self.async_client.run(req)
        return status

    def _on_query(self, table, meta):
        req = ReqEvent(rpc_type=table, meta=meta)
        resp = self.async_client.run(req)
        return resp[0]

    def on_login(self, user_id, exp_id):
        """
            user_id exp_id ---> account_id ---> account_data (latest record) ---> ledger
        """
        exp_meta = {"user_id": user_id, "experiment_id": exp_id} 
        account_id = self._on_query(meta=exp_meta)
        if account_id:
            resp = self._on_query("account", meta={"account_id": account_id})
            exp_meta["account_id"] = account_id
            print("resp", resp) 
        else:
            exp_meta["account_id"] = str(uuid.uuid4()) 
            meta = {"body": {"experiment": exp_meta}}
            self._on_persist(meta=meta)
            resp = [{"positions":0, "portfolio": 0, "cash": self.fund}]

        # update account
        resp[0].update({"metadata": exp_meta})
        self.ledger.restore(resp[0])

    def on_event(self, event: Event) -> None:

        self.ledger.on_events(events=event)

    def on_trade(self, broker_event: BrokerEvent) -> None:

        trades = self.broker.on_impl(broker_event)
        self.ledger.on_trade(trades)

        # order serialize
        order_data = pickle.dumps(broker_event.order)
        trade_meta = {"experiment": self.ledger.account_obj.metadata, 
                      "body": {"transaction": [order_data, trades]}}
        status = self._on_persist(meta=trade_meta)
        return status
    
    def sync_on_date(self, sync_event: SyncEvent) -> None:

        self.ledger.on_sync(sync_event)
        body = self.ledger.account_obj.serialize()
        meta = {"experiment": self.ledger.account_obj.metadata, 
                "body": {"account": body}}
        status = self._on_persist(meta=meta)
        return status 
    
    def on_log(self, log_event: LogEvent) -> None:
        """
        Log event push.
        self.event_engine.put(event)
        """

    def on_diconnect(self):
        """
            stop tradeApi
        """


if __name__ == "__main__":

        # client
        addr = ("localhost", 10000)
        trade_api = TradeApi(addr=addr)
        # ut args
        user_name = "hengxin"
        phone = 13776668123
        # trade_api.on_register(user_name=user_name, phone=phone)
        user_id = "d0a6c7ae-0b48-4b0e-ba19-2bc80c0c7dca"
        experiment_id = "test"
        trade_api.on_login(user_id=user_id, experiment_id=experiment_id)
        fund = 100000
        date=20240424

        # data = {"table": "user_account", "data": [{"name": "test", "fullname": "liuhengxin", "phone": 13776668122}]}

        # positions = {"sid": "600001", "cost": 15.6, "volume": 100}
        # data = {"table": "account", "data": [{"date": "20240233", "positions": json.dumps(positions), "portfolio": 100000, "cash": 50000}]}
        # # data = {"table": "transaction", "data": [{"sid": "600001", "created_at": 20240322, "match_price": 13, "ticker_price": 12, "volume": 50000, "cost": 100}]}
        # body = {"meta": json.dumps(data).encode("utf-8")}
        # req = {"user_id": 1, "algo_id": "test_api", "body": body, "type": "trade"}
        # message = pickle.dumps(req)
        # print("message", message)
        # # atexit.register(exit)