#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pdb
import pytz
import datetime
import json
import base64
import pandas as pd
from typing import Dict,Any
from meta import SingletonMeta
from datasets import BtDataset, Request


# class DataFeed(object):

#     def notify_data(self, data, status, *args, **kwargs):
#         '''Receive data notifications in cerebro

#         This method can be overridden in ``Cerebro`` subclasses

#         The actual ``*args`` and ``**kwargs`` received are
#         implementation defined (depend entirely on the *data/broker/store*) but
#         in general one should expect them to be *printable* to allow for
#         reception and experimentation.
#         '''
#         pass

#     def adddata(self, data, name=None):
#         '''
#         Adds a ``Data Feed`` instance to the mix.

#         If ``name`` is not None it will be put into ``data._name`` which is
#         meant for decoration/plotting purposes.
#         '''
#         if name is not None:
#             data._name = name

#         data._id = next(self._dataid)
#         data.setenvironment(self)

#         self.datas.append(data)
#         self.datasbyname[data._name] = data
#         feed = data.getfeed()
#         if feed and feed not in self.feeds:
#             self.feeds.append(feed)

#         if data.islive():
#             self._dolive = True

#         return data

#     def chaindata(self, *args, **kwargs):
#         '''
#         Chains several data feeds into one

#         If ``name`` is passed as named argument and is not None it will be put
#         into ``data._name`` which is meant for decoration/plotting purposes.

#         If ``None``, then the name of the 1st data will be used
#         '''
#         dname = kwargs.pop('name', None)
#         if dname is None:
#             dname = args[0]._dataname
#         d = bt.feeds.Chainer(dataname=dname, *args)
#         self.adddata(d, name=dname)

#         return d

#     def rolloverdata(self, *args, **kwargs):
#         '''Chains several data feeds into one

#         If ``name`` is passed as named argument and is not None it will be put
#         into ``data._name`` which is meant for decoration/plotting purposes.

#         If ``None``, then the name of the 1st data will be used

#         Any other kwargs will be passed to the RollOver class

#         '''
#         dname = kwargs.pop('name', None)
#         if dname is None:
#             dname = args[0]._dataname
#         d = bt.feeds.RollOver(dataname=dname, *args, **kwargs)
#         self.adddata(d, name=dname)
#         return d

#     def _datanotify(self):
#         for data in self.datas:
#             for notif in data.get_notifications():
#                 status, args, kwargs = notif
#                 self._notify_data(data, status, *args, **kwargs)
#                 for strat in self.runningstrats:
#                     strat.notify_data(data, status, *args, **kwargs)

#     def _notify_data(self, data, status, *args, **kwargs):
#         for callback in self.datacbs:
#             callback(data, status, *args, **kwargs)

#         self.notify_data(data, status, *args, **kwargs)
        


class DataFeed(metaclass=SingletonMeta):

    def __init__(self):
        self.bt = BtDataset()

    @property
    def trading_calendar(self):
        return self.bt.cal.calendar
    
    @property
    def instruments(self):
        return self.bt.inst.instruments
    
    def build_dataset(self, cfg_path, dataset_type):
        self.bt.build_dataset(cfg_path, dataset_type)

    def replay_asset(self, request: Request):
        insts = [asset for asset in self.instruments if asset.first_trading >= request.range.start and 
                 asset.delist >= request.range.end ]
        return insts
    
    def replay_dataset(self, request: Request):
        return self.bt.dataset.get_datasets(request)
    
    def replay_adjustment(self, request: Request):
        adjustmenst = self.bt.adj.get_dividends(request)
        return adjustmenst

    def replay_rights(self, request: Request):
        rights = self.bt.right.get_rights(request)
        return rights
    
    def replay_experiment(self, request: Request):
        exps = self.bt.experiment.get_experiments(request)
        return exps

    def replay_transaction(self, request):
        """
            transactions
        """
        txn_records = self.bt.txn.get_transactions(request) 
        return txn_records
         
    def replay_order(self, request):
        """
            orders
        """
        txn_records = self.bt.txn.get_transactions(request) 
        return txn_records
    
    def replay_account(self, request):
        """
            transactions and account
        """
        account_info = self.bt.account.get_account(request) 
        return account_info

    def on_persist(self, metadata: Dict[str, Any]):
        """
            persist transactions and account detail into database
        """
        # 网络传输base64编码替换 +与/ 特殊字符 返回bytes
        # bytes ---> base64 encode(bytes-like ---> bytes-like object)
        # base64 decode (str / bytes-like ---> bytes-like)
        body = metadata.pop("body")
        decode = {k: base64.b64decode(v) for k, v in body.items()}
        decode = {k: json.loads(v.decode("utf-8")) for k, v in decode.items()}
        return self.bt.on_handle(decode)


data_feed = DataFeed()


# if __name__ == "__main__":

#     # test transaction
#     _gateway.replay_experiment(request={})
