# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import pdb
import bisect
import numpy as np
import pandas as pd
import datetime
from sqlalchemy import select
from sqlalchemy.orm import Session
from typing import List, Union, Optional, Dict
from meta import with_metaclass, MetaBase
from utils.cache import lazyproperty
from .model import *
from .serialize import *
from .schema import builder


class _BaseProvider(with_metaclass(MetaBase, object)):
    """Local provider class
    It is a set of interface that allow users to access data.
    Because PITD is not exposed publicly to users, so it is not included in the interface.

    To keep compatible with old qlib provider.
    """
    params = (
        ("host", "localhost"),
        ("port", "5432"),
        ("user", "postgres"),
        ("pwd", "20210718"),
        ("db", "bt2live"),
        ("engine", "psycopg"),
        ("pool_size", 20),
        ("max_overflow", 10),
        ("pool_pre_ping", True),
        ("echo", True)
    )

    @classmethod
    def doinit(cls):
        builder(cls)

    def on_handle(cls, *args, **kwargs):
        raise NotImplementedError("__BaseProvider offer on_handler method")
    
    @staticmethod
    def trans2Req(message: Dict[str, Union[str, int]]):
        interval = Interval(start=message["start_date"], end=message["end_date"])
        sids = message.get("sid", [])
        req = Request(range=interval, sids=sids)
        return req

    def __len__(self):
        raise NotImplementedError("length")
    
    def __getitem__(self, index):
        raise NotImplementedError("getitem")
    

class _CalendarProvider(_BaseProvider):
    """Calendar provider base class

    Provide calendar data.
    """
    params = (
        # ("alias", "calendar"),
        ("table", "calendar"),
        ("opens", ("9:30", "13:00")),
        ("closes", ("11:30", "15:00"))
    )

    @lazyproperty
    def calendar(self):
        """
            Get calendar of certain market in given time range. 
        """
        calendar = []
        cal = self.tables[self.p.table]
        with Session(bind=self.engine) as session:
            # stmt = select(cal).execution_options(**self.options)
            stmt = select(cal.c.trading_date)
            for trading_dt in session.scalars(stmt).yield_per(10):
            # for trading_dt in session.execute(stmt).yield_per(10):
            # for trading_dt in session.query(cal).all():
                calendar.append(Calendar(trading_dt))
        return calendar
    
    def __len__(self):
        return len(self.calendar)
    
    def __getitem__(self, index):
        return self.calendar[index]
    
    def get_range(self, request: Request):
        """Get calendar of certain market in given time range.

        Parameters
        ----------
        start_time : str
            start of the time range.
        end_time : str
            end of the time range.
        freq : str
            time frequency, available: year/quarter/month/week/day.
        future : bool
            whether including future trading day.

        Returns
        ----------
        list
            calendar list
        """
        start_time, end_time = request.range.serialize()
        if start_time == "None":
            start_time = None
        if end_time == "None":
            end_time = None
        # strip
        if start_time:
            start_time = pd.Timestamp(start_time)
            if start_time > self.calendar[-1].trading_date:
                return np.array([])
        else:
            start_time = self.calendar[0]
        if end_time:
            end_time = pd.Timestamp(end_time)
            if end_time < self.calendar[0].trading_date:
                return np.array([])
        else:
            end_time = self.calendar[-1]
        _, _, si, ei = self.locate_index(start_time, end_time)
        return self.calendar[si : ei + 1]

    def locate_index(self, start_time, end_time):
        """Locate the start time index and end time index in a calendar under certain frequency.

        Parameters
        ----------
        start_time : pd.Timestamp
            start of the time range.
        end_time : pd.Timestamp
            end of the time range.
        Returns
        -------
        pd.Timestamp
            the real start time.
        pd.Timestamp
            the real end time.
        int
            the index of start time.
        int
            the index of end time.
        """
        start_time = pd.Timestamp(start_time.trading_date)
        end_time = pd.Timestamp(end_time.trading_date)
        calendar_index = {x.trading_date: i for i, x in enumerate(self.calendar)}
        if start_time not in calendar_index:
            try:
                start_time = self.calendar[bisect.bisect_left(self.calendar, start_time)]
            except IndexError as index_e:
                raise IndexError(
                    "`start_time` uses a future date, if you want to get future trading days, you can use: `future=True`"
                )
        start_index = calendar_index[start_time]
        if end_time not in calendar_index:
            end_time = self.calendar[bisect.bisect_right(self.calendar, end_time) - 1]
            # loc = np.searchsorted(data["date"], cur_time_int, side="right")
        end_index = calendar_index[end_time]
        return start_time, end_time, start_index, end_index

    def get_trading_tickers(self, date: Union[str, int], freq="M"):
        dt = datetime.datetime.strptime(str(date), "%Y-%m-%d")
        intervals = zip(self.p.opens, self.p.closes)
        tickers = []
        for interval in intervals:
            open = dt + pd.Timedelta(interval[0])
            close = dt + pd.Timedelta(interval[1])
            ranges = pd.date_range(open, close, freq=freq, inclusive="left")
            tickers.extend(list(ranges))
        return tickers

    def on_handle(cls, *args, **kwargs):
        return super().on_handle(*args, **kwargs)


class _InstrumentProvider(_BaseProvider):
    """Instrument provider base class

    Provide instrument data.
    """
    params = (
        ("alias", "assets"),
        ("table", "instrument")
    )

    @lazyproperty
    def instruments(self):
        """List the overall instruments.

        Returns
        -------
        dict or list
            instruments list or dictionary with time spans
        """
        assets = list()
        inst = self.tables[self.p.table]
        with Session(self.engine) as session:
            # stmt = select(inst).execution_options(**self.options)
            stmt = select(inst)
            # for sid in self.session.scalars(stmt).yield_per(10):
            for sid in session.execute(stmt).yield_per(10):
                assets.append(Instrument(*sid[1:]).serialize())
        return assets

    def __len__(self):
        return len(self.instruments)
    
    def __getitem__(self, sid):
        item = [sid for sid in self.instruments if sid.sid == sid]
        return item[0]

    def get_instruments(self, message: Dict[str, Union[str, int]]):
        """Get the existiongconfig dictionary for a base market adding several dynamic filters.
        """
        req = self.trans2Req(message)
        _instruments = [sid for sid in self.instruments if sid.first_trading_date >= req.range.start 
                        and sid.first_trading_date <= req.range.end]
        _instruments = [sid for sid in _instruments if sid.delist > req.range.end]
        return _instruments
    
    def on_handle(cls, *args, **kwargs):
        return super().on_handle(*args, **kwargs)
    
    
class _DatasetProvider(_BaseProvider):
    """Dataset provider class
    Provide Dataset data.
    """
    params = (
        ("alias", "dataset"),
        ("table", "minute")
    )

    def get_datasets(self, message: Dict[str, Union[str, int]]):
        """Get dataset data.

        Parameters
        ----------
        request:  DataRequest

        Returns
        ----------
        pd.DataFrame
            a pandas dataframe with <instrument, datetime> index.
        """
        req = self.trans2Req(message)
        dataset = self.tables[self.p.table]
        with Session(self.engine) as session:
            # stmt = select(dataset).where(dataset.utc.between(*request.range)).execution_options(**self.options)
            stmt = select(dataset).where(dataset.c.tick.between(*req.range.serialize()))
            # session.scalars(stmt).yield_per(10):
            for line in session.execute(stmt).yield_per(10):
                meta = Line(*line[1:]).serialize()
                yield meta
    
    def on_handle(cls, *args, **kwargs):
        return super().on_handle(*args, **kwargs)


class _AdjustmentProvider(_BaseProvider):
    """
        Calendar provider base class
        Provide calendar data.
    """
    params = (
        ("alias", "dividends"),
        ("table", "adjustment")
    )

    def get_dividends(self, message: Dict[str, Union[str, int]]):
        """Get dvidends of certain asset in given time range.

        Parameters
        ----------
        request : Request
            start of the time range.

        Returns
        ----------
        """
        req = self.trans2Req(message)
        adj = self.tables[self.p.table]
        with Session(self.engine) as session:
            stmt = select(adj).where(adj.c.ex_date.between(*req.range.serialize()))
            for data in session.execute(stmt).yield_per(10):
                meta = Adjustment(*data[1:]).serialize()
                yield meta 

    def on_handle(cls, *args, **kwargs):
        return super().on_handle(*args, **kwargs)


class _RightsProvider(_BaseProvider):
    """
        Calendar provider base class
        Provide calendar data.
    """
    params = (
        ("alias", "rights"),
        ("table", "rightment")
    )

    def get_rights(self, message: Dict[str, Union[str, int]]):
        """Get dvidends of certain asset in given time range.

        Parameters
        ----------
        request : Request
            start of the time range.

        Returns
        ----------
        """
        req = self.trans2Req(message)
        rgt = self.tables[self.p.table]
        with Session(self.engine) as session:
            stmt = select(rgt).where(rgt.c.ex_date.between(*req.range.serialize()))
            for data in session.execute(stmt).yield_per(10):
                meta = Rightment(*data[1:]).serialize()
                yield meta 
    
    def on_handle(cls, *args, **kwargs):
        return super().on_handle(*args, **kwargs)


class _ExperimentProvider(_BaseProvider):
    """
        Algo provider base class
    """
    params = (
        ("alias", "exp"),
        ("table", "experiment")
    )

    def get_experiments(self, message: Dict[str, Union[str, int]]):
        """Get dvidends of certain asset in given time range.

        Parameters
        ----------
        request : Request
            start of the time range.

        Returns
        ----------
        """
        exp = self.tables[self.p.table]
        with Session(self.engine) as session:
            # session.no_autoflush = True
            stmt = select(exp)
            for data in session.execute(stmt).yield_per(10):
                print("_ExperimentProvider data ", data)
                meta = Experiment(*data[1:]).serialize()
                yield meta
            # in case select data is null
            yield {}
        
    def on_handle(cls, *args, **kwargs):
        return super().on_handle(*args, **kwargs)
    

class _OrderProvider(_BaseProvider):
    """
        order provider base class
    """
    params = (
        ("alias", "ord"),
        ("table", "order")
    )

    def get_orders(self, message: Dict[str, Union[str, int]]):
        """Get dvidends of certain asset in given time range.

        Parameters
        ----------
        request : Request
            start of the time range.

        Returns
        ----------
        """
        req = self.trans2Req(message)
        ord = self.tables[self.p.table]
        with Session(self.engine) as session:
            stmt = select(ord).where(ord.c.created_at.between(*req.range.serialize()))
            if "account_id" in message:
                stmt = stmt.where(ord.experiment.account_id==message["account_id"])
            for data in session.execute(stmt).yield_per(10):
                meta = Order(*data[1:]).serialize()
                yield meta 
    
    def on_handle(cls, *args, **kwargs):
        return super().on_handle(*args, **kwargs)
    

class _TxnProvider(_BaseProvider):
    """
        Calendar provider base class
        Provide calendar data.
    """
    params = (
        ("alias", "txn"),
        ("table", "transaction")
    )

    def get_transactions(self, message: Dict[str, Union[str, int]]):
        """Get dvidends of certain asset in given time range.

        Parameters
        ----------
        request : Request
            start of the time range.

        Returns
        ----------
        """
        req = self.trans2Req(message)
        txn = self.tables[self.p.table]
        with Session(self.engine) as session:
            stmt = select(txn).where(txn.c.created_at.between(*req.range.serialize()))
            if "order_id" in message:
                stmt = stmt.where(txn.order.order_id==message["order_id"])
            if "account_id" in message:
                stmt = stmt.where(txn.experiment.account_id==message["account_id"])
            for data in session.execute(stmt).yield_per(10):
                meta = Transaction(*data[1:]).serialize()
                yield meta 
    
    def on_handle(cls, *args, **kwargs):
        return super().on_handle(*args, **kwargs)
    

class _AccountProvider(_BaseProvider):
    """
        Calendar provider base class
        Provide calendar data.
    """
    params = (
        ("alias", "account"),
        ("table", "account")
    )

    def get_account(self, message: Dict[str, Union[str, int]]):
        """Get dvidends of certain asset in given time range.

        Parameters
        ----------
        request : Request
            start of the time range.

        Returns
        ----------
        """
        req = self.trans2Req(message)
        account = self.tables[self.p.table]
        with Session(self.engine) as session:
            stmt = select(account).where(account.c.date.between(*req.range.serialize()))
            if "account_id" in message:
                stmt = stmt.where(account.experiment.account_id==message["account_id"])
            for data in session.execute(stmt).yield_per(10):
                meta = Account(*data[1:]).serialize()
                yield meta 
    
    def on_handle(cls, *args, **kwargs):
        return super().on_handle(*args, **kwargs)


_Providers = dict(
    (("cal", _CalendarProvider()),
    ("inst", _InstrumentProvider()),
    ("dataset", _DatasetProvider()),
    ("adjument", _AdjustmentProvider()),
    ("right", _RightsProvider()),
    ("experiment", _ExperimentProvider()),
    ("order", _OrderProvider()),
    ("transaction", _TxnProvider()),
    ("account", _AccountProvider())))
