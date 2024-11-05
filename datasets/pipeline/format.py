#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import datetime
import pytz
import pdb
import pandas as pd
from typing import Any, Iterator, Union
from toolz import valmap
import numpy as np
import pdb
from utils.registry import registry
from meta import ParamBase


@registry
class Normalize(ParamBase):

    params = (
        ("alias", "normalize"),
        ("fields", ["dates", "sub_dates"])
        )

    def _transform(self, frame):
        frame = frame.to_dict()
        frame = valmap(lambda x: int(x), frame)
        year = frame['dates'] // 2048 + 2004
        month = (frame['dates'] % 2048) // 100
        day = (frame['dates'] % 2048) % 100
        hour = frame['sub_dates'] // 60
        minutes = frame['sub_dates'] % 60
        trans_date = datetime.datetime(year, month, day, hour, minutes)
        # pdb.set_trace()
        return trans_date
    
    def on_handle(self, frame: pd.DataFrame):
        if len(frame):
            assert "dates" in frame.columns, "missing dates column"
            trans_dates = frame.loc[:, self.p.fields].apply(lambda x: self._transform(x), axis=1)
            frame.loc[:, "trans_date"] = trans_dates
            # pdb.set_trace()
        return frame

    def __repr__(self):
        format_string = "format: %s" % self.__class__.__name__
        return format_string


@registry
class UTC(ParamBase):

    params = (
        ("alias", "utc"),
        ("local_tz", "Asia/Shanghai")
        )

    def canonical_time(self, ts: int):
        if isinstance(ts, (int, float)):
            pass
        elif isinstance(ts, str):
            pass
        else:
            raise TypeError("")
        
    def _transform(self, dt: Union[datetime.datetime, str, datetime.datetime.timestamp]):
        # if not dt.tzinfo:
        #     dt = dt.replace(tzinfo=pytz.timezone(tz))
        # return dt.replace(tzinfo=pytz.utc) 
        tz = pytz.timezone("UTC")
        # pdb.set_trace()
        if isinstance(dt, datetime.datetime):
            local_dt = dt.tz_localize(tz=self.p.local_tz)
        elif isinstance(dt, datetime.datetime.timestamp):
            local_dt = datetime.datetime.fromtimestamp(dt, tz=self.p.local_tz)
        else:
            local_dt = datetime.datetime.strptime(dt, self.p.format)
        # utc_timestamp = utc_dt.astimezone(tz=tz).timestamp() 
        utc_dt = local_dt.astimezone(tz=tz)
        # pdb.set_trace()
        return utc_dt.timestamp()
    
    def on_handle(self, frame: pd.DataFrame) -> Any:
        """
            Converts a UTC tz-naive timestamp to a tz-aware timestamp.
            Normalize a time. If the time is tz-naive, assume it is UTC.
            Drop the nanoseconds field. warn=False suppresses the warning
            that we are losing the nanoseconds; however, this is intended.
            return pd.Timestamp(ts.to_pydatetime(warn=False), tz='UTC')
        """
        if len(frame):
            frame.loc[:, "utc"] = frame["trans_date"].apply(lambda x: self._transform(x))
        return frame

    def __repr__(self):
        format_string = "format: %s" % self.__class__.__name__
        return format_string


@registry
class Date2Int(ParamBase):

    params = (
        ("fields", ("trading_date", "first_trading", "delist", "register_date", "ex_date", "effective_date")),
        ("sep", ['/', '-', '*', '.', '~', '"', '^', '#'])
        # ("re", '[/|-|*|.|~|"|^|#]')
    )
    
    def _trans_dt(self, dt):
        sep = [s for s in self.p.sep if s in str(dt)]
        assert len(sep) <= 1, f"{sep} has at least one sep"
        if len(sep):
            dt = int(str(dt).replace(sep[0], ""))
        return dt

    def on_handle(self, frame: pd.DataFrame):
        if len(frame):
            cols = list(set(frame.columns) & set(self.p.fields))
            if cols:
                frame.loc[:, cols] = frame.loc[:, cols].map(lambda x: self._trans_dt(x))
        #         frame.loc[:, cols].replace(self.p.re, "", regex=True, inplace=True)
        return frame


# from numpy import (
#     bool_,
#     dtype,
#     float32,
#     float64,
#     int32,
#     int64,
#     int16,
#     uint16,
#     ndarray,
#     uint32,
#     uint8,
# )

# BOOL_DTYPES = frozenset(
#     map(dtype, [bool_, uint8]),
# )
# FLOAT_DTYPES = frozenset(
#     map(dtype, [float32, float64]),
# )
# INT_DTYPES = frozenset(
#     # NOTE: uint64 not supported because it can't be safely cast to int64.
#     map(dtype, [int16, uint16, int32, int64, uint32]),
# )
# DATETIME_DTYPES = frozenset(
#     map(dtype, ['datetime64[ns]', 'datetime64[D]']),
# )
# # We use object arrays for strings.
# OBJECT_DTYPES = frozenset(map(dtype, ['O']))
# STRING_KINDS = frozenset(['S', 'U'])
