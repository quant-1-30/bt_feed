# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import pydantic
from enum import Enum
from numpy import isfinite
from dataclasses import dataclass, field
from typing import List, Union, Tuple, Mapping, Any, Dict
from collections import namedtuple
from model import Order




def nop_context():
    pass


class Event(pydantic.BaseModel):
    """
        dividend / rights
    """
    type: str
    data: Any


class BrokerEvent(pydantic.BaseModel):

    line: Any
    order: Order


class ReqEvent(pydantic.BaseModel):

    rpc_type: str
    meta: Mapping[str, Any] = {}


class SyncEvent(pydantic.BaseModel):

    session: int
    data: Any


class LogEvent(pydantic.BaseModel):

    log_level: str
    log_meta: Any


# class Event(object):
#
#     def __init__(self, initial_values=None):
#         if initial_values:
#             self.__dict__.update(initial_values)
#
#     def keys(self):
#         return self.__dict__.keys()
#
#     def __eq__(self, other):
#         return hasattr(other, '__dict__') and self.__dict__ == other.__dict__
#
#     def __contains__(self, name):
#         return name in self.__dict__
#
#     def __repr__(self):
#         return "Event({0})".format(self.__dict__)
#
#     def to_series(self, index=None):
#         return pd.Series(self.__dict__, index=index)

# --- event manager 用于处理ledger(righted violated expired postion)


class EventManager(object):
    """Manages a list of Event objects.
    This manages the logic for checking the rules and dispatching to the
    handle_data function of the Events.

    Parameters
    ----------
    create_context : (BarData) -> context manager, optional
        An optional callback to produce a context manager to wrap the calls
        to handle_data. This will be passed the current BarData.
    """
    def __init__(self, create_context=None):
        self._events = []
        # _create_context CallbackManager
        self._create_context = (
            create_context
            if create_context is not None else
            lambda *_: nop_context
        )

    def add_event(self, event, prepend=False):
        """
        Adds an event to the manager.
        """
        if prepend:
            self._events.insert(0, event)
        else:
            self._events.append(event)

    # __enter__ __exit_ 调用 __call__
    def handle_data(self, context, data, dt):
        with self._create_context(data):
            for event in self._events:
                event.handle_data(
                    context,
                    data,
                    dt,
                )


# class Event(namedtuple('Event', ['rule', 'callback'])):
#     """
#     An event is a pairing of an EventRule and a callable that will be invoked
#     with the current algorithm context, data, and datetime only when the rule
#     is triggered.
#     """
#     # 实例之前(__init__) 调用__new__
#     def __new__(cls, rule, callback=None):
#         callback = callback or (lambda *args, **kwargs: None)
#         return super(cls, cls).__new__(cls, rule=rule, callback=callback)

#     def handle_data(self, context, data, dt):
#         """
#         Calls the callable only when the rule is triggered.
#         """
#         if self.rule.should_trigger(dt):
#             self.callback(context, data)
