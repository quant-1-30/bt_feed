#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import numpy as np
from pydantic import BaseModel, Field
from typing import List
from dataclasses import dataclass
from functools import total_ordering


@dataclass(eq=False)
class Calendar:

    trading_date: str

    def serialize(self) -> str:
        return {"trading_date": self.trading_date}


@dataclass(eq=False)
class Instrument:

    sid: str
    name: str
    first_trading: str
    delist: str

    def serialize(self) -> dict:
        return {"sid": self.sid, "name": self.name, "first_trading": self.first_trading, "delist": self.delist}
    

@dataclass(frozen=True)
@total_ordering
class Line:

    sid: str
    utc: int
    open: int
    high: int
    low: int
    close: int
    volume: int
    amount: int

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, self):
            raise TypeError
        return self.sid == __value.sid and self.ticker == __value.ticker
    
    def __lt__(self, other):
        if not isinstance(other, self):
            raise TypeError
        return self.ticker > other.ticker

    def serialize(self) -> dict:

        return {"sid": self.sid, "utc": self.utc, "open": self.open, "high": self.high, 
                "low": self.low, "close": self.close, "volume": self.volume, "amount": self.amount}
    

@dataclass(frozen=True, order=True)
class Experiment:

    user_id: str
    account_id: str
    experiment_id: str

    def serialize(self) -> dict:
        return {"user_id": self.user_id, "account_id": self.account_id, "experiment_id": self.experiment_id}


@dataclass(frozen=True, order=True)
class Adjustment:

    sid: str
    register_date: int
    ex_date: int
    stock_bonus: int = 0
    bonus: int = 0

    def serialize(self) -> dict:
        return {"sid": self.sid, "register_date": self.register_date, "ex_date": self.ex_date, 
                "stock_bonus": self.stock_bonus, "bonus": self.bonus}


@dataclass(frozen=True, order=True)
class Rightment:

    sid: str
    register_date: int
    ex_date: int
    effective_date: int
    price: int
    ratio: int

    def serialize(self) -> dict:
        return {"sid": self.sid, "register_date": self.register_date, "ex_date": self.ex_date, 
                "effective_date": self.effective_date, "price": self.price, "ratio": self.ratio}


@dataclass(frozen=True, order=True)
class Order:

    order_id: str
    sid: int
    order_type: int
    created_dt: int
    order_price: int
    order_volume: int

    def serialize(self) -> dict:
        return {"order_id": self.order_id, "sid": self.sid, "order_type": self.order_type, 
                "created_at": self.created_at, "order_price": self.order_price, "order_volume": self.order_volume}
    

@dataclass(frozen=True, order=True)
class Transaction:

    sid: str
    created_at: int
    trade_price: int
    market_price: int
    volume: int
    cost: int

    def serialize(self) -> dict:
        return {"sid": self.sid, "created_at": self.created_at, "trade_price": self.trade_price, 
                "market_price": self.market_price, "volume": self.volume, "cost": self.cost}


@dataclass(frozen=True, order=True)
class Account:

    date: str
    positions: str
    portfolio: int
    cash: int

    def serialize(self) -> dict:
        return {"date": self.date, "positions": self.positions, 
                "portfolio": self.portfolio, "cash": self.cash}
