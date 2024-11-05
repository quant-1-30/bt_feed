#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import pdb
from dataclasses import dataclass


class A(object):

    def __init__(self, a, b):
        self.a = a
        self.b = b


@dataclass(frozen=True)
class Transaction(object):

    # __slots__ = ("sid", "created_dt", "price", "volume", "cost")

    sid: str=""
    created_at: int=0
    price: int=0
    volume: int=0
    cost: int = 0.0


if __name__ == "__main__":

    t = {"a": 3, "b": 4}
    # 解包
    A(**t)
    print("A ", A)
    txn = Transaction()
    print("transaction ", txn)
    pdb.set_trace()
