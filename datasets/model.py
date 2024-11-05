#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import numpy as np
from pydantic import BaseModel, Field
from typing import List
from functools import total_ordering


class Interval(BaseModel):

    start: int = Field(default=-np.inf)
    end: int = Field(..., gt=0)

    def serialize(self):
        return [self.start, self.end]


@total_ordering
class Request(BaseModel):

    range: Interval
    sids: List[str]=[]

    def serialize(self) -> str:
        return {"range": self.range, "sids": self.sids}

    def __lt__(self, other):
        return True if max(other.range) <= max(self.range) else False
    
    def __repr__(self) -> str:
        # __str__ / __repr__ ; print 默认调用__str__  ; 如果__str__没有重写返回__repr__
        pass
