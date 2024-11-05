#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from typing import Any


# descriptor
class lazyproperty:
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            value = self.func(instance)
            setattr(instance, self.func.__name__, value)
            return value


class CachProperty(object):
    """
    Parameters
    ----------
    disk_cache : int
        whether to skip(0)/use(1)/replace(2) disk_cache

    This function will try to use cache method which has a keyword `disk_cache`,
    and will use provider method if a type error is raised because the DatasetD instance
    is a provider class.
    """

    def __init__(self, func):
        self.func = func

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        func_name = self.func.__name__
        if not hasattr(self, func_name):
            res = self.func(*args, **kwds)
            setattr(self, func_name, res)
        return getattr(self, func_name)