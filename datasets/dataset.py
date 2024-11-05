#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pdb
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures import as_completed
from meta import ParamBase
from utils.io import build_from_cfg
from utils.loader import get_module_by_module_path
from datasets.pipeline import *
from ._provider import _Providers


class Dateset(ParamBase):
    """Client Provider

    Requesting data from server as a client. Can propose requests:

        - Calendar : Directly respond a list of calendars
        - Instruments (without filter): Directly respond a list/dict of instruments
        - Instruments (with filters):  Respond a list/dict of instruments
        - Features : Respond a cache uri

    The general workflow is described as follows:
    When the user use client provider to propose a request, the client provider will connect the server and send the request. The client will start to wait for the response. The response will be made instantly indicating whether the cache is available. The waiting procedure will terminate only when the client get the response saying `feature_available` is true.
    `BUG` : Everytime we make request for certain data we need to connect to the server, wait for the response and disconnect from it. We can't make a sequence of requests within one connection. You can refer to https://python-socketio.readthedocs.io/en/latest/client.html for documentation of python-socketIO client.
    """
    params = (
        ("alias", "dataset"),
    )

    @classmethod
    def doinit(cls):
        for k, v in _Providers.items():
            setattr(cls, k, v)
    
    def _build_dataset(self, cfg_module, type_name):
        # setup dataset path
        type_module = cfg_module.__dict__[type_name]
        path_module = build_from_cfg(type_module["path_module"])
        data_paths = path_module.on_handle(type_module["root_path"])
        # setup pipeline
        pipeline = []
        for trans in type_module["pipeline"]:
            transform = build_from_cfg(trans)
            pipeline.append(transform) 
        # execute
        for p in data_paths:
            print("data_path ", p)
            for transform in pipeline:
                print("transform ", transform)
                p = transform.on_handle(p)
        return p

    #    def _mp_execute(res):
    #        for transform in transforms:
    #            res = transform(res)
    #        return res
    #    
    #    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
    #        tasks = []
    #        for p in data_paths:
    #            f = pool.submit(_mp_execute, p)
    #            tasks.append(f)
    #        for t in as_completed(tasks):
    #            flag = t.result()
    #            print("future status ", flag)
    
    def on_handle(self, *args, **kwargs):
        raise NotImplementedError("__BaseProvider offer on_handler method")
    

class BtDataset(Dateset):

    cfg_cache = {}

    params = (
        ("alias", "dataset_tdx"),
        ("type", ["calendar", "instrument", "ticker", "adjustment", "right"]),
        ("sep", "_"),
        ("orm_cache", {})
    )

    @property
    def calendar(self):
        return self.cal.calendar
    
    @classmethod
    def get_cfg(cls, cfg_path):
        basename = os.path.basename(cfg_path)
        if cfg_path not in cls.cfg_cache:
            cls.cfg_cache[basename] = get_module_by_module_path(cfg_path)
        return cls.cfg_cache[basename]
    
    @property
    def instruments(self):
        return self.inst.instruments

    def build_dataset(self, cfg_path, dataset_type):
        assert dataset_type in self.p.type, ValueError(f"illega {dataset_type}")
        cfg_module = self.get_cfg(cfg_path=cfg_path)
        self._build_dataset(cfg_module, f"{dataset_type}_dataset")

    def _cache_orm(self, table):
        if table not in self.p.orm_cache:
            self.p.orm_cache[table] = OrmWriter(table)
        return self.p.orm_cache[table] 

    def on_handle(self, metadata):
        for table, data in metadata.items():
            table_obj = self._cache_orm(table=table)
            status = table_obj.on_handle(data)
            # pdb.set_trace()
            yield status
