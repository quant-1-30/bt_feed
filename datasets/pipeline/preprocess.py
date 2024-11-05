# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import pdb
import os
import re
from meta import ParamBase
from utils.registry import registry


@registry
class Restricted(ParamBase):
    
    params = (
        ("alias", "stock"),
        ("re", "^[6|0|3][0-9]{5}$"),
        ("sep", ".")
        # ("re", "^[15|51][0-9]{4}$"),
        )

    def on_handle(self, path):
        sid = os.path.basename(path).split(self.p.sep)[0][2:]
        m = re.match(self.p.re, sid)
        # m.group
        res= path if m else False
        return  res



class DatasetPath(ParamBase):

    def __init__(self, inner_dir):
        self.inner_dir = inner_dir

    def on_handle(self, data_root):
        absolute = os.path.expanduser(data_root)
        data_dirs = [os.path.join(absolute, sub_dir) for sub_dir in os.listdir(absolute) if not sub_dir.startswith(".")]
        data_paths = []
        for data_dir in data_dirs:
            complete_dir = os.path.join(data_dir, self.inner_dir)
            files = [os.path.join(complete_dir, file) for file in os.listdir(complete_dir) if not file.startswith(".")]
            data_paths.extend(files)
        return data_paths


class InstrumentPath(ParamBase):

    params = ()

    def on_handle(self, data_root):
        absolute = os.path.expanduser(data_root)
        data_paths = [os.path.join(absolute, file) for file in os.listdir(absolute) if not file.startswith(".")]
        return data_paths


@registry
class CalendarPath(ParamBase):

    params = ()

    def on_handle(self, data_root):
        absolute = os.path.expanduser(data_root)
        data_paths = [os.path.join(absolute, file) for file in os.listdir(absolute) if not file.startswith(".")]
        return data_paths
    

@registry
class AdjustmentPath(ParamBase):

    params = ()

    def on_handle(self, data_root):
        raise NotImplementedError()
    

@registry
class RightsPath(ParamBase):

    params = ()

    def build_path(self, data_root):
        raise NotImplementedError()
    
    def on_handle(self, data_root):
        raise NotImplementedError()
