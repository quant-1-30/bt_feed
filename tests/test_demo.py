#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pdb
import pandas as pd

def trans2csv(path, engine):
    # data = pd.read_excel(path, engine=engine, dtype={"A股代码": "str"})
    data = pd.read_excel(path, engine=engine, dtype={"证券代码": "str"})
    pdb.set_trace()
    data.to_csv()
    


if __name__ == "__main__":

    # path = "/Users/hengxinliu/Downloads/quant/assets/raw/GPLIST.xls"
    # path = "/Users/hengxinliu/Downloads/quant/assets/raw/ZZSSLIST.xls"
    # trans2csv(path, engine="xlrd")
    # path = "/Users/hengxinliu/Downloads/quant/assets/raw/A股列表.xlsx"
    path = "/Users/hengxinliu/Downloads/quant/assets/raw/终止上市.xlsx"
    trans2csv(path, engine="openpyxl")