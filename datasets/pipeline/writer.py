#!/usr/bin/env python3
# -*- coding: utf-8; py-indent-offset:4 -*-

# from __future__ import (absolute_import, division, print_function,
#                         unicode_literals)
import pdb
import collections
from collections.abc import Iterable
import io
import sys
import avro.schema
import pandas as pd
from typing import Any
from avro.io import DatumWriter
from avro.datafile import DataFileWriter
from sqlalchemy.orm import Session
from sqlalchemy import text
from meta import ParamBase
from utils.registry import registry
from datasets.schema import builder


try:  # For new Python versions
    collectionsAbc = collections.abc  # collections.Iterable -> collections.abc.Iterable
except AttributeError:  # For old Python versions
    collectionsAbc = collections  # Используем collections.Iterable


@registry
class AvroWriter(ParamBase):
    # import avro.schema
    # from avro.datafile import DataFileReader, DataFileWriter
    # from avro.io import DatumReader, DatumWriter

    # schema = avro.schema.parse(open("user.avsc", "rb").read())

    # writer = DataFileWriter(open("users.avro", "wb"), DatumWriter(), schema)
    # writer.append({"name": "Alyssa", "favorite_number": 256})
    # writer.append({"name": "Ben", "favorite_number": 7, "favorite_color": "red"})
    # writer.close()

    # reader = DataFileReader(open("users.avro", "rb"), DatumReader())
    # for user in reader:
    #     print (user)
    # reader.close()
    
    def on_handle(self, schema_path: str, data_path: str, frame: pd.DataFrame) -> Any:
        # ticker.avsc
        schema = avro.schema.parse(open(schema_path, "rb").read())
        # users.avro 
        writer = DataFileWriter((open(data_path), "wb"), DatumWriter(), schema)
        dicts = frame.to_dict()
        for ele in dicts.values:
            # ele --- {"name": "Ben", "favorite_number": 7, "favorite_color": "red"}
            writer.append(ele)
        writer.close()


@registry
class OrmWriter(ParamBase):

    params = (
        ("alias", "postgres"), 
        ("engine", "psycopg"), 
        ("host", "localhost"),
        ("port", "5432"),
        ("db", "bt2live"),
        ("user", "postgres"),
        ("pwd", "20210718"),
        ("pool_size", 100),
        ("max_overflow", 10),
        ("pool_pre_ping", True),
        ("echo", True)
        )
    
    def __init__(self, table):
        self.table = table

    @classmethod
    def doinit(cls):
        builder(cls)

    def _execute(self, data):
        if self.table not in self.tables:
            raise IOError(f"{self.table} is not valid in {self.tables.keys()}")
        table_instance = self.tables[self.table]

        if isinstance(data, pd.DataFrame):
            inserts = list(data.T.to_dict().values())
            # Iterable 可迭代对象 __iter__ 使用for / Iterator 迭代器 __iter__ , __next__ yield
        elif isinstance(data, Iterable):
            inserts = data
        else:
            inserts = [data]
        # pdb.set_trace()
        # with Session(self.engine) as session:
        with self.engine.connect() as connection:
            connection.execute(table_instance.insert(), inserts)
            connection.commit()
    
    def on_handle(self, data):
        # pdb.set_trace()
        # rollback
        if len(data):
            try:
                self._execute(data)
                status = {"status": 0, "error": ""}
            except Exception as e:
                print(e)
                status = {"status": 1, "error": str(e)}
        return status


@registry
class CsvWriter(ParamBase):

    '''The system wide writer class.
    It can be parametrized with:

      - ``out`` (default: ``sys.stdout``): output stream to write to

        If a string is passed a filename with the content of the parameter will
        be used.

        If you wish to run with ``sys.stdout`` while doing multiprocess optimization, leave it as ``None``, which will
        automatically initiate ``sys.stdout`` on the child processes.

      - ``csv`` (default: ``False``)

        If a csv stream of the data feeds, strategies, observers and indicators
        has to be written to the stream during execution

        Which objects actually go into the csv stream can be controlled with
        the ``csv`` attribute of each object (defaults to ``True`` for ``data
        feeds`` and ``observers`` / False for ``indicators``)

      - ``csv_filternan`` (default: ``True``) whether ``nan`` values have to be
        purged out of the csv stream (replaced by an empty field)

      - ``csv_counter`` (default: ``True``) if the writer shall keep and print
        out a counter of the lines actually output

      - ``indent`` (default: ``2``) indentation spaces for each level

      - ``separators`` (default: ``['=', '-', '+', '*', '.', '~', '"', '^',
        '#']``)

        Characters used for line separators across section/sub(sub)sections

      - ``seplen`` (default: ``79``)

        total length of a line separator including indentation

    '''
    params = (
        ('out', None),
        ('indent', 2),
        ("headers", ""),
        ("separator", ",")
    )

    def _start_output(self):
        # open file if needed
        if self.p.out is None:
            out = sys.stdout
        elif isinstance(self.p.out, str):
            out = open(self.p.out, 'w')
        else:
            out = self.p.out
        return out
    
    def __init__(self):
        self.out = self._start_output

    def stop(self):
        self.out.close()

    def _writeline(self, line):
        self.out.write(line + '\n')

    def writelineseparator(self):
        if self.p.headers:
            headers = self.p.headers.split(self.p.separator)
            sep = ' ' * self.p.indent + self.p.separator
            csv_header = sep.join(headers)
            self._writeline(csv_header)

    def on_handle(self, lines):
        if lines:
            self.writelineseparator()
            for l in lines:
                self._writeline(l + '\n')
            self.stop() 

        
@registry
class WriterStringIO(ParamBase):

    params = (
        ('out', io.StringIO),
        )

    def __init__(self):
        super(WriterStringIO, self).__init__()

    def stop(self):
        # super(WriterStringIO, self).stop()
        # Leave the file positioned at the beginning
        self.out.seek(0)

    def on_handle(self, *args, **kwargs):
        super().on_handle(*args, **kwargs)
