# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import os
import logging
import duckdb
import threading
import json
import asyncio
import queue
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from .utils import schema_range, preprocess_req 

from rpc_feed.utils.wrapper import singleton


class ConnectionPoolExhausted(RuntimeError):
    """Raised when no DuckDB connection becomes available within the timeout."""


class ConnectionPool:
    def __init__(self, db_path, max_connections):
        self.db_path = db_path
        self.max_connections = max_connections
        self._pool = queue.Queue(max_connections)
        self._initialize_pool()
    
    def _initialize_pool(self):
        """初始化连接池"""
        for _ in range(self.max_connections):
            conn = duckdb.connect(self.db_path)
            self._init_connection(conn)
            self._pool.put(conn)
    
    def _init_connection(self, conn):
        """load plugins"""
        try:
            conn.execute("INSTALL httpfs;")  # downloads from network on first use
        except Exception as e:  # offline / already installed locally
            logging.warning("DuckDB INSTALL httpfs failed offline or maybe already installed, trying LOAD only: %s", e)
        conn.execute("LOAD httpfs;")
        conn.execute("SET enable_object_cache=true;")
        # Parquet TIMESTAMP(tz) vs naive TIMESTAMP comparison resolves the naive side
        # in the session timezone -> pin it so the query window never drifts with host TZ
        conn.execute("SET TimeZone='UTC';")
        # conn.execute("SET max_expression_depth = 2000;") # not UNION ALL

    def get_connection(self, timeout=5):
        try:
            return self._pool.get(timeout=timeout)
        except queue.Empty:
            raise ConnectionPoolExhausted(
                f"DuckDB connection pool exhausted ({self.max_connections} conns, "
                f"timeout={timeout}s); raise DUCKCONNECTION or lower MAX_CONCURRENT_STREAMS"
            )
    
    def return_connection(self, conn):
        try:
            self._pool.put(conn, timeout=1)
        except queue.Full:
            logging.warning("DuckDB connection pool full, closing connection")
            conn.close()
            
    def close_all(self):
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except queue.Empty:
                break


@singleton
class DuckDBManager:
    """
    - avoid Macro / View and Catalog
    - Glob + Hive Partitioning
    - Parameters Binding solve AST Depth Limit
    """
    # StopIteration not direct passed to run_in_executor Future replace sentinel 
    # asyncio "StopIteration interacts badly with generators and cannot be raised into a Future"
    _EXHAUSTED = "sentinel"

    def __init__(self):
        self.dataset_root = Path(os.getenv("DUCKDATASET")).expanduser()
        self.batch_size = int(os.getenv("DUCKBATCHSIZE", 100000))

        # benchmark must be matched FIRST and anchored: b"399006" also matches
        # the stock pattern ^(6|0|3)\d{5} and would be silently routed to a
        # nonexistent stock/ partition (empty result, no error)
        self.regex_rules = {
            "benchmark": re.compile(b"^(1A0001|1B0688|2A01|399006)$"), # 上证/科创/深证/创业
            "stock": re.compile(b"^(6|0|3)\d{5}"), # encode('ascii')
            "fund": re.compile(b"^(51|15|16)\d{4}"),
        }

        max_connections = int(os.getenv("DUCKCONNECTION", 10))
        # bound concurrent parquet scans to the pool size so the N+1th stream
        # queues instead of raising "pool exhausted" (stream semaphore may be larger)
        self._query_sema = asyncio.Semaphore(max_connections)
        cache_path = Path(__file__).resolve().parent / "cache" / os.getenv("DUCKDB")
        self.connection_pool = ConnectionPool(cache_path, max_connections)

    async def __aenter__(self):
        return self
        
    def _glob_path(self, req: dict) -> list:
        ranges = schema_range(req) 
        exact_globs =[]
        
        for sid_bytes in req["sid"]:

            # Determine category based on regex matching
            type_name = None
            for name, regex in self.regex_rules.items():
                if regex.match(sid_bytes):
                    type_name = name
                    break
                    
            if not type_name:
                continue 
                
            sid_str = sid_bytes.decode("utf-8")
            for y, q, ym in ranges:
                dir_path = os.path.join(
                    self.dataset_root, 
                    type_name, 
                    f"year={y}", 
                    f"quarter={q}", 
                    f"sid={sid_str}", 
                    f"date={ym}"
                )
                
                if os.path.exists(dir_path): # os cache
                    exact_globs.append(os.path.join(dir_path, "*.parquet"))
        return exact_globs
    
    @staticmethod
    def _read_batch_safe(reader):
        try:
            return reader.read_next_batch()
        except StopIteration:
            return DuckDBManager._EXHAUSTED

    async def query(self, req: dict, raw_template: str):
        loop = asyncio.get_running_loop()
        async with self._query_sema:
            # _glob_path stats one dir per sid*month -> keep the stat storm off the event loop
            file_globs = await loop.run_in_executor(None, self._glob_path, req)
            if not file_globs:
                logging.info("DuckDB query: no file globs")
                return

            sql_meta = preprocess_req(req)
            sids = sql_meta["sids"]
            if not sids:
                return

            conn = self.connection_pool.get_connection()
            try:
                # conn.execute / fetch_record_batch / read_next_batch sync api block event_loop
                reader = await loop.run_in_executor(
                    None,
                    lambda: conn.execute( # C++ Parameter Binding
                        raw_template,
                        [file_globs, sids, sql_meta["start_str"], sql_meta["end_str"]]
                    ).fetch_record_batch(self.batch_size)
                )

                while True:
                    # read_next_batch sync api block
                    batch = await loop.run_in_executor(None, self._read_batch_safe, reader)
                    if batch is self._EXHAUSTED:
                        break
                    yield batch
            finally:
                self.connection_pool.return_connection(conn)
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False


_duck_inst = None
_duck_lock = threading.Lock()


def get_duckdb_manager():
    global _duck_inst
    with _duck_lock:
        if _duck_inst is None:
            _duck_inst = DuckDBManager()
    return _duck_inst