#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import asyncio
import sys
import signal
import pdb
import pickle
import pydantic
import socket
import threading
import atexit
import socketserver
from sim._internal.rpc_client import rpc_client
from sim.schema.serialize.pb import service_pb2


class MyTCPHandler(socketserver.BaseRequestHandler):
    """
    This class works similar to the TCP handler class, except that
    self.request consists of a pair of data and client socket, and since
    there is no connection the client address must be given explicitly
    when sending data back via sendto().
    """
    chunk_size = 1024

    def __init__(self, request, client_address, server):
        super().__init__(request, client_address, server)

    def handle(self):
        # self.request is the TCP socket connected to the client
        data = self.request.recv(1024).strip()
        req_map = pickle.loads(data)
        rpc_type = req_map.pop("rpc_type")
        meta = req_map["meta"]
        meta["experiment"] = service_pb2.Experiment(meta["experiment"]) if "experiment" in meta else service_pb2.Experiment() 
        if rpc_type == "persist":
            meta["body"]= {k: json.dumps(v).encode("utf-8") for k, v in meta["body"].items()} 
            request = service_pb2.PersistRequest(**meta)
        else:
            request = service_pb2.TradeRequest(**meta)
        # cur_thread = threading.current_thread()
        # response = bytes("{}: {}".format(cur_thread.name, data), 'ascii')
        print("request", request)
        response_iterator = rpc_client.on_delegate(request=request, rpc_type=rpc_type)
        print("recevied from {}:".format(self.client_address[0]))
        checksum = bytes("sentinel", 'utf-8') 
        for res in response_iterator:
            serialize = pickle.dumps(res)
            for idx in range(0, len(serialize), self.chunk_size):
                # send 返回成功的字节大小, sendall重复send
                self.request.sendall(serialize[idx: idx+self.chunk_size])
            self.request.sendall(checksum)
            print("send serialize ", len(serialize), serialize)
        self.request.sendall(bytes("shutdown", 'utf-8'))
        print("send shutdown")


def quit_handler(server):
    server.socket.close()

def on_handler(signum, frame):
    print("ctrl + c handler and value", signal.SIGINT.value)
    sys.exit(0)


# ctrl + c
signal.signal(signal.SIGINT, on_handler)


if __name__ == "__main__":
    HOST, PORT = "localhost", 10000
    # with socketserver.TCPServer((HOST, PORT), MyTCPHandler) as server:

    with socketserver.ThreadingTCPServer((HOST, PORT), MyTCPHandler, bind_and_activate=False) as server:
        # SOL_SOCKET 通用的
        server.allow_reuse_address = True
        server.server_bind()
        server.server_activate()
        # server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.serve_forever()
        atexit.register(quit_handler, server)