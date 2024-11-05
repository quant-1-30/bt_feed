#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pdb
import socket
import asyncio
import pickle
from queue import Queue


class AsyncStreamClient(object):

    # asyncio.open_connection() --- socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
    # getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)

    def __init__(self, addr):
        self.buffer = 1024
        self.host, self.port = addr

    async def get_data(self, req):
        reader, writer = await asyncio.open_connection(host=self.host, port=self.port)
        message = pickle.dumps(req.model_dump())
        print(f'Send: {message!r}')
        writer.write(message)
        await writer.drain()

        chunks=b""
        while True:
            # recv 面向连接 
            recv_message = await reader.read(self.buffer)
            print("recv_message", recv_message)
            chunks = chunks + recv_message 
            if recv_message == b"sentinel": 
                try:
                    received = pickle.loads(chunks)
                    yield received
                except Exception as e:
                    print("error", e)
                print("Received: {}".format(len(received)))
                chunks = b""
            elif recv_message == b"shutdown":
                print('Close the connection')
                writer.close()
                await writer.wait_closed()
                break

    async def _generator(self, req):
        
        result = []
        async for data in self.get_data(req):
            print("data", data)
            if data:
                result.append(data)
        return result

    def on_exit(self):
        asyncio

    def run(self, req):
        raw = asyncio.run(self._generator(req))
        return raw


class AsyncDatagramClient(object):
        
        # transport sendto / abort
        # transport.sendto(message, self.addr)
        # sock = transport.get_extra_info("socket")
        # transport.close()
        # Resource temporarily unavailable need sleep or pdb

    def __init__(self, addr):
        self.buffer = 1024
        self.addr = ('127.0.0.1', 9999)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) 

    async def _received(self, message):
       # As you can see, there is no connect() call; UDP has no connections.
       # Instead, data is directly sent to the recipient via sendto().
       # sock.sendto(bytes(message + "\n", "utf-8"), (host, port))
       # Get a reference to the event loop as we plan to use
       # low-level APIs.
        loop = asyncio.get_running_loop()
        transport, protocol = await loop.create_datagram_endpoint(asyncio.DatagramProtocol, remote_addr=self.addr)

        # Send the message
        await loop.sock_sendto(self.sock, message, self.addr)
        
        chunks = b""
        while True:
            # recvfrom 无连接 
            recv_message, _ = await loop.sock_recvfrom(self.sock, self.buffer)
            # pdb.set_trace()
            print(f"Received: {recv_message}")
            chunks = chunks + recv_message 
            if recv_message == b"sentinel": 
                try:
                    received = pickle.loads(chunks)
                    yield received
                except Exception as e:
                    print("error", e)
                print("Received: {}".format(len(received)))
                chunks = b""
            elif recv_message == b"shutdown":
                break
            print("not nil", len(chunks))

    async def _generator(self, req):
        # Get a reference to the event loop as we plan to use
        # low-level APIs.
        datas = []
        message = pickle.dumps(req.model_dump())
        async for data in self._received(message):
            print("data", data)
            datas.append(data)
        return datas
    
    def on_exit(self):
        self.sock.close()

    def run(self, req):
        resp = asyncio.run(self._generator(req))
        return resp


# class EchoClientProtocol:
    
#     def __init__(self, fut):
#         self.fut = fut
#         self.transport = None

#     def connection_made(self, transport):
#         self.transport = transport
#         print('Send:', self.message)
#         self.transport.sendto(self.message.encode())

#     def datagram_received(self, data, addr):
#         print("Received:", data.decode())
#         self.fut.set_result((data, addr))

#         print("Close the socket")
#         self.transport.close()

#     def error_received(self, exc):
#         print('Error received:', exc)

#     def connection_lost(self, exc):
#         print("Connection closed")
#         self.on_con_lost.set_result(True)
