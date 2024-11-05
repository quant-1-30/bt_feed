#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import pdb
import grpc
import signal
import aiohttp
import logging
import threading
from typing import Iterable
from concurrent.futures import ThreadPoolExecutor
from google.protobuf.json_format import MessageToDict
from google.protobuf import empty_pb2
from core.serialize.pb import service_pb2, service_pb2_grpc
from datefeed import data_feed


# def create_state_response(
#     call_state: service_pb2.Calendar,
# ) -> service_pb2.StreamCallResponse:
#     response = service_pb2.get_calendar()
#     response.Calendar = call_state
#     return response


class Simulator(service_pb2_grpc.btSimulatorServicer):

    def __init__(self):
        self._id_counter = 0
        self._lock = threading.RLock()

    # def _clean_call_session(self, call_info: service_pb2.CallInfo) -> None:
    #     logging.info("Call session cleaned [%s]", MessageToJson(call_info))
        
    def CalendarCall(
        self,
        request: empty_pb2.Empty,
        context: grpc.ServicerContext,
    ) -> service_pb2.Calendar: # type: ignore
        
        # try:
        #     request = next(request_iterator)
        #     logging.info(
        #         "Received a phone call request for number [%s]",
        #         request.phone_number,
        #     )
        # except StopIteration:
        #     raise RuntimeError("Failed to receive call request")
        
        # context.set_compression(grpc.Compression.NoCompression)
        logging.info("Received calendar")

        for key, value in context.invocation_metadata():
            print("Received initial metadata: key=%s value=%s" % (key, value))

        context.set_trailing_metadata(
            (
                ("checksum-bin", b"I agree"),
                ("retry", "false"),
            )
        )

        # context.add_callback(lambda: self._clean_call_session(call_info))
        trading_days = [c.trading_date for c in data_feed.trading_calendar] 
        response = service_pb2.Calendar()
        response.tz_info = "Asia/shanghai"
        response.date.extend(trading_days)
        print("calendar repsonse ", response)
        return response

    def InstrumentCall(
        self,
        request: service_pb2.QuoteRequest,
        context: grpc.ServicerContext,
    ) -> service_pb2.Calendar: # type: ignore
        
        logging.info("Received InstrumentCall %s" % request.SerializeToString())

        response = service_pb2.InstFrame()
        # context.add_callback(lambda: self._clean_call_session(call_info))
        instruments = data_feed.instruments
        if len(instruments):
            assets = [service_pb2.Instrument(**item) for item in instruments]
            response.assets.extend(assets)
            print("instrument repsonse ", response)
            print("InstrumentCall repsonse size ", response.ByteSize())
        return response
    
    def DatasetStreamCall(
        self,
        request: service_pb2.QuoteRequest,
        context: grpc.ServicerContext,
    ) -> service_pb2.TickerFrame: # type: ignore
        
        logging.info("Received dataset")
        req = MessageToDict(request, preserving_proto_field_name=True)
        response_iterator = data_feed.replay_dataset(req)
        # context.add_callback(lambda: self._clean_call_session(call_info))
        for resp in response_iterator:
            # pdb.set_trace()
            response = service_pb2.TickerFrame()
            response.ticker = resp.pop("utc")
            lines = service_pb2.Line(**resp)
            # lines = [service_pb2.Line(**item) for item in lines]
            response.line.extend([lines])
            print("dataset repsonse ", response)
            print("DatasetStreamCall ticker repsonse size ", response.ByteSize())
            yield response

    def AdjustmentStreamCall(
        self,
        request: service_pb2.QuoteRequest,
        context: grpc.ServicerContext,
    ) -> service_pb2.AdjFrame: # type: ignore
        
        # context.set_compression(grpc.Compression.NoCompression)
        logging.info("Received adjustment")
        req = MessageToDict(request, preserving_proto_field_name=True)
        response_iterator = data_feed.replay_adjustment(req)
        # context.add_callback(lambda: self._clean_call_session(call_info))
        for adjs in response_iterator:
            response = service_pb2.AdjFrame()
            response.date = adjs.pop("date")
            adjustments = service_pb2.Adjustment(**adjs)
            response.adj.extend([adjustments])
            print("adjustment repsonse ", response)
            print("AdjustmentStreamCall ticker repsonse size ", response.ByteSize())
            yield response

    def RightStreamCall(
        self,
        request: service_pb2.QuoteRequest,
        context: grpc.ServicerContext,
    ) -> service_pb2.RightmentFrame: # type: ignore
        
        logging.info("Received right")
        req = MessageToDict(request, preserving_proto_field_name=True)
        response_iterator = data_feed.replay_rights(req)
        # context.add_callback(lambda: self._clean_call_session(call_info))
        for rgts in response_iterator:
            response = service_pb2.RightmentFrame()
            response.date = rgts.pop("date")
            rights = service_pb2.Rightment(**rgts)
            response.rgt.extend([rights])
            print("rightment repsonse ", response)
            print("RightStreamCall ticker repsonse size ", response.ByteSize())
            yield response

    def OrderStreamCall(
        self,
        request: service_pb2.QuoteRequest,
        context: grpc.ServicerContext,
    ) -> service_pb2.TransactionFrame: # type: ignore
        
        logging.info("Received order")
        req = MessageToDict(request, preserving_proto_field_name=True)
        response_iterator = data_feed.replay_order(req)
        # context.add_callback(lambda: self._clean_call_session(call_info))
        for ord in response_iterator:
            response = service_pb2.OrderFrame()
            # pdb.set_trace()
            ord_obj = service_pb2.Order(**ord)
            response.ord.extend([ord_obj])
            print("order repsonse ", response)
            print("OrderStreamCall repsonse size ", response.ByteSize())
            yield response

    def TransactionStreamCall(
        self,
        request: service_pb2.TradeRequest,
        context: grpc.ServicerContext,
    ) -> service_pb2.TransactionFrame: # type: ignore
        
        logging.info("Received transaction")
        req = MessageToDict(request, preserving_proto_field_name=True)
        response_iterator = data_feed.replay_transaction(req)
        # context.add_callback(lambda: self._clean_call_session(call_info))
        for txn in response_iterator:
            response = service_pb2.TransactionFrame()
            # pdb.set_trace()
            txn_obj = service_pb2.Transaction(**txn)
            response.txn.extend([txn_obj])
            print("transaction repsonse ", response)
            print("TransactionStreamCall repsonse size ", response.ByteSize())
            yield response

    def ExperimentStreamCall(
        self,
        request: service_pb2.TradeRequest,
        context: grpc.ServicerContext,
    ) -> service_pb2.Experiment: # type: ignore
        
        logging.info("Received experiment")

        # # context.add_callback(lambda: self._clean_call_session(call_info))
        response_iterator = data_feed.replay_experiment(request=request)
        for records in response_iterator:
            response = service_pb2.Experiment(**records)
            print("experiment repsonse ", response)
            print("ExperimentStreamCall repsonse size ", response.ByteSize())
            # pdb.set_trace()
            yield response

    def AccountStreamCall(
        self,
        request: service_pb2.TradeRequest,
        context: grpc.ServicerContext,
    ) -> service_pb2.AccountFrame: # type: ignore
        
        logging.info("Received account")
        req = MessageToDict(request, preserving_proto_field_name=True)
        response_iterator = data_feed.replay_account(req)
        # context.add_callback(lambda: self._clean_call_session(call_info))
        for records in response_iterator:
            response = service_pb2.AccountFrame()
            account_meta = service_pb2.Account(**records)
            response.account.extend([account_meta])
            print("account repsonse ", response)
            print("AccountStreamCall repsonse size ", response.ByteSize())
            yield response

    def PersistStreamCall(
        self,
        request_iterator: Iterable[service_pb2.PersistRequest],
        context: grpc.ServicerContext,
    ) -> service_pb2.Status: # type: ignore
        # context.add_callback(lambda: self._clean_call_session(call_info))
        
        # try:
        #     request = next(request_iterator)
        #     logging.info(
        #         "Received a phone call request for number [%s]",
        #         request.phone_number,
        #     )
        # except StopIteration:
        #     raise RuntimeError("Failed to receive call request")
        logging.info("Received data which need to dump into database")
        for req_obj in request_iterator:
            # 网络传输base64编码替换 +与/ 特殊字符 返回bytes
            # bytes ---> base64 encode(bytes-like ---> bytes-like object)
            # base64 decode (str / bytes-like ---> bytes-like)
            req_map = MessageToDict(req_obj, preserving_proto_field_name=True)
            response_iterator = data_feed.on_persist(req_map)
            for resp in response_iterator:
                # pdb.set_trace()
                response = service_pb2.Status(**resp)
                print("PersistStreamCall repsonse size ", response.ByteSize(), response)
                yield response

  
def serve(address: str, MAX_MESSAGE_LENGTH=1024 * 1024 * 1024) -> None:
    """
    grpc.keepalive_time_ms: The period (in milliseconds) after which a keepalive ping is
        sent on the transport.
    grpc.keepalive_timeout_ms: The amount of time (in milliseconds) the sender of the keepalive
        ping waits for an acknowledgement. If it does not receive an acknowledgment within
        this time, it will close the connection.
    grpc.http2.min_ping_interval_without_data_ms: Minimum allowed time (in milliseconds)
        between a server receiving successive ping frames without sending any data/header frame.
    grpc.max_connection_idle_ms: Maximum time (in milliseconds) that a channel may have no
        outstanding rpcs, after which the server will close the connection.
    grpc.max_connection_age_ms: Maximum time (in milliseconds) that a channel may exist.
    grpc.max_connection_age_grace_ms: Grace period (in milliseconds) after the channel
        reaches its max age.
    grpc.http2.max_pings_without_data: How many pings can the client send before needing to
        send a data/header frame.
    grpc.keepalive_permit_without_calls: If set to 1 (0 : false; 1 : true), allows keepalive
        pings to be sent even if there are no calls in flight.
    For more details, check: https://github.com/grpc/grpc/blob/master/doc/keepalive.md
    """
    server_options = [
        ("grpc.keepalive_time_ms", 20000),
        ("grpc.keepalive_timeout_ms", 10000),
        ("grpc.http2.min_ping_interval_without_data_ms", 5000),
        ("grpc.max_connection_idle_ms", 10000),
        ("grpc.max_connection_age_ms", 30000),
        ("grpc.max_connection_age_grace_ms", 5000),
        ("grpc.http2.max_pings_without_data", 5),
        ("grpc.keepalive_permit_without_calls", 1),
        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
    ]

    server = grpc.server(ThreadPoolExecutor(), compression=grpc.Compression.Gzip, options=server_options)
    service_pb2_grpc.add_btSimulatorServicer_to_server(Simulator(), server)
    server.add_insecure_port(address)
    server.start()
    logging.info("Server serving at %s", address)
    server.wait_for_termination()


def handler(signum, frame):
    print("ctrl + c value", signal.SIGINT.value)
    sys.exit(0)


# ctrl + c
signal.signal(signal.SIGINT, handler)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    serve("[::]:50051")
