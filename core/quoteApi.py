# /usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import atexit
import signal
import datetime
import signal
from utils.dt_utilty import market_utc, date2utc
from core.schema.event import Event, ReqEvent
from core.async_client import AsyncDatagramClient


class QuoteApi(object):
    """
        udp client 
    """
    def __init__(self, addr):
        # provider_uri
        self.async_client = AsyncDatagramClient(addr=addr)
        # validate and ssl
        # self.quote_username = "***"
        # self.quote_password = "***"
        # self.quote_token = ""

    def onSubCalendar(self):
        end_date = datetime.datetime.now().strftime("%Y%m%d")
        meta={"start_date": 19900101, "end_date": int(end_date)}
        req = ReqEvent(rpc_type="calendar", meta=meta)
        calendars = self.async_client.run(req)
        # event
        event = Event(event_type="calendar", data=calendars)
        return event
    
    def onSubAsset(self, s_date=None, e_date=None, sid=[]):
        # validate args
        s_date = s_date if s_date else 19900101
        e_date = e_date if e_date else datetime.datetime.now().strftime("%Y%m%d")
        meta = {"start_date": int(s_date), "end_date": int(e_date), "sid": sid}
        req = ReqEvent(rpc_type="instrument", meta=meta)
        intruments = self.async_client.run(req)
        # event
        event = Event(event_type="asset", data=intruments)
        return event

    def onSubTickData(self, date, sid=[]):
        """
            tick data
        """
        s, e = market_utc(date)
        meta = {"start_date": int(s.timestamp()), "end_date": int(e.timestamp()), "sid": sid}
        req = ReqEvent(rpc_type="dataset", meta=meta)
        ticks = self.async_client.run(req)
        # event
        event = Event(event_type="tick", data=ticks)
        return event
    
    def OnSubDatasets(self, s_date, e_date, sid):
        """
            history datasets
        """
        s = date2utc(s_date)
        e = date2utc(e_date)
        meta = {"start_date": int(s.timestamp()), "end_date": int(e.timestamp()), "sid": sid}
        req = ReqEvent(rpc_type="dataset", meta=meta)
        datasets = self.async_client.run(req)
        # event
        event = Event(event_type="dataset", data=datasets)
        return event

    def onSubEvent(self, s_date, e_date, event_type, sid=[]):
        """
            adjs / rights
        """
        meta = {"start_date": int(s_date), "end_date": int(e_date), "sid": sid}
        req = ReqEvent(rpc_type=event_type, meta=meta)
        datas = self.async_client.run(req)
        # event
        event = Event(event_type="event", data=datas)
        return event
    
def on_handler(signum, frame):
    print("ctrl + c handler and value", signal.SIGINT.value)
    sys.exit(0)


# def exit_handle(quote):
#     print("handler at exit")
#     quote.onDisconnect()


# ctrl + c
signal.signal(signal.SIGINT, on_handler)

# # set alarm
# signal.signal(signal.alarm, handler)
# signal.alarm(5)



if __name__ == "__main__":

        # client
        addr = ("127.0.0.1", 9999)
        quoteApi = QuoteApi(addr=addr)
        # atexit.register(exit_handle, quoteApi)
        calendars = quoteApi.onSubCalendar()
        print("calendars", calendars)
        assets = quoteApi.onSubAsset()
        print("assets", assets)
        tick_datas = quoteApi.onSubTickData(date=20211015)
        print("tick_datas", tick_datas)
        datasets = quoteApi.OnSubDatasets(s_date=20211001, e_date=20211109, sid=['600000'])
        print("datasets", datasets)
        adjustments = quoteApi.onSubEvent(start_date=20000301, end_date=20240401, event_type="adjustment")
        print("adjustments", adjustments)
        rightments = quoteApi.onSubEvent(start_date=20000301, end_date=20240401, event_type="rightment")
        print("rightsments", rightments)

