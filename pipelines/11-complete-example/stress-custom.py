import asyncio
import logging
import pathlib
import sys
import threading
import aiohttp
import time

import pandas as pd
from aiohttp import ClientSession, ClientConnectorError
import random
import math
# import plot_builder as pb

class StressGenerator:
    def __init__(self, url, is_dataset, data_path = None):
        self.final_data = []
        self.url = url
        self.is_dataset = is_dataset
        self.data_path = data_path
        self.idx = 0



    def nextParam():
        return random.randint(100, 500)
        return generator()



    def nextTimeReqeust(rateParameter):
        return -math.log(1.0 - random.random()) / rateParameter


    async def fetch(self, session: ClientSession, index, **kwargs):
        start_time = int(round(time.time() * 1000))
        try:
            resp = await session.post(url=self.url,
                                        json={"data":{"ndarray":[[1.0, 2.0, 5.0, 6.0]]}},
                                        ssl=False, **kwargs)
            end_time = int(round(time.time() * 1000))
            self.final_data.append((index, resp.status, 1, start_time, end_time, end_time - start_time))
        except ClientConnectorError:
            return (self.url, 404)
        return (self.url, resp.status)


    async def make_request(self, idx, **kwargs):
        headers = {
                'Content-Type': 'application/json',
                
            }
        async with ClientSession(headers=headers,trust_env=True) as session:
            tasks = []
            result = await fetch(session, idx, **kwargs)
            await session.close()


    async def generator(url, index, **kwargs):
        await asyncio.gather(make_request(url, index))


    def send_request_thread(idx):
        asyncio.run(generator(idx))

    def run():
        seconds = str(time.ctime()).split(":")[2][:2]
        start = seconds
        counter = 0
        param = nextParam()
        threads = []
        while True:
            if str(time.ctime()).split(":")[2][:2] != seconds:
                seconds = str(time.ctime()).split(":")[2][:2]
                counter += 1
                param = nextParam()

                print("time is " + seconds + " param is "+str(param))
                if counter > 180:
                    break
            sender = threading.Thread(target=send_request_thread, args=(idx,))
            idx += 1
            threads.append(sender)
            sender.start()
            next_event = nextTimeReqeust(param)
            time.sleep(round(next_event, 4))

        for th in threads:
            th.join()







def main():
    assert sys.version_info >= (3, 7), "Script requires Python 3.7+."
    here = pathlib.Path(__file__).parent
    print("start load testing...")
    global idx
    seconds = str(time.ctime()).split(":")[2][:2]
    start = seconds
    counter = 0
    param = nextParam()
    threads = []
    while True:
        if str(time.ctime()).split(":")[2][:2] != seconds:
            seconds = str(time.ctime()).split(":")[2][:2]
            counter += 1
            param = nextParam()

            print("time is " + seconds + " param is "+str(param))
            if counter > 180:
                break
        sender = threading.Thread(target=send_request_thread, args=(idx,))
        idx += 1
        threads.append(sender)
        sender.start()
        next_event = nextTimeReqeust(param)
        time.sleep(round(next_event, 4))
    print("wait till all conncetions done...")
    for th in threads:
        th.join()
        time.sleep(5)
    df = pd.DataFrame(final_data, columns=['id', 'status', 'isok', 'start', 'end', 'latency'])
    time.sleep(1)
    print(df)
    print("finish works")
    df.to_csv("users.csv")

