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
from prom_client import *
# import plot_builder as pb


def nextParam():
    return random.randint(100, 500)
        
class StressGenerator:
    def __init__(self, url, is_dataset = False, data_path = None):
        self.final_data = []
        self.url = url
        self.is_dataset = is_dataset
        self.data_path = data_path
        self.idx = 0
        self.type = "POST"
        self.headers = {
                'Content-Type': 'application/json',
                
            }
        self.data_input = {"data":{"ndarray":[[1.0, 2.0, 5.0, 6.0]]}}

    def set_request_type(self, type):
        self.type = type

    def set_headers(self, headers):
        self.headers = headers

    def set_input(self, input):
        self.data_input = {
            "data": input
        }



    def nextParamFile(self, file_object, chunk_size):

        while True:
            data = file_object.read(chunk_size)
            if not data:
                break
            yield data



    def nextTimeReqeust(self, rateParameter):
        return -math.log(1.0 - random.random()) / rateParameter


    async def fetch(self, session: ClientSession, index, **kwargs):
        start_time = int(round(time.time() * 1000))
        try:
            if self.type == "POST":
                resp = await session.post(url=self.url,
                                            json=self.data_input,
                                            ssl=False, **kwargs)
                end_time = int(round(time.time() * 1000))
                self.final_data.append((index, resp.status, 1, start_time, end_time, end_time - start_time))
            else:
                resp = await session.get(url=self.url, ssl=False, **kwargs)
                end_time = int(round(time.time() * 1000))
                self.final_data.append((index, resp.status, 1, start_time, end_time, end_time - start_time))

        except ClientConnectorError:
            return (self.url, 404)
        return (self.url, resp.status)


    async def make_request(self, idx, **kwargs):
        headers = self.headers
        async with ClientSession(headers=headers,trust_env=True) as session:
            tasks = []
            result = await self.fetch(session, idx, **kwargs)
            await session.close()


    async def generator(self, index, **kwargs):
        await asyncio.gather(self.make_request(index))


    def send_request_thread(self, idx):
        asyncio.run(self.generator( idx))

    def run(self):
        seconds = str(time.ctime()).split(":")[2][:2]
        start = seconds
        counter = 0
        param = nextParam()
        threads = []
        while True:
            if str(time.ctime()).split(":")[2][:2] != seconds:
                seconds = str(time.ctime()).split(":")[2][:2]
                counter += 1
                if self.is_dataset:
                    param = self.nextParamFile(f)
                else:
                    param = nextParam()

                print("time is " + counter.__str__() + " param is "+str(param))
                if counter > 10:
                    break
            sender = threading.Thread(target=self.send_request_thread, args=(self.idx,))
            self.idx += 1
            threads.append(sender)
            sender.start()
            next_event = self.nextTimeReqeust(param)
            time.sleep(round(next_event, 8))

        for th in threads:
            th.join()
        df = pd.DataFrame(self.final_data, columns=['id', 'status', 'isok', 'start', 'end', 'latency'])
        time.sleep(1)
        print(df)
        print("finish works")
        df.to_csv("users.csv")
        # pr = PromClient(self.pod_name)
        # pr.cpu_data()


sc = StressGenerator(url='http://localhost:32000/seldon/default/sklearn-prof-daqiq/api/v1.0/predictions')
sc.set_input()
sc.run()






