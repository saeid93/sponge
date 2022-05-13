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

final_data = []


def nextParam():
    return random.randint(100, 500)



def nextTimeReqeust(rateParameter):
    return -math.log(1.0 - random.random()) / rateParameter


async def fetch(url: str, session: ClientSession, index, **kwargs):
    start_time = int(round(time.time() * 1000))
    try:
        resp = await session.post(url=url,
                                    json={"data":{"ndarray":[[1.0, 2.0, 5.0, 6.0]]}},
                                     ssl=False, **kwargs)
        end_time = int(round(time.time() * 1000))
        final_data.append((index, resp.status, 1, start_time, end_time, end_time - start_time))
    except ClientConnectorError:
        return (url, 404)
    return (url, resp.status)


async def make_request(url, idx, **kwargs):
    headers = {
            'Content-Type': 'application/json',
            
        }
    async with ClientSession(headers=headers,trust_env=True) as session:
        tasks = []
        result = await fetch(url, session, idx, **kwargs)
        await session.close()


async def generator(url, index, **kwargs):
    await asyncio.gather(make_request(url, index))


def send_request_thread(idx):
    asyncio.run(generator("http://localhost:32000/seldon/default/sklearn-prof-daqiq/api/v1.0/predictions", idx))


idx = 0


def main():
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

    for th in threads:
        th.join()


if __name__ == "__main__":
    assert sys.version_info >= (3, 7), "Script requires Python 3.7+."
    here = pathlib.Path(__file__).parent
    print("start load testing...")
    main()
    print("wait till all conncetions done...")
    time.sleep(5)
    df = pd.DataFrame(final_data, columns=['id', 'status', 'isok', 'start', 'end', 'latency'])
    time.sleep(1)
    print(df)
    print("finish works")
    df.to_csv("users.csv")
    
    # while True:
