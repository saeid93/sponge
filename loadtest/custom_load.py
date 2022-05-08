import asyncio
import logging
import pathlib
import sys
import threading
import aiohttp
import time

from aiohttp import ClientSession, ClientConnectorError
import random
import math

final_data = []


def nextParam():
    return random.randint(5, 200)


def nextTimeReqeust(rateParameter):
    return -math.log(1.0 - random.random()) / rateParameter


async def fetch(url: str, i, session: ClientSession, index, **kwargs):
    start_time = int(round(time.time() * 1000))
    try:
        resp = await session.request(method="GET", url=url, ssl=False, **kwargs)
        end_time = int(round(time.time() * 1000))
        final_data.append((idx, resp.status, 1, start_time, end_time))
    except ClientConnectorError:
        return (url, 404)
    print("end ", i)
    return (url, resp.status)


async def make_request(url, count, idx, **kwargs):
    async with ClientSession(trust_env=True) as session:
        tasks = []
        result = await fetch(url, count, session, idx, **kwargs)
        print(result)


async def generator(url, index, **kwargs):
    await asyncio.gather(make_request(url, index))


def send_request_thread(idx):
    asyncio.run(generator("https://realpython.com/", idx))


idx = 0


def main():
    global idx
    seconds = str(time.ctime()).split(":")[2][:2]
    param = nextParam()
    while True:
        if str(time.ctime()).split(":")[2][:2] != seconds:
            seconds = str(time.ctime()).split(":")[2][:2]
            param = nextParam()
        sender = threading.Thread(target=send_request_thread, args=(idx,))
        idx += 1
        sender.start()
        next_event = nextTimeReqeust(param)
        time.sleep(round(next_event, 4))


if __name__ == "__main__":
    assert sys.version_info >= (3, 7), "Script requires Python 3.7+."
    here = pathlib.Path(__file__).parent
    main()
    # while True:
