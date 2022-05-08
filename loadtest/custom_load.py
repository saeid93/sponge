import asyncio
import logging
import pathlib
import sys
import threading
import aiohttp
from aiohttp import ClientSession, ClientConnectorError


async def fetch(url: str, i, session: ClientSession, index, **kwargs):
    try:
        print("start ", i, index)
        resp = await session.request(method="GET", url=url, ssl=False, **kwargs)

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
    for i in range(10):
        await asyncio.gather(make_request(url, i, index))


def send_request_thread(url):
    asyncio.run(generator("https://realpython.com/", url))


if __name__ == "__main__":
    assert sys.version_info >= (3, 7), "Script requires Python 3.7+."
    here = pathlib.Path(__file__).parent
    while True:
        pass



    for index in range(3):
        logging.info("Main    : create and start thread %d.", index)
        x = threading.Thread(target=send_request_thread, args=(index,))
        x.start()
