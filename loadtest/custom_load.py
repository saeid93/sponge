import asyncio
import pathlib
import sys

import aiohttp
from aiohttp import ClientSession, ClientConnectorError


async def fetch(url: str, session: ClientSession, **kwargs):
    try:
        resp = await session.request(method="GET", url=url, ssl=False, **kwargs)

    except ClientConnectorError:
        return (url, 404)
    return (url, resp.status)

async def make_request(url,count,  **kwargs):
    async with ClientSession(trust_env=True) as session:
        tasks = []
        for _ in range(count):
            tasks.append(fetch(url, session, **kwargs))
        results = await asyncio.gather(*tasks)
        for result in results:
            print(f'{result[1]} - {str(result[0])}')

if __name__ == "__main__":
    assert sys.version_info >= (3, 7), "Script requires Python 3.7+."
    here = pathlib.Path(__file__).parent
    asyncio.run(make_request("https://stackoverflow.com/", 10))


