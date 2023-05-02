from typing import List, Tuple, Any
import time
import asyncio

# from aiohttp import ClientSession, TCPConnector


class BatchingQueue:
    def __init__(self, batch_size: int, delay: int) -> None:
        self.__batch_size = batch_size
        self.__delay = delay  # ms
        self.__q = asyncio.Queue()

    def update_queue(self, batch_size, delay):
        self.__batch_size = batch_size
        self.__delay = delay

    async def get(self):
        items = []
        item = await self.__q.get()
        *item, first_entrance = item
        target_time = first_entrance + self.__delay / 1000
        items.append(item)
        while True:
            print("QS", self.__q.qsize())
            if len(items) >= self.__batch_size:
                break
            try:
                item = await asyncio.wait_for(
                    self.__q.get(), timeout=target_time - time.perf_counter()
                )
            except asyncio.TimeoutError:
                break
            *item, _ = item
            items.append(item)

        # if timeout of the first query passed but there are more queries in the queue
        if self.__q.qsize() > 0 and len(items) < self.__batch_size:
            for _ in range(min(self.__batch_size - len(items), self.__q.qsize())):
                item = await self.__q.get()
                *item, _ = item
                items.append(item)
        return items

    async def put(self, item: tuple):
        await self.__q.put((*item, time.perf_counter()))


class QueueHandler:
    def __init__(self) -> None:
        self.__batching_queue = BatchingQueue(4, 200)
        # self.__session = ClientSession(connector=TCPConnector(limit=0))
        self.__target_url = ""

    def initialize(self):
        asyncio.create_task(self._queue_manager())

    async def process_request(self, req):
        fut = await self.__submit_query(req)
        await fut
        return fut.result()

    async def __submit_query(self, req) -> asyncio.Future:
        fut = asyncio.Future()
        await self.__batching_queue.put((fut, req))
        return fut

    async def _queue_manager(self):
        while True:
            items: List[Tuple[asyncio.Future, Any]] = await self.__batching_queue.get()
            await asyncio.create_task(
                self.__submit_batch(items)
            )  # if concurrent batches, remove "await"

    async def __submit_batch(self, items):
        batch = [item[1] for item in items]
        futures = [item[0] for item in items]
        # async with self.__session.post(self.target_url, batch) as response:
        #     response = await response.json()
        await asyncio.sleep(1)
        response = [d * 2 for d in batch]
        idx = 0
        for resp in response:
            fut: asyncio.Future = futures[idx]
            idx += 1
            fut.set_result(resp)


async def main():
    handler = QueueHandler()
    handler.initialize()

    t = time.perf_counter()
    res1 = asyncio.create_task(handler.process_request(2))
    res2 = asyncio.create_task(handler.process_request(5))
    res3 = asyncio.create_task(handler.process_request(6))
    res4 = asyncio.create_task(handler.process_request(-8))
    res5 = asyncio.create_task(handler.process_request(10))
    res6 = asyncio.create_task(handler.process_request(10))

    print(
        await res1,
        await res2,
        await res3,
        await res4,
        await res5,
        await res6,
        time.perf_counter() - t,
    )


asyncio.run(main())
