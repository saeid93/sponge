import time
import random
from fastapi import FastAPI
import asyncio

app = FastAPI()
@app.post('/')
async def sumer():
    await asyncio.sleep(1)
    return {'1': "1"}