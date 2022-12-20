# start commnad:
# python -m uvicorn fastapi-test:app
import time
from fastapi import FastAPI
import asyncio

app = FastAPI()
@app.post('/')
async def sumer():
    arrival = time.time()
    # await asyncio.sleep(0.2)
    time.sleep(0.3)
    serving = time.time()
    # to make it consistent with the Seldon data model
    model_name = 'mock_one'
    output = {'model_name': model_name, 'outputs': [{'data': [f'{{"time": {{"arrival_{model_name}": {arrival}, "serving_{model_name}": {serving}}}, "output": []}}']}]}
    return output