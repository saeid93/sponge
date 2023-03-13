import os
import time
import json
from typing import List
import numpy as np
from PIL import Image
import asyncio
from barazmoon import Data
from barazmoon import MLServerAsyncGrpc

def load_data(data_type: str, pipeline_path: str):
    if data_type == 'audio':
        input_sample_path = os.path.join(
            pipeline_path, 'input-sample.json'
        )
        input_sample_shape_path = os.path.join(
            pipeline_path, 'input-sample-shape.json'
        )
        with open(input_sample_path, 'r') as openfile:
            data = json.load(openfile)
        with open(input_sample_shape_path, 'r') as openfile:
            data_shape = json.load(openfile)
            data_shape = data_shape['data_shape']
    elif data_type == 'text':
        input_sample_path = os.path.join(
            pipeline_path, 'input-sample.txt'
        )
        input_sample_shape_path = os.path.join(
            pipeline_path, 'input-sample-shape.json'
        )
        with open(input_sample_path, 'r') as openfile:
            data = openfile.read()
        # with open(input_sample_shape_path, 'r') as openfile:
        #     data_shape = json.load(openfile)
        #     data_shape = data_shape['data_shape']
            data_shape = [1]
    elif data_type == 'image':
        input_sample_path = os.path.join(
            pipeline_path, 'input-sample.JPEG'
        )
        data = Image.open(input_sample_path)
        data_shape = list(np.array(data).shape)
        data = np.array(data).flatten()
    data_1 = Data(
        data=data,
        data_shape=data_shape,
        custom_parameters={'custom': 'custom'},
    )

    # Data list
    data = []
    data.append(data_1)
    return data

def check_load_test(
        pipeline_name: str, data_type: str,
        pipeline_path: str):
    data = load_data(
        data_type=data_type,
        pipeline_path=pipeline_path)
    loop_timeout = 5
    ready = False
    while True:
        print(f'waited for {loop_timeout} seconds to check for successful request')
        time.sleep(loop_timeout)
        try:
            load_test(
                pipeline_name=pipeline_name,
                data=data,
                data_type=data_type,
                workload=[1])
            ready = True
        except UnboundLocalError:
            pass
        if ready:
            return ready

def warm_up(
        pipeline_name: str, data_type: str,
        pipeline_path: str, warm_up_duration: int):
    workload = [1] * warm_up_duration
    data = load_data(
        data_type=data_type,
        pipeline_path=pipeline_path)
    load_test(
        pipeline_name=pipeline_name,
        data=data,
        data_type=data_type,
        workload=workload)

def remove_pipeline(pipeline_name):
    os.system(f"kubectl delete seldondeployment {pipeline_name} -n default")
    print('-'*50 + f' pipeline {pipeline_name} successfuly removed ' + '-'*50)
    print('\n')

def load_test(
        pipeline_name: str, data_type: str,
        workload: List[int],
        data: List[Data],
        namespace: str='default',
        no_engine: bool = False,
        mode: str = 'step',
        benchmark_duration=1):
    start_time = time.time()

    endpoint = "localhost:32000"
    deployment_name = pipeline_name
    model = None
    namespace = "default"
    metadata = [("seldon", deployment_name), ("namespace", namespace)]
    load_tester = MLServerAsyncGrpc(
        endpoint=endpoint,
        metadata=metadata,
        workload=workload,
        model=model,
        data=data,
        mode=mode, # options - step, equal, exponential
        data_type=data_type,
        benchmark_duration=benchmark_duration)
    responses = asyncio.run(load_tester.start())
    end_time = time.time()

    # remove ouput for image inputs/outpus (yolo)
    # as they make logs very heavy
    for second_response in responses:
        for response in second_response:
            response['outputs'] = []
    return start_time, end_time, responses