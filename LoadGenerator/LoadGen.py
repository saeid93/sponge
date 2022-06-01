import asyncio
import logging
import pathlib
import sys
import os
import PIL
from PIL import Image
from typing import Dict
from abc import ABC, abstractmethod
from multiprocessing import Process, Manager
import numpy as np
from PIL import ImageFile

from seldon_core.seldon_client import SeldonClient

import threading
import aiohttp
import time

import pandas as pd
from aiohttp import ClientSession, ClientConnectorError
import random
import math
# from prom_client import *
# import plot_builder as pb


def nextParam():
    return 3
    return random.randint(5, 20)

def image_loader(folder_path, image_name):
    image = Image.open(
        os.path.join(folder_path, image_name))
    # if there was a need to filter out only color images
    # if image.mode == 'RGB':
    #     pass
    return image

        
class StressGenerator:
    def __init__(self, url=None, is_dataset = False,mode='seldon', data_path = None):
        manager = Manager()
        self.final_data = manager.dict()

        # self.final_data = []
        self.url = url
        self.is_dataset = is_dataset
        self.data_path = data_path
        self.idx = 0
        self.mode =mode
        self.type = "POST"
        self.headers = {
                'Content-Type': 'application/json',
                
            }
        self.data_input = {"data":{"ndarray":[[1.0, 2.0, 5.0, 6.0]]}}
        self.data_folder_path = dataset_folder_path

    @abstractmethod
    async def make_seldon_request(self, idx):
        raise NotImplementedError("no .forward method")


    def set_request_type(self, type):
        self.type = type

    def set_headers(self, headers):
        self.headers = headers

    def set_input(self, input):
        if self.mode == 0:
            self.data_input = {
                "data": input
            }
        else:
            self.data_input = {
                image_name: image_loader(
                    self.dataset_folder_path, image_name) for image_name in image_names[
                        :num_loaded_images]}




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
        if self.mode =="natural":
            await asyncio.gather(self.make_request(index))
        
        if self.mode == 'seldon':
            await asyncio.gather(self.make_seldon_request(index))


    def send_request_thread(self, idx):
        asyncio.run(self.generator( idx))

    def run(self):
        seconds = str(time.ctime()).split(":")[2][:2]
        start = seconds
        counter = 0
        param = nextParam()
        threads = []
        counter1 = 0
        while True:
            if str(time.ctime()).split(":")[2][:2] != seconds:
                seconds = str(time.ctime()).split(":")[2][:2]
                counter += 1
                if self.is_dataset:
                    param = self.nextParamFile(f)
                else:
                    param = nextParam()

                print("time is " + counter.__str__() + " param is "+str(param))
                print("idx is ", self.idx)

                if counter > 10:
                    break
            sender = Process(target=self.send_request_thread, args=(self.idx,))
            next_event = self.nextTimeReqeust(param)
            next_event = next_event + time.time()
            self.idx += 1
            threads.append(sender)
            sender.start()
            next_event = next_event - time.time()
            if next_event  > 0:
                time.sleep(round(next_event, 8))
            if next_event <= 0:
                counter1 += 1


        for th in threads:
            th.join()
        df = pd.DataFrame(self.final_data, columns=['id', 'status', 'isok', 'start', 'end', 'latency'])
        time.sleep(1)
        print(counter1)
        print(df)
        print("finish works")
        df.to_csv("users.csv")
        # pr = PromClient(self.pod_name)
        # pr.cpu_data()



class StressGeneratorSeldon(StressGenerator):
    def __init__(self, data_folder_path,dataset_folder_path,classes_file_path,transport,gateway, gateway_endpoint, deployment_name, namespace, *args, **kwargs):
        super(StressGeneratorSeldon, self).__init__(*args, **kwargs)
        self.data_folder_path = data_folder_path
        self.dataset_folder_path = dataset_folder_path
        self.classes_file_path = classes_file_path
        self.transport = transport
        self.gateway=gateway
        self.gateway_endpoint = gateway_endpoint
        self.deployment_name = deployment_name
        self.namespace = namespace
        self.num_loaded_images = 2
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        self.image_names = os.listdir(self.dataset_folder_path)
        self.image_names.sort()
        self.sc = SeldonClient(
            gateway_endpoint=gateway_endpoint,
            gateway=gateway,
            transport=transport,
            deployment_name=deployment_name,
            namespace=namespace)
        
        self.images = {
                    image_name: self.image_loader(
                        self.dataset_folder_path, image_name) for image_name in self.image_names[
                            :self.num_loaded_images]}
    
    def image_loader(self,folder_path, image_name):
        image = Image.open(
            os.path.join(folder_path, image_name))
        # if there was a need to filter out only color images
        # if image.mode == 'RGB':
        #     pass
        return image

    async def fetch(self, index):
        start_time = int(round(time.time() * 1000))
        try:
            if self.type == "POST":
                
                with open(self.classes_file_path) as f:
                    classes = [line.strip() for line in f.readlines()]
                
                image_name, image = list(self.images.items())[random.randint(0,len(self.images)-1)]
                image = np.array(image)
                print(f"{index} send")
                response = self.sc.predict(
                    data=image
                )

                if response.success:
                    request_path = response.response['meta']['requestPath'].keys()
                    pipeline_response = response.response['jsonData']
                    print(f"done {index}")
                    end_time = int(round(time.time() * 1000))
                    item = self.final_data['list_item'] = list()
                    item.append((index, 200, 1, start_time, end_time, end_time - start_time))
                    self.final_data['list_item'] = item
                    return self.final_data
                
           

        except Exception as e :
            print(f"there is error {e}")
            return (self.url, 404)
        return (self.url, 400)



    async def make_seldon_request(self, idx, **kwargs):
        response = await self.fetch(idx, **kwargs)


data_folder_path = '/home/cc/my_mounting_point/datasets'
dataset_folder_path = os.path.join(
   data_folder_path, 'ILSVRC/Data/DET/test'
)
classes_file_path = os.path.join(
    data_folder_path, 'imagenet_classes.txt'
)
deployment_name = 'inferline-preprocess'
transport='rest'
gateway='istio'
gateway_endpoint="localhost:32000"
namespace='alrieza'
seldon = StressGeneratorSeldon(
        data_folder_path = data_folder_path,
        dataset_folder_path = dataset_folder_path,
        classes_file_path = classes_file_path,
        transport=transport,
        gateway = gateway,
        gateway_endpoint = gateway_endpoint,
        deployment_name = deployment_name,
        namespace = namespace
)
seldon.run()






