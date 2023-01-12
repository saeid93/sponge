import os
from time import sleep
import time
import argparse
import yaml
import sys

from jinja2 import Environment, FileSystemLoader
import torch
from PIL import Image
import requests
import tritonclient.http as httpclient
# from tritonclient.utils import InferenceServerException
from torchvision import transforms
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoTokenizer

from prom import (
    get_cpu_usage,
    get_memory_usage, 
    get_inference_duration,
    get_queue_duration,
    get_inference_count)

project_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
sys.path.append(os.path.normpath(os.path.join(project_dir, '..', '..')))
PATH = "/home/cc/infernece-pipeline-joint-optimization/profiling/titon_profiler"
deploy = "deploy"
service = "service"
pod_monitor = "pod-monitor"
from utils.constants import (
    TEMP_MODELS_PATH,
    KUBE_YAMLS_PATH
    )

os.system('sudo umount -l ~/my_mounting_point')
os.system('cc-cloudfuse mount ~/my_mounting_point')
 
data_folder_path = '/home/cc/my_mounting_point/datasets'
dataset_folder_path = os.path.join(
    data_folder_path, 'ILSVRC/Data/DET/test'
)
classes_file_path = os.path.join(
    data_folder_path, 'imagenet_classes.txt'
)
 
image_names = os.listdir(dataset_folder_path)
image_names.sort()
with open(classes_file_path) as f:
    classes = [line.strip() for line in f.readlines()]

def image_loader(folder_path, image_name):
    image = Image.open(
        os.path.join(folder_path, image_name))
    return image

def create_batch_image(batch_size):
    num_loaded_images = batch_size
    images = {
        image_name: image_loader(
            dataset_folder_path, image_name) for image_name in image_names[
                :num_loaded_images]}
    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)])
 
    return torch.stack(list(map(lambda a: transform(a), list(images.values()))))

class Profiler:
    def __init__(self, type, port, yaml_file, database, name, mr, cpu_count):
        self.type = type
        self.port = port
        self.yaml_file = yaml_file
        self.database = database
        self.app_name = name
        self.model_repository = mr
        self.cpu_count = cpu_count

        if not os.path.exists(database):
            os.makedirs(database)

    def yaml_runner(self,enviroment, space, data):
        template = enviroment.get_template(f"{space}.yaml")
        content = template.render(data)
        content =  yaml.safe_load(content)
        with open(f'{space}.yaml', 'w') as yaml_file:
            yaml.dump(content, yaml_file, default_flow_style=False)

        os.system(f"kubectl apply -f {space}.yaml")

    def build_kuber(self):
        deploy_vars = {
        "app_name": self.app_name,
        "cpu_count": self.cpu_count,        
        "model_repository": self.model_repository}

        service_vars = {
            "service_name": self.app_name + "-service",
            "app_name": self.app_name,
            "http_port": self.port,
            "grpc_port": int(self.port) + 1,
            "metrics_port": int(self.port) + 2 
        }

        pod_monitor_vars = {
            "app_name": self.app_name,
            "podmonitor_name": self.app_name + "-podmonitor"
        }
        enviroment = Environment(
            loader=FileSystemLoader(os.path.join(
                PATH, "templates/")))
        self.yaml_runner(enviroment, deploy, deploy_vars )
        self.yaml_runner(enviroment, service, service_vars)
        self.yaml_runner(enviroment, pod_monitor, pod_monitor_vars)
    

    def send_request(self, model_name, model_version, inputs, outputs, batch_size, name_space="default"):
        start_load = time.time()
        res = requests.post(url=f'http://localhost:{self.port}/v2/repository/models/{model_name}/load')
        load_time = time.time() - start_load
        

        with open(database + "/" + "load-time.txt", "a") as f:
            f.write(f"load time of {model_name} is {load_time} \n")

        sleep(60)
        cpu_usages = []
        memory_usages = []
        infer_times = []
        input_times = []
        output_times = []
        queue_times = []
        success_times = []

    
        
        print(model_name, model_version, "start")
        for i in range(170):
            try:
                triton_client = httpclient.InferenceServerClient(
                    url=f'localhost:{self.port}'
                )
            except Exception as e:
                print("context creation failed: " + str(e))

            start_time = time.time()
            result = triton_client.infer(
                        model_name=model_name,model_version=model_version, inputs=inputs, outputs=outputs)
            latency = time.time() - start_time

            triton_client.close()
            print(i)

            
            minutes = 1
            if i > 10:
                cpu_usages.append(
                    get_cpu_usage(
                        self.app_name, name_space, minutes, minutes))
                memory_usages.append(
                    get_memory_usage(
                        self.app_name, name_space, minutes, minutes, True))
                infer_times.append(
                    get_inference_duration(
                        model_name, model_version, self.app_name))
                queue_times.append(
                    get_queue_duration(
                        model_name, model_version, self.app_name))
                success_times.append(
                    get_inference_count(model_name, model_version, self.app_name))
        
        end_time = 5
    
        sleep(80)
        total_time = time.time() - start_load
        minutes = total_time // 60
        minutes = int(minutes)
        if minutes < 2:
            minutes = 2
        end_infer = 0
        if minutes < 10:
            end_infer = 10
        
        else:
            end_infer = minutes + 5


        cpu_usages.append(
            get_cpu_usage(
                self.app_name, name_space, minutes, minutes))
        memory_usages.append(
            get_memory_usage(
                self.app_name, name_space, minutes, minutes))
        infer_times.append(
            get_inference_duration(
                model_name, model_version, self.app_name))
        success_times.append(
            get_inference_count(
                model_name, model_version, self.app_name))
        queue_times.append(
            get_queue_duration(
                model_name, model_version, self.app_name))

        with open(self.database + "/" +"cpu.txt", "a") as cpu_file:
            cpu_file.write(
                f"usage of {model_name} {model_version} on batch {batch_size} is {cpu_usages} \n")

        with open(self.database + "/" +"memory.txt", 'a') as memory_file:
            memory_file.write(
                f"usage of {model_name} {model_version} on batch {batch_size} is {memory_usages} \n")

        with open(self.database + "/" +"infer-prom.txt", "a") as infer:
            infer.write(
                f"infertime of {model_name} {model_version} on batch {batch_size} is {infer_times} \n")
        
        with open(self.database + "/" +"queue_times.txt", 'a') as q:
            q.write(
                f"Queuetimes of {model_name} {model_version} on batch {batch_size} is {queue_times} \n")

        with open(self.database + "/" +"success.txt", "a") as s:
            s.write(
                f"success of {model_name} {model_version} on batch {batch_size} is {success_times} \n")

        requests.post(
            url=f'http://localhost:{url}/v2/repository/models/{model_name}/unload')


    def input_output_creator(self, bat, model_name = None):
        if self.type == "image":
            os.system('sudo umount -l ~/my_mounting_point')
            os.system('cc-cloudfuse mount ~/my_mounting_point')
            inputs = []
            batch =create_batch_image(bat)
            inputs.append(
                            httpclient.InferInput(
                                name="input", shape=batch.shape, datatype="FP32")
                        )
            inputs[0].set_data_from_numpy(batch.numpy(), binary_data=False)

            outputs = []
            outputs.append(httpclient.InferRequestedOutput(name="output"))
            return inputs, outputs, -1
        
        if self.type == "object_detection":
            batch =torch.rand(bat, 3, 640, 640).to('cpu')
            inputs.append(
                            httpclient.InferInput(
                                name="images", shape=batch.shape, datatype="FP32")
                        )
            inputs[0].set_data_from_numpy(batch.numpy(), binary_data=False)

            outputs = []
            outputs.append(httpclient.InferRequestedOutput(name="output"))
            return inputs, outputs, -1
        
        if self.type == "text_classification":
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            inp = tokenizer(["This is a sample" for _ in range(bat)], return_tensors="pt")
            inputs = []
            inputs.append(
                httpclient.InferInput(
                    name="input_ids",shape=inp['input_ids'].shape, datatype="INT64"
            )
            )
            inputs[0].set_data_from_numpy(inp['input_ids'].numpy(), binary_data=False)
            
            inputs.append(
                httpclient.InferInput(
                    name="attention_mask", shape=inp['attention_mask'].shape, datatype="INT64")
            )
            inputs[1].set_data_from_numpy(inp['attention_mask'].numpy())
            
            outputs = []
            outputs.append(httpclient.InferRequestedOutput(name="logits"))


            model_name_s = model_name
            if "/" in model_names:
                model_name_s = model_names.replace("/","")
            
            return inputs, outputs, model_name_s

    def runner(self):
        self.build_kuber()
        print("sleep for one minute to heavy start")
        sleep(50)
        config_file_path = os.path.join(
            KUBE_YAMLS_PATH, f"{self.yaml_file}.yaml")
        with open(config_file_path, 'r') as cf:
            config = yaml.safe_load(cf)

        model_names = config['model_names']
        versions = config['versions']
        model_versions = [[] for _ in range(len(versions))]
        for k, version in enumerate(versions):
            for i in range(len(version)):
                model_versions[k].append(str(i+1))

        results = []
        processes = []
        for bat in [2]:
            print(f"start batch {bat}")
            inputs, outputs, m_name = self.input_output_creator(bat)
            for j,model_name in enumerate(model_names):
                for version in model_versions[j]:
                    if self.type != "text_classification":
                        self.send_request(model_name, version, inputs, outputs, bat)
                    else:
                        inputs, outputs, m_name = self.input_output_creator(bat, model_name)
                        self.send_request(m_name, version, inputs, outputs, bat)
                   


if __name__ == "__main__":
    experiment_root = "experiments"
    profile_parser = argparse.ArgumentParser()
    profile_parser.add_argument('-t',
                       '--type',
                       help='type of profiling',
                       default='text_classification')

    profile_parser.add_argument('-p',
                       '--port',
                       
                       help='port name',
                       default='30600')
    
    profile_parser.add_argument('-y',
                       '--yaml',
                       
                       help='yaml file for models',
                       default='temp')
    profile_parser.add_argument('-r',
                       '--reason',
                       help='reason to proifile',
                       default='nlp')
    
    profile_parser.add_argument('-f',
                       '--file',
                       help='file to save',
                       default='1')
    
    profile_parser.add_argument('-n',
                       '--name',
                       help='name of pod',
                       default='triton-text')

    profile_parser.add_argument('-c',
                       '--cpu',
                       help='cpu count',
                       default= 8)    
    profile_parser.add_argument('-m',
                       '--models',
                       help='model_repository',
                       default='triton-server-new-text')
    profile_parser.add_argument('-x',
                       '--exit',
                       help='exit')

    args =  profile_parser.parse_args()
    type_c = args.type
    cpu_count = int(args.cpu)
    pod_name = args.name
    mr = args.models
    database = os.path.join(experiment_root, args.reason, args.file)
    port = args.port
    yaml_file = args.yaml

    print(f"Start profile {type}")

    pr = Profiler(type_c, port, yaml_file, database, pod_name, mr, cpu_count)
    pr.runner()

    
    