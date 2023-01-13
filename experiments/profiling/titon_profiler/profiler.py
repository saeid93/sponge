import os
import time
import argparse
import yaml
import sys

from jinja2 import Environment, FileSystemLoader
import torch
from PIL import Image
import requests
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
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
# PATH = "/home/cc/infernece-pipeline-joint-optimization/experiments/profiling/titon_profiler/templates"
deploy = "deploy"
service = "service"
pod_monitor = "pod-monitor"
from utils.constants import (
    TRITON_PROFILING_CONFIGS_PATH,
    TRITON_PROFILING_TEMPLATES_PATH,
    NODE_PROFILING_RESULTS_TRITON_PATH
    )

TIMEOUT = 1


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
 
    return torch.stack(
        list(map(
            lambda a: transform(a), list(images.values()))))

class Profiler:
    def __init__(
        self, model_type, port, expriment_config,
        database, name, model_repository, cpu_count,
        batch_sizes, iterations, protocol):
        self.model_type = model_type
        self.port = port
        self.experiment_config = expriment_config
        self.database = database
        self.app_name = name
        self.model_repository = model_repository
        self.cpu_count = cpu_count
        self.batch_sizes = batch_sizes
        self.iterations = iterations
        self.protocol = protocol
        if protocol == 'http':
            self.client = httpclient
        elif protocol == 'grpc':
            self.client = grpcclient

        if not os.path.exists(database):
            os.makedirs(database)

    def runner(self):
        self.build_kuber()
        print("sleep for one minute to heavy start")
        time.sleep(TIMEOUT/5)
        config_file_path = os.path.join(
            TRITON_PROFILING_CONFIGS_PATH, f"{self.experiment_config}.yaml")
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
        for batch_size in batch_sizes:
            print(f"start batch {batch_size}")
            inputs, outputs, model_name = self.input_output_creator(batch_size)
            for j, model_name in enumerate(model_names):
                for version in model_versions[j]:
                    if self.model_type != "text_classification":
                        self.load_test(
                            model_name, version, inputs,
                            outputs, batch_size, iterations=self.iterations)
                    else:
                        inputs, outputs, model_name = self.input_output_creator(
                            batch_size, model_name)
                        self.load_test(
                            model_name, version, inputs, outputs,
                            batch_size, iterations=self.iterations)

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
            loader=FileSystemLoader(
                TRITON_PROFILING_TEMPLATES_PATH))
        self.yaml_runner(enviroment, deploy, deploy_vars )
        self.yaml_runner(enviroment, service, service_vars)
        self.yaml_runner(enviroment, pod_monitor, pod_monitor_vars)

    def yaml_runner(self,enviroment, space, data):
        template = enviroment.get_template(f"{space}.yaml")
        content = template.render(data)
        # content =  yaml.safe_load(content)
        # with open(f'{space}.yaml', 'w') as yaml_file:
        #     yaml.dump(content, yaml_file, default_flow_style=False)

        # os.system(f"kubectl apply -f {space}.yaml")
        command = f"""cat <<EOF | kubectl apply -f -
{content}
        """
        os.system(command)


    def input_output_creator(self, bat, model_name = None):
        if self.model_type == "image_classification":
            os.system('sudo umount -l ~/my_mounting_point')
            os.system('cc-cloudfuse mount ~/my_mounting_point')
            inputs = []
            batch =create_batch_image(bat)
            inputs.append(self.client.InferInput(
                name="input", shape=batch.shape, datatype="FP32"))
            if self.protocol == 'grpc':
                inputs[0].set_data_from_numpy(batch.numpy())
            elif self.protocol == 'http':
                inputs[0].set_data_from_numpy(batch.numpy(), binary_data=False)
            outputs = []
            outputs.append(self.client.InferRequestedOutput(name="output"))
            return inputs, outputs, -1
        
        if self.model_type == "object_detection":
            batch =torch.rand(bat, 3, 640, 640).to('cpu')
            inputs.append(self.client.InferInput(
                name="images", shape=batch.shape, datatype="FP32"))
            inputs[0].set_data_from_numpy(batch.numpy(), binary_data=False)

            outputs = []
            outputs.append(self.client.InferRequestedOutput(name="output"))
            return inputs, outputs, -1
        
        if self.model_type == "text_classification":
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            inp = tokenizer(["This is a sample" for _ in range(bat)], return_tensors="pt")
            inputs = []
            inputs.append(
                self.client.InferInput(
                    name="input_ids",shape=inp['input_ids'].shape, datatype="INT64"
            )
            )
            inputs[0].set_data_from_numpy(inp['input_ids'].numpy(), binary_data=False)
            
            inputs.append(
                self.client.InferInput(
                    name="attention_mask", shape=inp['attention_mask'].shape, datatype="INT64")
            )
            inputs[1].set_data_from_numpy(inp['attention_mask'].numpy())
            outputs = []
            outputs.append(self.client.InferRequestedOutput(name="logits"))
            model_name_s = model_name
            if "/" in model_names:
                model_name_s = model_names.replace("/","")
            return inputs, outputs, model_name_s

    def load_test(
        self, model_name, model_version, inputs,
        outputs, batch_size, iterations, name_space="default"):
        start_load = time.time()
        res = requests.post(
            url=f'http://localhost:{self.port}/v2/repository/models/{model_name}/load')
        load_time = time.time() - start_load

        with open(database + "/" + "load-time.txt", "a") as f:
            f.write(f"load time of {model_name} is {load_time} \n")

        time.sleep(TIMEOUT)
        cpu_usages = []
        memory_usages = []
        infer_times = []
        input_times = []
        output_times = []
        roundtrip_latencies = []
        queue_times = []
        success_times = []

        print(model_name, model_version, "start")
        for i in range(iterations):
            try:
                triton_client = self.client.InferenceServerClient(
                    url=f'localhost:{self.port}'
                )
            except Exception as e:
                print("context creation failed: " + str(e))

            start_time = time.time()
            result = triton_client.infer(
                        model_name=model_name,
                        model_version=model_version,
                        inputs=inputs, outputs=outputs)
            roundtrip_latency = time.time() - start_time
            roundtrip_latencies.append(roundtrip_latency)
            triton_client.close()
            print(i)

            minutes = 1
            # if i > 10:
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
                get_inference_count(
                    model_name, model_version, self.app_name))
            roundtrip_latencies.append(roundtrip_latency)

        total_time = time.time() - start_load
        minutes = int(total_time // 60)

        # cpu_usages.append(
        #     get_cpu_usage(
        #         self.app_name, name_space, minutes, minutes))
        # memory_usages.append(
        #     get_memory_usage(
        #         self.app_name, name_space, minutes, minutes))
        # infer_times.append(
        #     get_inference_duration(
        #         model_name, model_version, self.app_name))
        # success_times.append(
        #     get_inference_count(
        #         model_name, model_version, self.app_name))
        # queue_times.append(
        #     get_queue_duration(
        #         model_name, model_version, self.app_name))
        time.sleep(TIMEOUT/5)

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
            url=f'http://localhost:{self.port}/v2/repository/models/{model_name}/unload')

if __name__ == "__main__":
    experiment_root = NODE_PROFILING_RESULTS_TRITON_PATH
    profile_parser = argparse.ArgumentParser()
    # options obect_detection, image_classification,
    #  text_classification
    profile_parser.add_argument('-t',
                       '--type',
                       help='type of profiling',
                       default='image_classification')

    profile_parser.add_argument('-p',
                       '--port', 
                       help='port name',
                       default='30600')
    
    profile_parser.add_argument('-y',
                       '--yaml',                   
                       help='yaml file for models',
                       default='model-load')
    profile_parser.add_argument('-r',
                       '--reason',
                       help='reason to proifile',
                       default='resnet')
    
    profile_parser.add_argument('-f',
                       '--file',
                       help='file to save',
                       default='1')
    
    profile_parser.add_argument('-n',
                       '--name',
                       help='name of pod',
                       default='triton-image')

    profile_parser.add_argument('-c',
                       '--cpu',
                       help='cpu count',
                       default= 1)    

    profile_parser.add_argument('-m',
                       '--models_repository',
                       help='models repository',
                       default='triton-server-all-new')

    profile_parser.add_argument('-i',
                       '--iterations',
                       help='iterations',
                       default='10')

    profile_parser.add_argument('-pr',
                       '--protocol',
                       help='sending protocol',
                       default='grpc')

    profile_parser.add_argument('-x',
                       '--exit',
                       help='exit')

    args =  profile_parser.parse_args()
    model_type = args.type
    cpu_count = int(args.cpu)
    pod_name = args.name
    models_repository = args.models_repository
    database = os.path.join(
        experiment_root, args.reason, args.file)
    port = args.port
    yaml_file = args.yaml
    iterations = int(args.iterations)
    protocol = args.protocol

    batch_sizes = [1, 2, 4, 8] # batch sizes to check
    print("Profiling sarted ... ")

    pr = Profiler(
        model_type, port, yaml_file, database,
        pod_name, models_repository, cpu_count,
        batch_sizes, iterations, protocol)
    pr.runner()

    
    