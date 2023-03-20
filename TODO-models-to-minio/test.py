from torch.autograd import Variable
import torch.onnx
import torchvision
import torch
import onnx
import numpy as np
import os
from numpy import asarray
  
from PIL import Image
from torchvision import transforms

import onnxruntime as ort
def create_batch(batch_size):
    batch = []
    for i in range(batch_size):
        batch.append(read_file())
    return batch

# %%
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

# %%
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

def test_model():
    onnx_model = onnx.load("temp.onnx")
    ort_sess = ort.InferenceSession('temp.onnx')
    cuda = torch.device('cuda:0')
    image = Image.open("zidane.jpg")
    image = np.array(image).astype(np.float32)
    image /= 255.
    image = np.moveaxis(image, -1, 0)
    images = np.array([image])
    print(images.shape)
    images = torch.rand(4, 3, 640, 640).to('cpu')
    outputs = ort_sess.run(None, {'images': images.numpy()})
    print(len(outputs[0][0]))

# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom
# print( next(model.parameters()))
# # Images
# img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list

# # Inference
# results = model(img)

# # Results
# results.print()  

def test_onnx():
    imgsz = [640, 640]
    model_variant = "resnet34"
    dummy_input = torch.zeros(1, 3, *imgsz).to('cpu')
    print(f"$$$$$$$$$$$$$$ {model_variant}$$$$$$$$$$$$$$$$$$44")
    model = torch.hub.load('ultralytics/yolov5', model_variant, pretrained=True)
    torch.onnx.export( model,  # --dynamic only compatible with cpu
        dummy_input,
        "temp.onnx",
        verbose=False,
        input_names=['images'],
        output_names=['output'],
        dynamic_axes={
            'images': {
                0: 'batch',
                2: 'height',
                3: 'width'},  # shape(1,3,640,640)
            'output': {
                0: 'batch',
                1: 'anchors'}  # shape(1,25200,85)
        } )

test_onnx()