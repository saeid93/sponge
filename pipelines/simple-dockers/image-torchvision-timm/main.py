import os
import numpy as np
import torch
from torchvision import models
from torchvision import transforms
from PIL import Image
# import timm
import time
import logging

try:
    ITERATIONS = int(os.environ['ITERATIONS'])
    logging.warning(f'ITERATIONS set to: {ITERATIONS}')
except KeyError as e:
    ITERATIONS = 60
    logging.warning(
        f"ITERATIONS env variable not set, using default value: {ITERATIONS}")

dir = os.path.dirname(__file__)
image_name = 'input-sample.JPEG'
path = os.path.join(dir, image_name)

start = time.time()
X = np.array(Image.open(path))

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)])
X_trans = Image.fromarray(X.astype(np.uint8))
X_trans = transform(X_trans)
batch = torch.unsqueeze(X_trans, 0)
postprocessing_time = time.time() - start
logging.warning(f"preprocessing time: {postprocessing_time}")

start = time.time()
resnet =  models.resnet18(pretrained=True)
resnet.eval()
logging.warning(f"model loading time: {time.time() - start}")

logging.warning('starting the experiments')
model_times = []
softmax_times = []

for i in range(20):
    out = resnet(batch)

    # time.sleep(4)

    # logging.warning(f'iteration {i} time: {iter_time}')
    # start = time.time()
    percentages = torch.nn.functional.softmax(out, dim=1)[0] * 100
    percentages = percentages.detach().numpy()
    image_net_class = np.argmax(percentages)
    # softmax_times.append(time.time() - start)


start = time.time()

for i in range(ITERATIONS):
    out = resnet(batch)

    # time.sleep(4)

    # logging.warning(f'iteration {i} time: {iter_time}')
    # start = time.time()
    percentages = torch.nn.functional.softmax(out, dim=1)[0] * 100
    percentages = percentages.detach().numpy()
    image_net_class = np.argmax(percentages)
    # softmax_times.append(time.time() - start)
iter_time = time.time() - start
# model_times.append(iter_time)

logging.warning('times:')
logging.warning(iter_time/ITERATIONS)

# logging.warning('model times:')
# logging.warning(model_times)

# logging.warning('softmax times:')
# logging.warning(softmax_times)

# total_times = postprocessing_time + np.array(model_times) + np.array(softmax_times)
# logging.warning('total times:')
# logging.warning(total_times)

# logging.warning('total times average:')
# logging.warning(np.average(total_times))

# logging.warning('total times p99:')
# logging.warning(np.percentile(total_times, 99))

# logging.warning('model times average:')
# logging.warning(np.average(model_times))

# # logging.warning('softmax times average:')
# # logging.warning(np.average(softmax_times))