import logging
import torch
from copy import deepcopy

logger = logging.getLogger(__name__)

class VideoYolo(object):
    def __init__(self) -> None:
        super().__init__()
        self.loaded = False

    def load(self):
        logger.info('Loading the ML models')
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        # self.resnet = models.resnet101(pretrained=True)
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        # self.resnet.eval()
        self.loaded = True
        logger.info('model loading complete!')

    def predict(self, X, features_names=None):
        if self.loaded == False:
            self.load()
        logger.info(f"Incoming input:\n{X}\nwas recieved!")
        # TODO some transformation might be needed
        objs = self.model(X)
        output = self.get_cropped(objs)
        logger.info(f"Output:\n{output}\nwas sent!")
        return output

    def get_cropped(result):
        """
        crops selected objects for
        the subsequent nodes
        """
        result = result.crop()
        liscense_labels = ['car', 'truck']
        car_labels = ['car']
        person_labels = ['person']
        output_list = {'person': [], 'car': [], 'liscense': []}
        for obj in result:
            for label in liscense_labels:
                if label in obj['label']:
                    output_list['liscense'].append(deepcopy(obj['im']))
                    break
            for label in car_labels:
                if label in obj['label']:
                    output_list['car'].append(deepcopy(obj['im']))
                    break
            for label in person_labels:
                if label in obj['label']:
                    output_list['person'].append(deepcopy(obj['im']))
                    break
        return output_list
