
import torchvision
import torch
import pandas as pd
import numpy as np
from torchvision import transforms
from PIL import Image
import os


data_folder_path = '/home/cc/inference/infernece-pipeline-joint-optimization/pipelines/12-inferline/cascade/data'
dataset_folder_path = os.path.join(
    data_folder_path, 'ILSVRC/Data/DET/test'
)
classes_file_path = os.path.join(
    data_folder_path, 'imagenet_classes.txt'
)


# load images
with open(classes_file_path) as f:
    classes = [line.strip() for line in f.readlines()]

x = np.array([])
directory = os.fsencode(dataset_folder_path)

for root, dirs, files in os.walk(dataset_folder_path):
    for filename in files:
        x = np.append(x, filename)
df = pd.DataFrame(data=x, columns=["images"])
df['images'][0]

# resnet models
def resnet_model(img):
    """
    ResNet101 for image classification on ResNet
    """
    # standard resnet image transformation
    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )])
    
    resnet = torchvision.models.resnet101(pretrained=True)
    resnet.eval()
    img_t = transform(img['images'])
    batch_t = torch.unsqueeze(img_t, 0)
    out = resnet(batch_t)
    _, indices = torch.sort(out, descending=True)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    percentage = percentage.detach().numpy()
    return indices.detach().numpy()[0], percentage, percentage[indices[0][0]]


def inceptionv3_model(img):
    transform = transforms.Compose([
    transforms.Resize(256),                    
    transforms.CenterCrop(224),                
    transforms.ToTensor(),                     
    transforms.Normalize(                      
    mean=[0.485, 0.456, 0.406],                
    std=[0.229, 0.224, 0.225]                  
    )])
    
    resnet = torchvision.models.inception_v3(pretrained=True)
    resnet.eval()
    img_t = transform(img['images'])
    batch_t = torch.unsqueeze(img_t, 0)
    out = resnet(batch_t)
    _, indices = torch.sort(out, descending=True)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    percentage = percentage.detach().numpy()
    return indices.detach().numpy()[0], percentage, percentage[indices[0][0]]

def cascade_predict(row):
    """
    cascade predict based on resnet/alexnet results
    """
    r_index = row[1]
    r_perc = row[2]
    r_max_prob = row[3]
    i_index = row[4]
    i_perc = row[5]
    i_max_prob= row[6]
#     print(np.isnan(i_max_prob))
    
    if np.isnan(i_max_prob):
        # didn't go to inception because resnet prediction was confident enough
        return r_index, r_perc, classes[r_index[0]]
    else:
        #choose the distribution with the higher max_prob
        if r_max_prob > i_max_prob:
            return r_index, r_perc, classes[r_index[0]]
        else:
            return i_index, i_perc, classes[i_index[0]]

def filter_color_images(img):
    img_2 = Image.open(os.path.join(dataset_folder_path, img[0]))
    if img_2.mode == 'RGB':
        return True
    return False

def show(img):
    img_2 = Image.open(os.path.join(dataset_folder_path, img))
    img_2.show()

def load_pics(img):
    img = Image.open(os.path.join(dataset_folder_path, img))
    return img

# -------- preprocess --------
df = df.sort_values(by=['images'])

df_s = df.head(5)
df_s = df_s[df_s.apply(filter_color_images, axis=1)]
df_s['images'] = df_s['images'].apply(load_pics)

img_index = 0


resnet_cutoff = 85
# -------- with dataframe operations --------
resnet_pred = resnet_model(df_s.iloc[img_index])



# -------- with dataframe operations --------
resnet_preds = df_s.apply(
    resnet_model, axis=1, result_type="expand").rename(
        columns={0: "resnet_indices", 1: "resnet_percentage", 2: "resnet_max_prob"}) 

# I used pandas notation here, but the value inside the join would be a WHERE sql query
# Might want to explore query optimization here? (Between where/join/apply)
inception_preds = df_s.join(
    resnet_preds[resnet_preds['resnet_max_prob'] < resnet_cutoff], how='right').apply(
        inceptionv3_model, axis=1, result_type="expand").rename(
            columns={0: "inception_indices", 1: "inception_percentage", 2:"inception_max_prob"}) 

all_preds = df_s.join([resnet_preds, inception_preds])

cascade_df = all_preds.join(
    all_preds.apply(
        cascade_predict, axis=1, result_type="expand").rename(
            columns={0: "cascade_indices", 1: "cascade_percentage", 2:"cascade_prediction"}))
