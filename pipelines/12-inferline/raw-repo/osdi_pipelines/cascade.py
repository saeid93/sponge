from regex import R
import torchvision
import torch
import pandas as pd
import numpy as np
from torchvision import transforms
from PIL import Image
import os


# ---------------------
with open('./imagenet_classes.txt') as f:
    classes = [line.strip() for line in f.readlines()]

x = np.array([])
directory = os.fsencode('ILSVRC/Data/DET/test')

for root, dirs, files in os.walk("ILSVRC/Data/DET/test"):
    for filename in files:
        x = np.append(x, filename)
df = pd.DataFrame(data=x, columns=["images"])
df['images'][0]

# ---------------------
def resnet_model(img):
    """
    ResNet101 for image classification on ResNet
    """
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
    img_2 = Image.open('ILSVRC/Data/DET/test/'+img[0])
    # try:
    img_t = transform(img_2)
        # processed += 1
        # print(f"image: {img[0]} successful!")
    # except RuntimeError:
    #     print(f"image: {img[0]} was skipped due to single channel")
    #     skipped += 1 
    batch_t = torch.unsqueeze(img_t, 0)
    out = resnet(batch_t)
    _, indices = torch.sort(out, descending=True)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    p_2 = percentage.detach().numpy()
    return indices.detach().numpy()[0], p_2, p_2[indices[0][0]]

# ---------------------
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
    img_2 = Image.open('ILSVRC/Data/DET/test/'+img[0])
    img_t = transform(img_2)
    batch_t = torch.unsqueeze(img_t, 0)
    out = resnet(batch_t)
    _, indices = torch.sort(out, descending=True)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    p_2 = percentage.detach().numpy()
    return indices.detach().numpy()[0], p_2, p_2[indices[0][0]]

# ---------------------
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
    img_2 = Image.open('ILSVRC/Data/DET/test/'+img[0])
    if img_2.mode == 'RGB':
        return True
    return False

# ---------------------
#Calling the functions

df_s = df.head(20)
df_s = df_s[df_s.apply(filter_color_images, axis=1)]

resnet_preds = df_s.apply(
    resnet_model, axis=1, result_type="expand").rename(
        columns={0: "resnet_indices", 1: "resnet_percentage", 2: "resnet_max_prob"}) 

# I used pandas notation here, but the value inside the join would be a WHERE sql query
# Might want to explore query optimization here? (Between where/join/apply)
inception_preds = df_s.join(
    resnet_preds[resnet_preds['resnet_max_prob'] < 85], how='right').apply(
        inceptionv3_model, axis=1, result_type="expand").rename(
            columns={0: "inception_indices", 1: "inception_percentage", 2:"inception_max_prob"}) 

all_preds = df_s.join([resnet_preds, inception_preds])
all_preds 


# ---------------------
# Calling the cascading function
cascade_df = all_preds.join(all_preds.apply(cascade_predict, axis=1, result_type="expand") \
                    .rename(columns={0: "cascade_indices", 1: "cascade_percentage", 2:"cascade_prediction"}))
print(cascade_df)

