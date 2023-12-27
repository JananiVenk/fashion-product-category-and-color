from fastapi import FastAPI, File, UploadFile, Request
import uvicorn
import shutil
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import torch
import pickle
import cv2
from PIL import Image
import numpy as np
import os
from io import BytesIO
from torchvision.transforms import transforms
import torch.nn.functional as F
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import fast_colorthief

app = FastAPI()
templates = Jinja2Templates(directory="templates")

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 =torch.nn.Conv2d(3, 6, 5)
        self.pool =torch.nn.MaxPool2d(2, 2)
        self.conv2 =torch.nn.Conv2d(6, 16, 5) 
        self.fc1 =torch.nn.Linear(3264, 120)
        self.fc2 =torch.nn.Linear(120, 84)
        self.fc3 =torch.nn.Linear(84, 7)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,3264)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
class Net1(torch.nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 =torch.nn.Conv2d(3, 6, 5)
        self.pool =torch.nn.MaxPool2d(2, 2)
        self.conv2 =torch.nn.Conv2d(6, 16, 5) 
        self.fc1 =torch.nn.Linear(3264, 120)
        self.fc2 =torch.nn.Linear(120, 84)
        self.fc3 =torch.nn.Linear(84, 45)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,3264)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
model_path = os.path.join(os.path.dirname(__file__), 'category.pth')
net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
net.eval()

subnet = Net1()
model_path_1 = os.path.join(os.path.dirname(__file__), 'subcategory.pth')
subnet.load_state_dict(torch.load(model_path_1, map_location=torch.device('cpu')))
subnet.eval()
path=os.path.join(os.path.dirname(__file__), 'color.pkl')
with open(path, "rb") as f:
    clf = pickle.load(f)

@app.get("/", response_class=HTMLResponse)
async def upload(request: Request):
   return templates.TemplateResponse("uploadfile.html", {"request": request})

classes = ('Apparel','Accessories','Footwear','Personal Care','Free Items','Sporting Goods','Home')
subclasses=('Topwear', 'Bottomwear', 'Watches', 'Socks', 'Shoes', 'Belts',
       'Flip Flops', 'Bags', 'Innerwear', 'Sandal', 'Shoe Accessories',
       'Fragrance', 'Jewellery', 'Lips', 'Saree', 'Eyewear', 'Nails',
       'Scarves', 'Dress', 'Loungewear and Nightwear', 'Wallets',
       'Apparel Set', 'Headwear', 'Mufflers', 'Skin Care', 'Makeup',
       'Free Gifts', 'Ties', 'Accessories', 'Skin', 'Beauty Accessories',
       'Water Bottle', 'Eyes', 'Bath and Body', 'Gloves',
       'Sports Accessories', 'Cufflinks', 'Sports Equipment', 'Stoles',
       'Hair', 'Perfumes', 'Home Furnishing', 'Umbrellas', 'Wristbands',
       'Vouchers')
def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    transform = transforms.Compose([transforms.Resize(size=(80,60)),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    pred=net(transform(image).unsqueeze(0))
    _, predicted = torch.max(pred.data, 1)
    subpred=subnet(transform(image).unsqueeze(0))
    _, subpredicted = torch.max(subpred.data, 1)
    img = np.array(image)
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners=cv2.goodFeaturesToTrack(gray,20,0.06,7,useHarrisDetector=True,k=0.03)
    corners = np.int0(corners)
    x1=[x[0][0] for x in corners]
    y1=[y[0][1] for y in corners]
    xmax,ymax,xmin,ymin=max(x1),max(y1),min(x1),min(y1)
    crop=hsv[ymin:ymax,xmin:xmax]
    alpha_channel = np.ones((crop.shape[0], crop.shape[1], 1), dtype=np.uint8) * 255
    rgba = np.concatenate((crop, alpha_channel), axis=2)
    dominant_color = fast_colorthief.get_dominant_color(rgba,quality=2)
    if dominant_color[0]<100 and dominant_color[1]<80 and dominant_color[2]>220:
        dominant_color = fast_colorthief.get_palette(rgba,quality=2)[1]
    c=clf.predict([dominant_color])
    data={'category':classes[predicted],'subcategory':subclasses[subpredicted],'color':c[0]}
    return data

