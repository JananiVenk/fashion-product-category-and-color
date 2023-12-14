from fastapi import FastAPI, File, UploadFile, Request
import uvicorn
import shutil
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import torch
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
color2rgb={'Navy Blue': (0,0,128), 'Blue': (0,0,255), 'Silver': (245,245,245), 'Black': (45,45,45), 'Grey': (128,128,128), 'Green': (0,255,0), 'Purple': (128,0,128), 'White': (255,255,255), 'Beige':(245,245,220), 'Brown': (165,42,42),'Teal':(0,128,128), 'Pink':(255, 192, 203), 'Off White':(250, 249, 246), 'Maroon':(128,0,0), 'Red': (255,0,0), 'Khaki':(240, 230, 140), 'Orange': (255, 165, 0),'Yellow':(255, 255, 0), 'Charcoal': (85,85,85), 'Gold': (255, 215, 0), 'Steel': (113, 121, 126), 'Tan':(210, 180, 140), 'Magenta': (255, 0, 255), 'Lavender': (230, 230, 250),'Cream':(255, 253, 208), 'Peach': (255, 229, 180), 'Olive': (128, 128, 0), 'Skin': (232, 190, 172), 'Grey Melange': (204,202,201) }
color=pd.DataFrame(color2rgb)
color=color.T
color.reset_index(level=0, inplace=True) 
color.rename(columns = {'index':'color'}, inplace = True) 
y=color['color']
x=color.drop('color',axis=1)
clf=KNeighborsClassifier(n_neighbors=1)
clf.fit(x,y)
model = KMeans(n_clusters=3, random_state=42)

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
net = Net()
model_path = os.path.join(os.path.dirname(__file__), 'category.pth')
net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
net.eval()

@app.get("/", response_class=HTMLResponse)
async def upload(request: Request):
   return templates.TemplateResponse("uploadfile.html", {"request": request})

classes = ('Apparel','Accessories','Footwear','Personal Care','Free Items','Sporting Goods','Home')
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
    print(classes[predicted])
    img = np.array(image)
    rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners=cv2.goodFeaturesToTrack(gray,10,0.001,10,useHarrisDetector=True)
    corners = np.int0(corners)
    x1=[x[0][0] for x in corners]
    y1=[y[0][1] for y in corners]
    xmax,ymax,xmin,ymin=max(x1),max(y1),min(x1),min(y1)
    crop=rgb[ymin:ymax,xmin:xmax]
    image = crop.reshape((crop.shape[0] * crop.shape[1], 3))
    model.fit(image)
    labels=model.labels_
    labels=list(labels)
    centroid=model.cluster_centers_
    percent=[]
    maxperc=0
    for x in range(len(centroid)):
        j=labels.count(x)
        j=j/(len(labels))
        percent.append(j)
    perc=np.array(percent)
    sort_perc=np.sort(perc)[::-1]
    print(perc)
    if (sort_perc[0]-sort_perc[1])*100<3:
        res='multi'
    else:
        mperc=np.argmax(perc)
        y=centroid[mperc]
        res=clf.predict([y])
        res=res[0]
    print(res)
    data={'category':classes[predicted],'color':res}
    return data

