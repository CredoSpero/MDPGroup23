import torch
import os 
from os import listdir
import json
from roboflow import Roboflow

count = 0
array_list = []

folder_dir = "C:\\Users\\yixua\\OneDrive\\Desktop\\Image Recognition\\yolov5\\test\\images"

model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:\\Users\\yixua\\OneDrive\\Desktop\\Image Recognition\\yolov5\\task2_best.pt') 

for images in os.listdir(folder_dir):
    count += 1
    if (images.endswith(".jpg")):
        predictions = model(folder_dir + "\\" + images, size = 640)
        predictions.save()
        
        boxes = predictions.pandas().xyxy[0]
        json_data = boxes.to_json(orient='records')
        array_list.append(json_data)
    if count == 3:
        break

jsonToReturn = json.dumps({"predictions": array_list})
with open('prediction.json', 'w') as json_file:
    json_file.write(jsonToReturn)


# predictions = model(folder_dir + "\\" + "20230830_171735_jpg.rf.e23b4d0522bf477ebce47c1ee5e672f5.jpg", size = 640)

# predictions.print()

# boxes = predictions.pandas().xyxy[0]
# print(boxes)

# json_data = boxes.to_json(orient='records')
# print(json_data)


