from roboflow import Roboflow
from ultralytics import YOLO
import glob
import matplotlib.pyplot as plt
import cv2

rf = Roboflow(api_key="enter_api") # enter your own api
dataset = rf.workspace("sa-elvht").project("stop-sign-quls2-zh2ko").version(1).download("yolov8")
#dataset https://universe.roboflow.com/yolov8stopsign/stop-sign-quls2

model = YOLO("yolov8n.pt")
result = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=100,  
    batch=16,       
    imgsz=640,       
    optimizer="AdamW" 
)

model = YOLO("runs/detect/train/weights/best.pt")

result2 = model.predict(
    source=f"{dataset.location}/test/images", 
    conf=0.70,
    save=True                          
)



pimg = glob.glob("runs/detect/predict/*.jpg")

for img_path in pimg[20:40]:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis("off")
    plt.show()