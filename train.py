from ultralytics import YOLO
model = YOLO('yolov8l.pt')
if __name__ == '__main__':
    model.train(data='data.yaml', epochs=30, imgsz=640,workers=1,batch=5)
    model.val()  
