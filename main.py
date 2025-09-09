from ultralytics import YOLO

model = YOLO("yolov8l.yaml")
model = YOLO("yolov8l.pt")


if __name__ == '__main__':
    results = model.train(data="config.yaml", epochs=150)
    results = model.val()
    results = model("bag1.jpg")
    success = YOLO("yolov8l.pt").export(format="onnx")
    #yolo detect train data = config.yaml model "yolov8n.pt" epcchs=1









