## Deploy Realtime YOLO models

Adapted from [YOLOv8-TensorRT](https://github.com/triple-Mu/YOLOv8-TensorRT), we deploy realtime YOLO models via [Nividia TensorRT](https://developer.nvidia.com/tensorrt). 

### Pipeline Description
You will need a trained YOLO model, e.g. `yolov8s.pt`. 

Clone the repo: 
```
git clone https://github.com/triple-Mu/YOLOv8-TensorRT
cd YOLOv8-TensorRT
```

Create a venv and activate it:
```
python3 -m venv .venv
source .venv/bin/activate
```

Install the requirements:
```
pip install -r requirements.txt
pip install ultralytics
pip install tensorrt==10.11.0.33
```

Now we convert the `.pt` to `.onnx`:
```
python3 export-det.py \
--weights yolov8s.pt \
--iou-thres 0.65 \
--conf-thres 0.25 \
--topk 10 \
--opset 11 \
--sim \
--input-shape 1 3 640 640 \
--device cuda:0
```
Change the arguments to fit your use case. 

Finally, use `trtexec` to compile `.onnx` to `.engine`:
```
~/nvidia/TensorRT/bin/trtexec --onnx=yolov8s.onnx --saveEngine=yolov8s.engine --fp16
```

We can move this `.engine` file into the default directory that orange looks at:
```
cp ./yolov8s.engine ~/orange_data/detect/yolov8s.engine
```

