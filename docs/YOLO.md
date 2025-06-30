## Deploy Realtime YOLO models

Adapted from [YOLOv8-TensorRT](https://github.com/triple-Mu/YOLOv8-TensorRT), we deploy realtime YOLO models via [Nividia TensorRT](https://developer.nvidia.com/tensorrt). 

### Preparation
You will need a `mp4` recording to create the training data for your YOLO model. 

### Roboflow Instructions
We first want to extract frames from our video using ffmpeg. Open a terminal in the folder where the video recording is and enter the following: 
```
mkdir frames
ffmpeg -i input.mp4 -vf "fps=5" frames/frame_%05d.jpg
```
This will extract every 5 frames of your video and put them in a folder called frames.

Next, we want to set up Roboflow. Visit [roboflow](https://app.roboflow.com/) and create an account. A public account is limited to one workspace, but that one workspace can have multiple projects. Create a workspace and a new project. 

You can upload the frames folder to Roboflow. This will put all the extracted frames into your project. Begin annotating and follow along with the Roboflow tutorial. When you've annotated enough images, back out and select the `Add to Dataset` button in the top right of the screen. Be sure to update the distribution of the pictures in the dropdown to ensure that the pictures will be split between training, testing, and validation datasets. 

Now in the Datasets tab, we can create a new dataset version. For the preprocessing steps, add a step that will resize the image to 640x640. For the augmentation step, select any augmentations you wish to have in your dataset. Create your new dataset and download it, making sure to select the `yolov8` format when prompted. 

Clone the repo: 
```
cd ~/src/
git clone https://github.com/triple-Mu/YOLOv8-TensorRT
cd YOLOv8-TensorRT
```

Create a venv and activate it:
```
python3 -m venv .venv
```

Now open the folder where you previously downloaded the YOLO dataset. 

Source the venv from earlier and install ultralytics:
```
source ~/src/YOLOv8-TensorRT/.venv/bin/activate
pip install ultralytics
```

Now we want to train the YOLO model. Download your preferred model size from the [yolov8 huggingface](https://huggingface.co/Ultralytics/YOLOv8/tree/main). Note that larger model sizes will result in higher latency in predictions. We can then train our model on our custom dataset: 
```
yolo task=detect mode=train model=yolov8m.pt data=data.yaml epochs=100 imgsz=640
```

This should give you a `.pt` file in `runs/detect/train/weights`. We want to use `best.pt`. Copy this file into the cloned repo from earlier: 

```
cd runs/detect/train/weights
cp ./best.pt ~/src/YOLOv8-TensorRT/best.pt
cd ~/src/YOLOv8-TensorRT/
```

Now we're ready to compile into an engine file. Install the requirements:
```
pip install -r requirements.txt
pip install tensorrt==10.11.0.33
```

Now we convert the `.pt` to `.onnx`:
```
python3 export-det.py \
--weights yolov8s.pt \
--iou-thres 0.65 \
--conf-thres 0.25 \
--topk 100 \
--opset 11 \
--sim \
--input-shape 1 3 640 640 \
--device cuda:0
```
Change the arguments to fit your use case. IOU treshold and confidence threshold can be adjusted to each setup. Topk sets the maximum amount of allowed bounding boxes to be drawn. 

Finally, use `trtexec` to compile `.onnx` to `.engine`:
```
~/nvidia/TensorRT/bin/trtexec --onnx=yolov8s.onnx --saveEngine=yolov8s.engine --device=0 --fp16
```

When running `trtexec`, the device used to compile the engine will be shown, as well as a list of devices available. Take note of which device your video output is connected to. You can change which device is used for compilation by changing the flag. 

We can move this `.engine` file into the default directory that orange looks at:
```
cp ./yolov8s.engine ~/orange_data/detect/yolov8s.engine
```

Finally, in the configuration file for the cameras, located in `~/orange_data/config/local/<CONFIG_NAME>`, update all cameras' `gpu_id` in the json files for the cameras you want to use the model with to match the id of the GPU that you used for compilation. 