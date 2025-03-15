# YOLOV2  -  You Only Look Once 

## Description
This is a project to run inference of pretrained YOLOV2 models on pytorch.

![Image](https://github.com/user-attachments/assets/92f6b3a7-ead5-45f5-9f36-d8590fef9806)
## 🚀 Quick Start

### Requirements
I recommend you to use python >= 3.9 to run project.

### **1️⃣ Clone the Project**

Clone with HTTPS
```bash
  git clone https://github.com/kendyle2702/yolov2-inference.git
  cd yolov2-inference
```
Clone with SSH
```bash
  git clone git@github.com:kendyle2702/yolov2-inference.git
  cd yolov2-inference
```

### **2️⃣ Install Library**
```bash
  pip install -r requirements.txt
```

### **3️⃣ Download Pretrained Model**

Download YOLOv2 [pretrained weight](http://pjreddie.com/media/files/yolo-voc.weights). 

Move pretrained weight to ```pretrained``` folder. 


### **4️⃣ Run Inference**
Make sure you have put the images you need to inference into the ```images``` folder.
```bash
  python main.py --conf {default 0.4}
```
The image inference results will be in the ```results``` folder.
