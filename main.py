import os
import argparse
import time
import torch
from torch.autograd import Variable
from PIL import Image
from utils.prepare_data import prepare_im_data
from models.yolov2 import Yolov2
from inference.yolo_eval import yolo_eval
from utils.visualize import draw_detection_boxes
from utils.network import WeightLoader


def main():
    parser = argparse.ArgumentParser(description="Inference YOLOv2 on a folder containing images.")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="Confidence threshold for confident predictions (default: 0.4).")
    args = parser.parse_args()
    images_dir = 'images'
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True) 

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print("No images found in directory:", images_dir)
        return

    print(f"Found {len(image_files)} images.")

    classes = ('aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')

    model = Yolov2()
    weight_loader = WeightLoader()
    weight_loader.load(model, 'pretrained/yolo-voc.weights')
    print('Model weights loaded.')

    if torch.cuda.is_available():
        model.cuda()
        print("Running on CUDA.")
    else:
        print("Running on CPU.")

    model.eval()
    print('Model loaded and ready.')

    for image_name in image_files:
        image_path = os.path.join(images_dir, image_name)
        print(f"Processing {image_name}...")

        
        img = Image.open(image_path)
        print(f"Original image mode: {img.mode}")  # Kiểm tra số kênh
        
        if img.mode != "RGB":
            img = img.convert("RGB")  # Chuyển đổi về RGB nếu có kênh Alpha
            print(f"Converted image mode: {img.mode}")
            
        im_data, im_info = prepare_im_data(img)

        if torch.cuda.is_available():
            im_data_variable = Variable(im_data).cuda()
        else:
            im_data_variable = Variable(im_data)

        tic = time.time()

        yolo_output = model(im_data_variable)
        yolo_output = [item[0].data for item in yolo_output]
        detections = yolo_eval(yolo_output, im_info, conf_threshold=args.conf, nms_threshold=0.4)

        toc = time.time()
        cost_time = toc - tic
        print(f"Detection completed in {cost_time:.4f} sec, FPS: {int(1 / cost_time)}")

        if len(detections) > 0:
            det_boxes = detections[:, :5].cpu().numpy()
            det_classes = detections[:, -1].long().cpu().numpy()
            im2show = draw_detection_boxes(img, det_boxes, det_classes, class_names=classes)

            output_path = os.path.join(output_dir, image_name)
            im2show.save(output_path)
            print(f"Saved detection result to {output_path}")
        else:
            print(f"No objects detected in {image_name}. Saving original image...")
            output_path = os.path.join(output_dir, image_name)
            img.save(output_path)  
            print(f"Saved original image to {output_path}")

if __name__ == '__main__':
    main()

