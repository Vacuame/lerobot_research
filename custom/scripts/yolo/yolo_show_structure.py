from ultralytics import YOLO
import cv2
import numpy as np

if __name__ == "__main__":
    model = YOLO("yolo11n-seg.pt")  # load a pretrained model
    for name, param in model.named_parameters():
        print(name)