from ultralytics import YOLO
import yaml                
import torch
from PIL import Image
import os
import cv2
import time
def main():
    data_config = 'dataset.yaml'
   
    model = YOLO('yolov8.yaml').load('yolov8n.pt')  #build from yaml and transfer weights

    model.tune(data=data_config, epochs=1, iterations=1, optimizer="AdamW", plots=False, save=False, val=False)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()