from ultralytics import YOLO 
from PIL import Image
import cv2
import numpy as np

model = YOLO('runs/detect/train7/weights/best.pt')

im = Image.open('cropped_enhanced/Bild9.png')

results = model.predict(source=im, imgsz=640, save=True)  #set imgsz to 640 to match training size
image = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


detection_results = results[0]


boxes = detection_results.boxes  #contain all detected bounding boxes
scores = boxes.conf  #confidence scores for each detected object
class_ids = boxes.cls  #class indices for each detected object


class_names = detection_results.names #access the class names

#annotate the image with bounding boxes and labels
for box, score, class_id in zip(boxes.xyxy, scores, class_ids):
    class_name = class_names[int(class_id)]  #get class name from class index
    x1, y1, x2, y2 = map(int, box)  #convert box coordinates to integers
    label = f'{class_name} {score:.2f}'  #create label with class name and confidence score
    
    #bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  #red bounding box
    
    #label above the bounding box
    (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image, (x1, y1 - label_height - baseline), (x1 + label_width, y1), (0, 0, 255), -1)  #red background for text
    cv2.putText(image, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  #white ;abel text


cv2.imshow('Annotated Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


print("Detected objects with confidence scores:")
for box, score, class_id in zip(boxes.xyxy, scores, class_ids):
    class_name = class_names[int(class_id)]  #get class name from class index
    print(f"Class: {class_name}, Confidence: {score:.4f}, Box: {box}")

