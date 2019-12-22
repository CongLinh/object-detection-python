import cv2 as cv
import numpy as np

print("Hi Linh, welcome to object detection program")
net = cv.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    for line in f.readlines():
        classes.append(line.strip())
    # classes = [line.strip() for line in f.readlines()]
    
#print(classes)
print("==========================================================")
layer_names = net.getLayerNames()
#print(layer_names)
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
#print(output_layers)

image = cv.imread("ns.jpg")
image = cv.resize(image, None, fx=1, fy=1)
height, width, channels = image.shape

blob = cv.dnn.blobFromImage(image, 0.00392, (416, 416), (0,0,0), True, crop=False)
for b in blob:
    for n, img_blob in enumerate(b):
        pass
#        cv.imshow(str(n), img_blob)
#        print(img_blob)
#        print("..................................................")


net.setInput(blob)
outs = net.forward(output_layers)
#print(outs)


class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores= detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            
            cv.circle(image, (center_x, center_y), 2, (0, 0, 255), 2)
            
            x = int(center_x - w/2)
            y = int(center_y - h/2)
            
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 255), 5)
            
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
     
colors = np.random.uniform(0, 255, size=(len(classes), 3))

indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
print(indexes)
number_of_objs = len(boxes)
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = classes[class_ids[i]]
        print(label)
        color = colors[i]
        cv.rectangle(image, (x, y), (x+w, y+h), color, 1)
        cv.putText(image, label, (x+20, y+20), cv.FONT_HERSHEY_PLAIN, 3, color, 2)

cv.imshow("Hinh anh", image)
cv.waitKey(0)
cv.destroyAllWindows()















