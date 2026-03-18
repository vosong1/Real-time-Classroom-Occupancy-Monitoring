import cv2
import numpy as np

cfg = "configs/yolov3-classroom.cfg"
weights = "weights/yolov3-classroom.weights"
names = "configs/classes.names"

# load class names
with open(names) as f:
    classes = [line.strip() for line in f.readlines()]

# load model
net = cv2.dnn.readNetFromDarknet(cfg, weights)


layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# open webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1/255, (416,416), swapRB=True)
    net.setInput(blob)

    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for out in outputs:
        for detection in out:

            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:

                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                bw = int(detection[2] * w)
                bh = int(detection[3] * h)

                x = int(center_x - bw/2)
                y = int(center_y - bh/2)

                boxes.append([x,y,bw,bh])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in indexes.flatten():

        x,y,w,h = boxes[i]

        label = classes[class_ids[i]]

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,label,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    cv2.imshow("YOLOv3",frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()