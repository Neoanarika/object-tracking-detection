from centroidtracker import CentroidTracker
import numpy as np
import argparse
import imutils
import time
import cv2
 
def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-c', '--config', required=True,
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
args = ap.parse_args()

classes = None
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]


ct = CentroidTracker()
(H, W) = (None, None)
 
# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNet(args.weights, args.config)
cap = cv2.VideoCapture(0)
# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")

while True:
	ok, image = cap.read()

	Width = image.shape[1]
	Height = image.shape[0]
	scale = 0.00392

	# create input blob 
	blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

	# set input blob for the network
	net.setInput(blob)


	# run inference through the network
	# and gather predictions from output layers
	detections = net.forward(get_output_layers(net))

	conf_threshold = 0.5
	nms_threshold = 0.4
	things = []
	people = []
	confidences_ppl = []
	confidences_things = []
	class_ids = []
	for out in detections:
	    for detection in out:
	        scores = detection[5:]
	        class_id = np.argmax(scores)
	        confidence = scores[class_id]
	        if confidence > 0.5:
	            center_x = int(detection[0] * Width)
	            center_y = int(detection[1] * Height)
	            w = int(detection[2] * Width)
	            h = int(detection[3] * Height)
	            x = center_x - w / 2
	            y = center_y - h / 2
	            if class_id == 0:
	            	confidences_ppl.append(float(confidence))
	            	people.append([round(x), round(y), round(w), round(h)])
	            else:
	            	confidences_things.append(float(confidence))
	            	things.append([round(x), round(y), round(w), round(h)])
	            	class_ids.append(class_id)
	# apply non-max suppression
	indices_t = cv2.dnn.NMSBoxes(things, confidences_things, conf_threshold, nms_threshold)
	indices_p = cv2.dnn.NMSBoxes(people, confidences_ppl, conf_threshold, nms_threshold)

	for i in indices_t:
	    i = i[0]
	    box = things[i]
	    x = box[0]
	    y = box[1]
	    w = box[2]
	    h = box[3]
	    cv2.rectangle(image,(round(x),round(y)), (round(x+w),round(y+h)), ( 255,0, 0), 2)
	    cv2.putText(image,classes[class_ids[i]], (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0, 255), 2)

	rects_f = []
	for i in indices_p:
	    i = i[0]
	    box = people[i]
	    x = box[0]
	    y = box[1]
	    w = box[2]
	    h = box[3]
	    rects_f.append((round(x),round(y),round(x+w),round(y+h)))
	    cv2.rectangle(image,(round(x),round(y)), (round(x+w),round(y+h)), ( 255,0, 0), 2)

	objects = ct.update(rects_f)

	# loop over the tracked objects
	for (objectID, centroid) in objects.items():
		# draw both the ID of the object and the centroid of the
		# object on the output frame
		text = "ID {}".format(objectID)
		if ct.disappeared[objectID] == 0:
			cv2.putText(image, text, (centroid[0] - 10, centroid[1] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			cv2.circle(image, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
	 	
	# show the output frame
	cv2.imshow("Frame", image)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
 
# do a bit of cleanup
cap.release()
cv2.destroyAllWindows()
