from random import randint
import argparse
import numpy as np
import cv2
from numba import jit

def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, person,confidence, x, y, x_plus_w, y_plus_h):

    if class_id == 0:
    	label = "{} {}".format(classes[class_id],person)
    else:
    	label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def confidence_prune(outs):
	boxes = []
	confidences = []
	for out in outs:
	    for detection in out:
	        scores = detection[5:]
	        class_id = np.argmax(scores)
	        confidence = scores[class_id]
	        if confidence > 0.5 and class_id == 0:
	            center_x = int(detection[0] * Width)
	            center_y = int(detection[1] * Height)
	            w = int(detection[2] * Width)
	            h = int(detection[3] * Height)
	            x = center_x - w / 2
	            y = center_y - h / 2
	            confidences.append(float(confidence))
	            boxes.append((x, y, w, h))
	return boxes,confidences
if __name__ == '__main__':
	# handle command line arguments
	ap = argparse.ArgumentParser()
	ap.add_argument('-c', '--config', required=True,
	                help = 'path to yolo config file')
	ap.add_argument('-w', '--weights', required=True,
	                help = 'path to yolo pre-trained weights')
	ap.add_argument('-cl', '--classes', required=True,
	                help = 'path to text file containing class names')
	args = ap.parse_args()

	# read class names from text file
	classes = None
	with open(args.classes, 'r') as f:
	    classes = [line.strip() for line in f.readlines()]

	# generate different colors for different classes 
	colors = np.random.uniform(0, 255, size=(len(classes), 3))

	# read pre-trained model and config file
	net = cv2.dnn.readNet(args.weights, args.config)
	cap = cv2.VideoCapture(0)

	# read webcam
	frames = 1
	success = False
	while True:
		ok, image = cap.read()

		if frames %100 == 0 or not success:
			Width = image.shape[1]
			Height = image.shape[0]
			scale = 0.00392

			# create input blob 
			blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

			# set input blob for the network
			net.setInput(blob)

			# run inference through the network
			# and gather predictions from output layers
			outs = net.forward(get_output_layers(net))

			# initialization
			conf_threshold = 0.5
			nms_threshold = 0.4

			boxes,confidences = confidence_prune(outs)

			# apply non-max suppression
			indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

			# Create MultiTracker object
			multiTracker = cv2.MultiTracker_create()
			 
			# Initialize MultiTracker 
			for bbox in boxes:
				tracker = cv2.TrackerMOSSE_create()
				multiTracker.add(tracker, image, bbox)

		success, boxes = multiTracker.update(image)
		# draw tracked objects
		if success:
			for i, newbox in enumerate(boxes):
				p1 = (int(newbox[0]), int(newbox[1]))
				p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
				cv2.putText(image, " Person "+ str(i+1), (int(newbox[0])-10,int(newbox[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i],2)
				cv2.rectangle(image, p1, p2, colors[i], 2, 1)

			# display output image
			cv2.putText(image, " Number of people: "+ str(len(boxes)), (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)    
			cv2.imshow("object detection", image)
			frames +=1
		# wait until any key is pressed
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

# release resources
cap.release()
cv2.destroyAllWindows()
