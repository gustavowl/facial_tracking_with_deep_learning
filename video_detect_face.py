import sys
import detect_face
import tensorflow as tf
from scipy import misc
import cv2
import numpy as np


MINSIZE = 10 # minimum size of face
THRESHOLD = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
FACTOR = 0.709 # scale factor

BREAK_LINE = "\n\n---------------------------------------\n\n"

def get_box(boxes, index):
	box = []
	for i in range(4):
		box.append(int(round(boxes[index][i])))
	return box


def draw_bounding_boxes(img, boxes, edge_width):
	for i in range(len(boxes)):
		box = get_box(boxes, i)
		
		#TODO: verify if edge is out of image range
		cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]),
			(0, 255, 0), edge_width )

def draw_facial_points(img, points, square_size):
	if len(points) > 0:
		#TODO: verify if points are out of image range when drawing
		left = -1 * (square_size // 2)
		right = square_size + left
		
		for i in range(len(points[0])):
			for j in range(len(points) // 2):
				x = int(round(points[j][i]))
				y = int(round(points[j + 5][i]))

				cv2.rectangle(img, (x + left, y + left), 
					(x + right, y + right), (0, 255, 0),
					thickness = cv2.FILLED)

def draw_label(img, boxes, labels):
	if (len(boxes) == len(labels)):
		for i in range(len(boxes)):
			box = get_box(boxes, i)

			#write label on top of bounding box
			bottom_left = (box[0], box[1] - 10)

			cv2.putText(img, labels[i], bottom_left, cv2.FONT_HERSHEY_SIMPLEX,
				0.8, (0, 255, 0), thickness = 2)

def get_face_encondings():



if (len(sys.argv) == 3):
	with tf.Graph().as_default():
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
		sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
			log_device_placement=False))
		with sess.as_default():
			pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

	capture = cv2.VideoCapture(sys.argv[1])


	width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = int(capture.get(cv2.CAP_PROP_FPS))

	fourcc = cv2.VideoWriter_fourcc(*'MJPG')
	out = cv2.VideoWriter(sys.argv[2], fourcc, fps, (width, height))

	while (capture.isOpened()):
		ret, frame = capture.read()

		if ret == True:
			boxes, points = detect_face.detect_face(frame, MINSIZE,
				pnet, rnet, onet, THRESHOLD, FACTOR)

			draw_bounding_boxes(frame, boxes, 3)
			draw_facial_points(frame, points, 3)
			draw_label(frame, boxes, ["P2112"] * len(boxes))

			out.write(frame)
			
			cv2.imshow(sys.argv[1], frame)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		
		else:
			break

	capture.release()
	out.release()

	#print("\n\nPress \'Enter\' to exit")
	#input()
	
	cv2.destroyAllWindows()

	import os

	duration  = 75 # milliseconds
	for j in range(2):
		frequency = 1000 # Hertz
		for i in range(7):
			frequency -= 100
			os.system('play -n synth %s sin %s' % (duration/1000, frequency))
		for i in range(7):
			frequency += 100
			os.system('play -n synth %s sin %s' % (duration/1000, frequency))

else:
	print(BREAK_LINE)
	print("ERROR! 2 arguments expected:\n" + 
		"\t1 - Video filepath" +
		"\n\t2 - Output filepath")
