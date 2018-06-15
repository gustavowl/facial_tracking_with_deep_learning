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

def draw_bounding_boxes(img, boxes, edge_width):
	for i in range(len(boxes)):
		box = []
		for j in range(4):
			box.append(int(round(boxes[i][j])))
		
		#TODO: verify if edge is out of image range
		cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]),
			(0, 255, 0), edge_width )

def draw_facial_points(img, points, square_size):
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

if (len(sys.argv) == 3):
	capture = cv2.VideoCapture(sys.argv[2])

	while (capture.isOpened()):
		ret, frame = capture.read()

		if ret == True:
			cv2.line(frame, (0, 0), (50, 50), (0, 0, 255), 5)
			cv2.imshow(sys.argv[2], frame)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		
		else:
			break

	capture.release()

	print("\n\nPress \'Enter\' to exit")
	input()
	
	cv2.destroyAllWindows()

	with tf.Graph().as_default():
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
		sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
			log_device_placement=False))
		with sess.as_default():
			pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

	img = cv2.imread(sys.argv[1])

	total_boxes, points = detect_face.detect_face(img, MINSIZE, pnet, rnet, onet,
		THRESHOLD, FACTOR)

	draw_bounding_boxes(img, total_boxes, 3)
	draw_facial_points(img, points, 5)

	cv2.imshow('dps', img)
	print("\n\nPress \'Any Key\' to exit")
	cv2.waitKey(0)

else:
	print(BREAK_LINE)
	print("ERROR! 2 arguments expected:\n" + 
		"\t1 - Image filepath" +
		"\t2 - Video filepath")