import sys
import detect_face
import tensorflow as tf
from scipy import misc
import cv2
import numpy as np
import os
import datetime
import dlib
import scipy.misc

MINSIZE = 10 # minimum size of face
THRESHOLD = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
FACTOR = 0.709 # scale factor
BREAK_LINE = "\n\n---------------------------------------"
TOLERANCE = 0.54
6
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

encodings_database = []

def get_box(boxes, index):
	box = []
	for i in range(4):
		box.append(int(round(boxes[index][i])))
	return box

def draw_bounding_boxes(img, boxes, edge_thickness):
	for i in range(len(boxes)):
		box = get_box(boxes, i)
		
		#TODO: verify if edge is out of image range
		cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]),
			(0, 255, 0), edge_thickness )

def draw_facial_points(img, points, square_size):
	if len(points) > 0:
		#TODO: verify if points are out of image range when drawing
		left = -1 * (square_size // 2)
		right = square_size + left

		if (isinstance(points[0], dlib.point)):
			for i in range(len(points)):
				x = int(round(points[i].x))
				y = int(round(points[i].y))

				cv2.rectangle(img, (x + left, y + left), 
					(x + right, y + right), (0, 255, 0),
					thickness = cv2.FILLED)
		else:
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

def extract_face_features(img):
	#expects cropped image, i.e. a bounding-boxed face
	height, width, _ = img.shape
	rectangle = dlib.rectangle(0, 0, width, height)

	return shape_predictor(img, rectangle)

def compare_encodings(base, target):
	return np.linalg.norm(known_faces - face, axis=1)

def find_match(base, target):
	if len(base) > 0:
		matches = compare_encodings(base, target)

		min_elem =  np.amin(matches)
		if min_elem <= TOLERANCE:
			min_index, = np.argwhere(matches == min_elem)
			return min_index

	return -1


def time_now():
	d = datetime.datetime.now()
	ret = str(d.year)
	ret += "_" + str(d.month)
	ret += "_" + str(d.day)
	ret += "_" + str(d.hour)
	ret += "_" + str(d.minute)
	ret += "_" + str(d.second)
	mcrscnd = str(d.microsecond)
	ret += "_" + '0' * (6 - len(mcrscnd)) + mcrscnd
	return ret

def save_image(directory, person_id, img):
	path = os.path.join(directory, person_id)
	if not os.path.exists(path):
		os.makedirs(path)

	cv2.imwrite(os.path.join(path, time_now() +
		".jpg"), img)

def crop_image(img, bounding_box):
	return img[bounding_box[1]:bounding_box[3],
		bounding_box[0]:bounding_box[2]]

def make_noises():
	duration  = 75 # milliseconds
	for j in range(2):
		frequency = 1000 # Hertz
		for i in range(7):
			frequency -= 100
			os.system('play -n synth %s sin %s' % (duration/1000, frequency))
		for i in range(7):
			frequency += 100
			os.system('play -n synth %s sin %s' % (duration/1000, frequency))


if (len(sys.argv) == 4):
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
			labels = []

			#mtcnn is more accurate with rgb images
			#for now, the points returned by mtcnn are not used
			boxes, points = detect_face.detect_face(
				cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), MINSIZE,
				pnet, rnet, onet, THRESHOLD, FACTOR)

			#processes each face detected
			if (len(boxes) > 0):
				for i in range(len(boxes)):
					crop = crop_image(frame, get_box(boxes, i))
					

					feats = extract_face_features(crop)

					encodings = np.array(
						face_recognition_model.compute_face_descriptor(
						crop, feats, 1))

					match_index = find_match(encodings_database, encodings)

					if match_index == -1:
						#new persona detected. Add to database and
						#save its image
						encodings_database.append(encodings)

						match_index = len(encodings_database) - 1
						save_image(sys.argv[3], "p" + str(
							match_index), crop)

					labels.append('p' + str(match_index))
					draw_facial_points(crop, feats.parts(), 3)
					'''
					height, width, _ = crop.shape
					face_box = dlib.rectangle(0, 0, width, height)

					face_traits = dlib.points()
					len_points = len(points) // 2
					for j in range(len_points):
						face_traits.append(dlib.point( round(float(points[j][i])) ,
							round(float(points[j + len_points][i]))))
					shapes_points = dlib.full_object_detection(face_box, face_traits)

					encodings = np.array(
						face_recognition_model.compute_face_descriptor(
						crop, shapes_points, 1))

					match_index = find_match(encodings_database, encodings)

					if match_index == -1:
						#new persona detected. Add to database and
						#save its image
						encodings_database.append(encodings)

						match_index = len(encodings_database) - 1
						save_image(sys.argv[3], "p" + str(
							match_index), crop)

					labels.append('p' + str(match_index))
					#draw_facial_points(crop, points, 3)
					'''

			draw_bounding_boxes(frame, boxes, 3)
			draw_facial_points(frame, points, 3)
			draw_label(frame, boxes, labels)

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

	make_noises()

else:
	print(BREAK_LINE)
	print("ERROR! 3 arguments expected:\n" + 
		"\t1 - Video filepath" +
		"\n\t2 - Output filepath" +
		"\n\t3 - Image database directory")
