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
DEBUG = True
debug_encodings = []

shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

encodings_database = []
encodings_database_position = []

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

def compare_encodings(target):
	return np.linalg.norm(encodings_database - target, axis=1)

def find_match(target, target_bounding_box, frame_dimensions):
	target_position = get_mean_point(target_bounding_box)
	
	if len(encodings_database) > 0 and len(encodings_database) == len(encodings_database_position):
		farthest_vertex = get_farthest_vertex_from_point(target_position,
			frame_dimensions)
		max_euclid_dist = get_euclidian_distance(target_position, farthest_vertex)

		encodings = compare_encodings(target)

		#compare_encodigs_with_distance
		for i in range(len(encodings_database_position)):
			distance = get_euclidian_distance(target_position,
				encodings_database_position[i])
			dist_prop = get_distance_proportion(max_euclid_dist, distance)
			encodings[i] = encodings[i] / dist_prop

		if DEBUG:
			del debug_encodings[:]
			for enc in encodings:
				debug_encodings.append(enc)

		min_elem =  np.amin(encodings)
		if min_elem <= TOLERANCE:
			min_index, = np.argwhere(encodings == min_elem)
			#updates position in database
			encodings_database_position[min_index[0]] = target_position
			return min_index, min_elem

	#new persona detected. Add to database
	encodings_database.append(target)
	encodings_database_position.append(target_position)
	return -1, 1

def get_mean_point(bounding_box):
	x = int(round( (bounding_box[0] + bounding_box[2]) / 2 ))
	y = int(round( (bounding_box[1] + bounding_box[3]) / 2 ))
	return dlib.point(x, y)

def get_farthest_vertex_from_point(point, dimensions):
	ret_point = dlib.point(0, 0)
	dist = get_euclidian_distance(point, ret_point)
	vertices = [dlib.point(0, dimensions.y), dlib.point(dimensions.x, 0),
		dlib.point(dimensions.x, dimensions.y)]

	for vertex in vertices:
		new_dist = get_euclidian_distance(point, vertex)
		if  new_dist > dist:
			dist = new_dist
			ret_point = vertex
	return ret_point

def get_euclidian_distance(point_a, point_b):
	x = (point_a.x - point_b.x)**2
	y = (point_a.y - point_b.y)**2
	return (x + y)**0.5

#does probability describe better?
def get_distance_proportion(farthest_distance, distance):
	return 1 - 0.999 / farthest_distance * distance


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
	dlib_frame_dim = dlib.point(width, height)

	fourcc = cv2.VideoWriter_fourcc(*'MJPG')

	if not os.path.exists(sys.argv[2][0:sys.argv[2].rfind('/')]):
		os.makedirs(sys.argv[2][0:sys.argv[2].rfind('/')])
	out = cv2.VideoWriter(sys.argv[2], fourcc, fps, (width, height))

	frame_count = 0

	debug_file = open("empty.txt", 'w')

	if DEBUG:
		debug_file = open(os.path.join(sys.argv[2][0:sys.argv[2].rfind('/')],
			"debug.txt"), 'a')


	while (capture.isOpened()):
		ret, frame = capture.read()

		if ret == True:
			
			if DEBUG:
				frame_count += 1

			labels = []

			#mtcnn is more accurate with rgb images
			#for now, the points returned by mtcnn are not used
			boxes, points = detect_face.detect_face(
				cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), MINSIZE,
				pnet, rnet, onet, THRESHOLD, FACTOR)

			#processes each face detected
			if (len(boxes) > 0):
				for i in range(len(boxes)):
					box = get_box(boxes, i)
					crop = crop_image(frame, box)

					feats = extract_face_features(crop)

					encodings = np.array(
						face_recognition_model.compute_face_descriptor(
						crop, feats, 1))

					match_index, match_value = find_match(
						encodings, box, dlib_frame_dim)

					if match_index == -1:
						#new persona detected. save its image

						match_index = len(encodings_database) - 1
						save_image(sys.argv[3], "p" + str(
							match_index), crop)

					#FIXME: Delete probability?
					labels.append('p' + str(match_index) + " (" +
						"%.3f" % match_value + ")")
					draw_facial_points(crop, feats.parts(), 3)

					if DEBUG and len(labels) > 0:
						#save debug info to file
						if len(labels) == 1:
							debug_file.write("---------------FRAME #" +
								str(frame_count) + "---------------\n")

						debug_file.write('p' + str(match_index) + ": " +
							str(debug_encodings) + "\n")


			draw_bounding_boxes(frame, boxes, 3)
			draw_facial_points(frame, points, 3)
			draw_label(frame, boxes, labels)

			if DEBUG and len(labels) > 0:
				debug_file.write('\n')

			out.write(frame)
			
			cv2.imshow(sys.argv[1], frame)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		
		else:
			break

	capture.release()
	out.release()
	
	cv2.destroyAllWindows()

	make_noises()

else:
	print(BREAK_LINE)
	print("ERROR! 3 arguments expected:\n" + 
		"\t1 - Video filepath" +
		"\n\t2 - Output filepath" +
		"\n\t3 - Image database directory")
