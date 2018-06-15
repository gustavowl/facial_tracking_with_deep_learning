import sys
import detect_face
import tensorflow as tf
from scipy import misc
from PIL import Image


MINSIZE = 10 # minimum size of face
THRESHOLD = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
FACTOR = 0.709 # scale factor

BREAK_LINE = "\n\n---------------------------------------\n\n"

def draw_bounding_box(img, points):
	print(BREAK_LINE)
	print("TODO")
	print(BREAK_LINE)

def draw_facial_points(img, points, square_size):
	print("---------")
	print(points[0])
	print(len(points))
	print(len(points[0]))
	mask_left = -1 * (square_size // 2)
	mask_right = square_size - mask_left
	pixels = img.load()
	
	for i in range(len(points[0])):
		for j in range(len(points) // 2):
			for x in range(mask_left, mask_right):
				for y in range(mask_left, mask_right):
					pixels[x + round(float(points[j][i])),
					y + round(float(points[j + 5][i]))] = (0, 255, 0)
	
	return img


if (len(sys.argv) == 2):
	with tf.Graph().as_default():
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
		sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
		with sess.as_default():
			pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

	print(BREAK_LINE)

	filepath = sys.argv[1]
	img = misc.imread(filepath)

	total_boxes, points = detect_face.detect_face(img, MINSIZE, pnet, rnet, onet,
		THRESHOLD, FACTOR)

	print("\n\n\n\nEITA\n\n\n\n")
	print(total_boxes)
	print(len(total_boxes))
	print("\n\n")
	print(points)

	pilimg = Image.open(filepath)
	pilimg = draw_facial_points(pilimg, points)
	pilimg.show()

else:
	print(BREAK_LINE)
	print("ERROR! 1 argument expected:\n" + 
		"\t1 - Image filepath")