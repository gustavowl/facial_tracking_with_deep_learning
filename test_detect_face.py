import detect_face
import tensorflow as tf
from scipy import misc
from PIL import Image


with tf.Graph().as_default():
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
	with sess.as_default():
		pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

def paint_points(img, points):
	print("---------")
	print(points[0])
	print(len(points))
	print(len(points[0]))
	mask_left = -2
	mask_right = 3
	pixels = img.load()
	for i in range(len(points[0])):
		for j in range(len(points) // 2):
			for x in range(mask_left, mask_right):
				for y in range(mask_left, mask_right):
					pixels[x + round(float(points[j][i])),
					y + round(float(points[j + 5][i]))] = (0, 255, 0)
	return img


minsize = 10 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

path = "pics/guilardo_russas.jpg"
img = misc.imread(path)

total_boxes, points = detect_face.detect_face(img, minsize, pnet, rnet, onet,
	threshold, factor)

print("\n\n\n\nEITA\n\n\n\n")
print(total_boxes)
print(len(total_boxes))
print("\n\n")
print(points)

pilimg = Image.open(path)
pilimg = paint_points(pilimg, points)
pilimg.show()