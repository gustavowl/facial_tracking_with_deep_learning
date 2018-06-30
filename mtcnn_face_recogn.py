import dlib
import scipy.misc
import numpy as np
import os
import detect_face
import tensorflow as tf
from scipy import misc
import cv2


with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, None)



minsize = 10 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

# Get Face Detector from dlib
# This allows us to detect faces in images
face_detector = dlib.get_frontal_face_detector()
# Get Pose Predictor from dlib
# This allows us to detect landmark points in faces and understand the pose/angle of the face
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# Get the face recognition model
# This is what gives us the face encodings (numbers that identify the face of a particular person)
face_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
# This is the tolerance for face comparisons
# The lower the number - the stricter the comparison
# To avoid false matches, use lower value
# To avoid false negatives (i.e. faces of the same person doesn't match), use higher value
# 0.5-0.6 works well
TOLERANCE = 0.58

#print("\n\n\n\n\n\n\n")

def get_points(detected_faces, index):
    return int(round(detected_faces[index][0])), int(round(detected_faces[index][1])), int(round(detected_faces[index][2])), int(round(detected_faces[index][3]))


def draw_facial_points(img, points, square_size):
    if len(points) > 0:
        #TODO: verify if points are out of image range when drawing
        left = -1 * (square_size // 2)
        right = square_size + left

        maximum = 3 * 255
        stride = maximum / len(points)
        colour = 0
        
        for i in range(len(points)):
            x = int(round(points[i].x))
            y = int(round(points[i].y))

            c = int(round(colour))
            blue = min(255, c)
            c = max(0, c - 255)
            green = min(255, c)
            c = max(0, c - 255)
            red = min(255, c)

            cv2.rectangle(img, (x + left, y + left), 
                (x + right, y + right), (blue, green, red),
                thickness = cv2.FILLED)

            colour += stride

def substitute_points(dlib_points, mtcnn_points):
    #face left, face right, nose, mouth left, mouth right
    SUBSTITUTE = [0, 16, 30, 48, 54]
    
    new_points = dlib.points()
    len_mtcnn = len(mtcnn_points) // 2

    for i in range(len_mtcnn):
        dlib_points[SUBSTITUTE[i]] = dlib.point(
            round(float(mtcnn_points[i])),
            round(float(mtcnn_points[i + len_mtcnn])))

    return dlib_points

# This function will take an image and return its face encodings using the neural network
def get_face_encodings(path_to_image):
    # Load image using scipy
    #image = scipy.misc.imread(path_to_image)
    image = cv2.imread(path_to_image)
    # Detect faces using the face detector
    #detected_faces = face_detector(image, 1)
    detected_faces, points = detect_face.detect_face(cv2.cvtColor(
        image, cv2.COLOR_BGR2RGB), minsize, pnet, rnet, onet,
        threshold, factor)
    
    #print(str(path_to_image))
    #print(detected_faces)
    left, top, right, bottom = get_points(detected_faces, 0)
    face_box = dlib.rectangle(left, top, right, bottom)
    #d = dlib.rectangle(1,2,3,4)
    #print(str(path_to_image) + "----" + str(detected_faces[0]) + "----" + str(d))
    # Get pose/landmarks of those faces
    # Will be used as an input to the function that computes face encodings
    # This allows the neural network to be able to produce similar numbers
    #for faces of the same people, regardless of camera angle and/or face positioning in the image
    shapes_faces = [shape_predictor(image, face_box)]
    mixed_points = substitute_points(shapes_faces[0].parts(), points)
    shapes_faces = [dlib.full_object_detection(face_box, mixed_points)]

    #if ( str(path_to_image) == "pics/rakoruja.jpg"):
    #print("\n\n--------------------------------------------")
    #img = cv2.imread(path_to_image)
    draw_facial_points(image, shapes_faces[0].parts(), 3)
    cv2.imshow("asdf", image)
    cv2.waitKey(30)
    #input()
    #print(face_traits)
    #print(points[0][0])
    #print("path to image: " + str(path_to_image))
    '''print("shapes_faces: " + str(shapes_faces[0]))
    print(left)
    print("shapes_faces_bounding box: " + str(shapes_faces[0].rect))
    print("mtcnn bounding box: " + str(face_box))
    print("num_parts: " + str(shapes_faces[0].num_parts))
    print("points of parts: " + str(shapes_faces[0].parts()))
    print("DLIB")
    print(shapes_faces)
    print("MTCNN")
    print(shapes_points)'''
    #print("--------------------------------------------")
    # For every face detected, compute the face encodings
    return [np.array(face_recognition_model.compute_face_descriptor(image, face_pose, 1)) for face_pose in shapes_faces]

# This function takes a list of known faces
def compare_face_encodings(known_faces, face):
    # Finds the difference between each known face and the given face (that we are comparing)
    # Calculate norm for the differences with each known face
    # Return an array with True/Face values based on whether or not a known face matched with the given face
    # A match occurs when the (norm) difference between a known face and the given face is less than or equal to the TOLERANCE value
    return np.linalg.norm(known_faces - face, axis=1)

# This function returns the name of the person whose image matches with the given face (or 'Not Found')
# known_faces is a list of face encodings
# names is a list of the names of people (in the same order as the face encodings - to match the name with an encoding)
# face is the face we are looking for
def find_match(known_faces, names, face):
    # Call compare_face_encodings to get a list of True/False values indicating whether or not there's a match
    matches = compare_face_encodings(known_faces, face)
    print("\n")
    print(matches)
    min_elem =  np.amin(matches)
    if min_elem <= TOLERANCE:
        #return min_index
        min_index, = np.argwhere(matches == min_elem)
        return names[min_index[0]]
    else:
        #return -1
        return 'Not Found'    


# Get path to all the known images
# Filtering on .jpg extension - so this will only work with JPEG images ending with .jpg
image_filenames = filter(lambda x: x.endswith('.jpg'), os.listdir('img_base/'))
# Sort in alphabetical order
image_filenames = sorted(image_filenames)
# Get full paths to images
paths_to_images = ['img_base/' + x for x in image_filenames]
# List of face encodings we have
face_encodings = []
# Loop over images to get the encoding one by one
for path_to_image in paths_to_images:
    # Get face encodings from the image
    face_encodings_in_image = get_face_encodings(path_to_image)
    # Make sure there's exactly one face in the image
    if len(face_encodings_in_image) != 1:
        print("Please change image: " + path_to_image + " - it has " + str(len(face_encodings_in_image)) + " faces; it can only have one")
        exit()
    # Append the face encoding found in that image to the list of face encodings we have
    face_encodings.append(get_face_encodings(path_to_image)[0])



# Get path to all the test images
# Filtering on .jpg extension - so this will only work with JPEG images ending with .jpg
test_filenames = filter(lambda x: x.endswith('.jpg'), os.listdir('pics/'))
# Get full paths to test images
paths_to_test_images = ['pics/' + x for x in test_filenames]
# Get list of names of people by eliminating the .JPG extension from image filenames
names = [x[:-4] for x in image_filenames]
# Iterate over test images to find match one by one
for path_to_image in paths_to_test_images:
    # Get face encodings from the test image
    face_encodings_in_image = get_face_encodings(path_to_image)
    # Make sure there's exactly one face in the image
    if len(face_encodings_in_image) != 1:
        print("Please change image: " + path_to_image + " - it has " + str(len(face_encodings_in_image)) + " faces; it can only have one")
        exit()
    # Find match for the face encoding found in this test image
    match = find_match(face_encodings, names, face_encodings_in_image[0])
    # Print the path of test image and the corresponding match
    print(path_to_image, match)


print("----------------------")
print(image_filenames)
print("----------------------")
print(sorted(filter(lambda x: x.endswith('.jpg'), os.listdir('pics/'))))


#delete this
image = cv2.imread("pics/lotr.jpg")
# Detect faces using the face detector
#detected_faces = face_detector(image, 1)
detected_faces, points = detect_face.detect_face(cv2.cvtColor(
    image, cv2.COLOR_BGR2RGB), minsize, pnet, rnet, onet,
    threshold, factor)

print(str(path_to_image))
print(detected_faces)
left, top, right, bottom = get_points(detected_faces, 0)
face_box = dlib.rectangle(left, top, right, bottom)
#d = dlib.rectangle(1,2,3,4)
#print(str(path_to_image) + "----" + str(detected_faces[0]) + "----" + str(d))
# Get pose/landmarks of those faces
# Will be used as an input to the function that computes face encodings
# This allows the neural network to be able to produce similar numbers
#for faces of the same people, regardless of camera angle and/or face positioning in the image
shapes_faces = [shape_predictor(image, face_box)]

face_traits = dlib.points()
len_points = len(points) // 2
for i in range(len_points):
    face_traits.append(dlib.point( round(float(points[i])) ,
        round(float(points[i + len_points]))))
shapes_points = [dlib.full_object_detection(face_box, face_traits)]

#if ( str(path_to_image) == "pics/rakoruja.jpg"):
#print("\n\n--------------------------------------------")
#img = cv2.imread(path_to_image)
draw_facial_points(image, shapes_faces[0].parts(), 3)
cv2.imwrite(os.path.join("./", "OUTPUT-COLOURFUL" +
    ".jpg"), image)
cv2.waitKey(30)

for i in range(len(shapes_faces[0].parts())):
    print(str(i) + " - " + str(shapes_faces[0].parts()[i]))
