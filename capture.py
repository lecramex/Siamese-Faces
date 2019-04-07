import cv2
import dlib
import argparse
from tools import rect_to_bb
from tools import check_if_directory_exist

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="Path to the location to save user faces")
ap.add_argument("-v", "--video_path", type=str, default=None, help="Instead of use the webcam grab a video file")
args = vars(ap.parse_args())

nfaces = 0
face_path = args['output']

if args['video_path'] is None:
	cam = cv2.VideoCapture(0)
	if not cam.isOpened():
		print('[ERROR] Please assure you that your webcam is correctly installed')
		exit(-1)
else:
	if os.path.isfile(args['video_path']):
		cam = cv2.VideoCapture(args['video_path'])
	else:
		print('[ERROR] The video file does not exist, verify your path')
		exit(-1)

if not check_if_directory_exist(face_path):
	exit(-1)
else:
	# The location exist, check if there are any face inside
	nfaces = len(os.listdir(face_path))

# Grab the name or ID of the input faceRects
temp = face_path.split('/')
idUser = temp[len(temp)-1]

if face_path[len(face_path)-1] != '/':
	face_path += '/'

print('[INFO] To use the capture press \'s\' to save the last detected face in disk')
detector = dlib.get_frontal_face_detector()
while(True):

	ret, frame = cam.read()
	
	if not ret and (args['video_path'] is not None):
		print('[INFO] End of the video')
		break
	
	# Detect faces
	faceRects = detector(frame, 0)
	f = None
	for rect in faceRects:
		# Crop faces
		x, y, w, h = rect_to_bb(rect)
		try:
			f = cv2.resize(frame[y:y+h, x:x+w, :].copy(), (224, 224))
		except Exception as e:
			f = None
	
	
	cv2.imshow('video', frame)
	
	# Only show or update frame when a face is detected
	if f is not None:
		cv2.imshow('face', f)
	
	key = cv2.waitKey(1)
	
	if key == ord('q'):
		break
	# Only save the face when the key s is pressed and only one face was detected on video
	elif key == ord('s') and len(faceRects) == 1:
		if cv2.imwrite('{}{}_{}.png'.format(face_path, idUser, nfaces), f):
			nfaces += 1
			print('[INFO] Captured {} faces'.format(nfaces))
		
cam.release()
cv2.destroyAllWindows()


