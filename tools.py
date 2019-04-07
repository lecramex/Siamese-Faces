import dlib
import os

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y

	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

# Check if a directory exist if not it will try to create if can complete the operation returns True and False otherwise
def check_if_directory_exist(path):
	if not os.path.exists(path):
		print('[INFO] Output folder does not exist. Creating directory {}...'.format(path))
		try:
			os.mkdir(path)
		except OSError as ecpErr:
			print('[ERROR] Could not create folder {}: {}'.format(path, ecpErr))
			return False
		else:
			print('[INFO] Folder created successfully')
	return True


