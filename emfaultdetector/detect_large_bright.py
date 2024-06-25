import os

import microviewer
import numpy as np
import simplejpeg
import fastmorph
import scipy
import fill_voids
import cc3d
import time

import cv2 as cv

HOME = os.environ["HOME"]
IMAGE_DIR = os.path.join(HOME, "code/em-fault-detection/faulty_images/ok")

# stds = []

def detect(filename):
	"""works with non-defective samples"""
	with open(filename, "rb") as f:
		binary = f.read()

	image = simplejpeg.decode_jpeg(binary, colorspace="GRAY")[:,:,0]
	clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
	image = clahe.apply(image)

	# exclude surrounding bg
	roi = (image > 50)
	image *= roi

	delta = 8
	out = cc3d.connected_components(image, delta=delta)
	out = cc3d.dust(out, threshold=500)

	for label, region in cc3d.each(out, binary=True, in_place=True):
		vals = image[region]
		mean = np.mean(vals)
		std = np.std(vals)
		if mean > 195 and std < 15:
			stds.append(std)
			return True

	return False

image_paths = os.listdir(IMAGE_DIR)

for filename in image_paths:
	filename = os.path.join(IMAGE_DIR, filename)
	s = time.time()
	result = detect(filename)
	elapsed = time.time() - s
	print(f"{filename}: {result}, {elapsed:.2f} sec")


# print([ float(x) for x in stds ])
# print(np.mean(stds))

