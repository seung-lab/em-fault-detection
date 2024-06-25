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

	delta = 6
	out = cc3d.connected_components(image, delta=delta)
	out = cc3d.dust(out, threshold=500)
	out = cc3d.largest_k(out, k=1) > 0
	out = fill_voids.fill(out, in_place=True)

	stats = cc3d.statistics(out)
	largest = np.max(stats["voxel_counts"][1:])
	bg = stats["voxel_counts"][0]

	image_fraction = 1 - (bg / image.size)
	region_fraction = largest / np.sum(roi)

	result = image_fraction > 0.25 and region_fraction > 0.85

	return result

image_paths = os.listdir(IMAGE_DIR)

for filename in image_paths:
	filename = os.path.join(IMAGE_DIR, filename)
	s = time.time()
	result = detect(filename)
	elapsed = time.time() - s
	print(f"{filename}: {result}, {elapsed:.2f} sec")


