import os

import microviewer
import numpy as np
import simplejpeg
import fastmorph
import scipy
import fill_voids
import cc3d

HOME = os.environ["HOME"]
IMAGE_DIR = os.path.join(HOME, "code/em-fault-detection/faulty_images/ok")

def extract_tissue_roi(filename):
	"""works with non-defective samples"""
	with open(filename, "rb") as f:
		binary = f.read()

	image = simplejpeg.decode_jpeg(binary, colorspace="GRAY")[:,:,0]
	orig = np.copy(image)


	# exclude surrounding bg
	roi = (image > 50)
	# handle thin cracks
	roi = scipy.ndimage.binary_closing(roi, structure=np.ones([3,3], dtype=bool), iterations=3) 
	# close big holes
	roi = fill_voids.fill(roi, in_place=True) 

	image *= roi

	out = cc3d.connected_components(image, delta=3)
	out = cc3d.largest_k(out, k=1)
	out = fill_voids.fill(out, in_place=True)

	image *= out

	out = cc3d.connected_components(image, delta=1)

	out = cc3d.dust(out, threshold=50)

	out = fastmorph.fill_holes(out, remove_enclosed=True)
	out = cc3d.largest_k(out, k=3)

	roi_center = cc3d.statistics(roi)["centroids"][0]
	roi_center = np.atleast_2d(roi_center)

	stats = cc3d.statistics(out)
	centroids = stats["centroids"][1:] # exclude bg 0

	distances = scipy.spatial.distance.cdist(roi_center, centroids)
	tissue_id = np.argmax(distances)
	out[out == tissue_id] = 0

	for i in range(3):
		out = fastmorph.dilate(out)

	image[out > 0] = 0
	mask = cc3d.largest_k(image > 0, k=1)

	microviewer.hyperview(orig, mask)

ok_images = os.listdir(IMAGE_DIR)

for filename in ok_images[2:3]:
	filename = os.path.join(IMAGE_DIR, filename)
	extract_tissue_roi(filename)



