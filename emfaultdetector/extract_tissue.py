import os

import microviewer
import numpy as np
import simplejpeg
import fastmorph
import scipy
import fill_voids
import cc3d
import fastremap
import cv2 as cv

HOME = os.environ["HOME"]
IMAGE_DIR = os.path.join(HOME, "code/em-fault-detection/faulty_images/ok")

def calculate_moments_of_inertia(binary_image):
    # Calculate moments of the binary image
	centroid = cc3d.statistics(binary_image)["centroids"][0]
	centroid_x, centroid_y = tuple(centroid)

	rows, cols = np.indices(binary_image.shape)

	# Calculate second moments (moments of inertia)
	Ixx = np.sum((rows - centroid_y)**2 * binary_image)
	Iyy = np.sum((cols - centroid_x)**2 * binary_image)
	Ixy = -np.sum((rows - centroid_y) * (cols - centroid_x) * binary_image)

	# Inertia tensor
	inertia_tensor = np.array([[Ixx, Ixy],
	                           [Ixy, Iyy]])

	return inertia_tensor, (centroid_x, centroid_y)

def compute_principal_axes(inertia_tensor):
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(inertia_tensor)

    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # The eigenvectors are the principal axes
    principal_axes = eigenvectors

    return principal_axes, eigenvalues

def get_axes(binary_image):
	tensor, origin = calculate_moments_of_inertia(binary_image)
	axes, eigenvalues = compute_principal_axes(tensor)
	return origin, axes

def draw_axis(image, origin, ax):
	axis_length = 200
	p1 = tuple([ int(x) for x in origin ])
	p2 = tuple([ int(x) for x in (origin + ax * axis_length) ])
	cv.line(image, p1, p2, (0, 0, 255), 2)

def extract_tissue_roi(filename):
	"""works with non-defective samples"""
	with open(filename, "rb") as f:
		binary = f.read()

	image = simplejpeg.decode_jpeg(binary, colorspace="GRAY")[:,:,0]
	clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
	image = clahe.apply(image)

	orig = np.copy(image)

	# exclude surrounding bg
	roi = (image > 50)
	# handle thin cracks
	roi = scipy.ndimage.binary_closing(roi, structure=np.ones([3,3], dtype=bool), iterations=3) 
	# close big holes
	roi = fill_voids.fill(roi, in_place=True) 

	image *= roi

	# find tape and cut it off
	out = cc3d.connected_components(image, delta=3)
	out = cc3d.largest_k(out, k=1)
	out = fill_voids.fill(out, in_place=True)
	image *= out

	# segment tissue and resin
	out = cc3d.connected_components(image, delta=1)
	out = cc3d.dust(out, threshold=50)
	out = fastmorph.fill_holes(out, remove_enclosed=True)
	out = cc3d.largest_k(out, k=3)

	# estimate tissue label based on centrality to roi
	# and remove it (tissue label is often an underestimate
	# while the resin segmentation is often better)
	roi_center = cc3d.statistics(roi)["centroids"][0]
	roi_center = np.atleast_2d(roi_center)

	stats = cc3d.statistics(out)
	centroids = stats["centroids"][1:] # exclude bg 0

	distances = scipy.spatial.distance.cdist(roi_center, centroids)
	tissue_id = np.argmax(distances)

	# remove the tissue ROI so we can get
	# a better segmentation by excluding resin
	out[out == tissue_id] = 0

	# fill out the resin a bit better
	for i in range(3):
		out = fastmorph.dilate(out)

	# remove the resin and obtain a more precise
	# tissue label
	image[out > 0] = 0
	mask = cc3d.largest_k(image > 0, k=1)

	microviewer.hyperview(orig, mask)

ok_images = os.listdir(IMAGE_DIR)

for filename in ok_images[2:3]:
	filename = os.path.join(IMAGE_DIR, filename)
	extract_tissue_roi(filename)



